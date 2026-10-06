#include "pmi/ggml_device.hpp"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
  try { action(); } catch (const std::exception&) { return; }
  throw std::runtime_error("expected adapter rejection");
}
struct Directory {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("pmi-ggml-device-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  Directory() { std::filesystem::create_directories(path); }
  ~Directory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;

pmi::DevicePageRequest descriptor(ggml_type type, uint64_t width, uint64_t rows) {
  pmi::DevicePageRequest request;
  request.format = pmi::GgmlDeviceBackend::format;
  request.layout = pmi::GgmlDeviceBackend::layout;
  request.source.name = "one.selected.expert";
  request.source.ggml_type = static_cast<uint32_t>(type);
  request.source.shape = {width, rows};
  request.source.byte_length = pmi::ggml_tensor_nbytes(static_cast<uint32_t>(type), request.source.shape);
  return request;
}

std::vector<uint8_t> encode(ggml_type type, size_t width, size_t rows, float phase) {
  std::vector<float> values(width * rows);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = std::sin(static_cast<float>(i) * 0.31f + phase) * 0.75f;
  std::vector<uint8_t> bytes(ggml_row_size(type, static_cast<int64_t>(width)) * rows);
  if (type == GGML_TYPE_F32) std::memcpy(bytes.data(), values.data(), bytes.size());
  else require(ggml_quantize_chunk(type, values.data(), bytes.data(), 0, rows, width, nullptr) == bytes.size(),
               "fixture quantization length");
  return bytes;
}

// Run retained ggml kernels with a weight supplied by the cache. Only the tiny
// input/output graph buffers are separately allocated; no model or device
// discovery is involved. Compare against a scalar dequantized CPU reference.
double projection(pmi::GgmlDeviceBackend& adapter, const pmi::DeviceLease& lease,
                  const std::vector<uint8_t>& expected_bytes) {
  auto* weight = adapter.tensor(*lease);
  const size_t width = static_cast<size_t>(weight->ne[0]);
  const size_t rows = static_cast<size_t>(weight->ne[1]);
  std::vector<uint8_t> readback(expected_bytes.size());
  ggml_backend_tensor_get(weight, readback.data(), 0, readback.size());
  ggml_backend_synchronize(adapter.backend());
  require(readback == expected_bytes, "quantized payload changed during upload/readback");

  std::vector<float> input(width), dense(width * rows);
  // The pinned CPU kernels quantize F32 activations to Q8_0 for both quantized
  // fixture types. With 0/+127/-127 in every block, d = amax/127 = 1 is exactly
  // representable in the stored half scale. Using +/-1 would round d = 1/127
  // and invalidate a scalar reference against the original F32 activations.
  const auto* cpu_traits = ggml_get_type_traits_cpu(weight->type);
  require(cpu_traits && cpu_traits->vec_dot_type ==
      (weight->type == GGML_TYPE_F32 ? GGML_TYPE_F32 : GGML_TYPE_Q8_0),
      "fixture activation exactness assumes the pinned F32/Q8_0 dot input path");
  for (size_t i = 0; i < width; ++i) input[i] = i % 3 == 0 ? 0.0f : (i % 2 ? 127.0f : -127.0f);
  if (weight->type == GGML_TYPE_F32) std::memcpy(dense.data(), readback.data(), readback.size());
  else {
    const auto* traits = ggml_get_type_traits(weight->type);
    require(traits->to_float != nullptr, "fixture type lacks dequantizer");
    for (size_t row = 0; row < rows; ++row)
      traits->to_float(readback.data() + row * ggml_row_size(weight->type, static_cast<int64_t>(width)),
                       dense.data() + row * width, static_cast<int64_t>(width));
  }

  ggml_init_params params{};
  params.mem_size = 8 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false);
  params.no_alloc = true;
  Context context(ggml_init(params), ggml_free);
  require(context != nullptr, "fixture graph context");
  auto* x = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, static_cast<int64_t>(width), 1);
  auto* result = ggml_mul_mat(context.get(), weight, x);
  require(ggml_backend_supports_op(adapter.backend(), result), "CPU fixture matmul unsupported");
  auto* graph = ggml_new_graph_custom(context.get(), 8, false);
  ggml_build_forward_expand(graph, result);
  Buffer buffer(ggml_backend_alloc_ctx_tensors(context.get(), adapter.backend()), ggml_backend_buffer_free);
  require(buffer != nullptr, "fixture graph allocation");
  ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
  require(ggml_backend_graph_compute(adapter.backend(), graph) == GGML_STATUS_SUCCESS, "fixture matmul failed");
  std::vector<float> actual(rows);
  ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));
  ggml_backend_synchronize(adapter.backend());
  double max_error = 0;
  for (size_t row = 0; row < rows; ++row) {
    double reference = 0;
    for (size_t column = 0; column < width; ++column)
      reference += static_cast<double>(dense[row * width + column]) * input[column];
    const double error = std::abs(actual[row] - reference);
    if (!std::isfinite(actual[row]) || error > 1e-5 + 1e-5 * std::abs(reference)) {
      std::ostringstream message;
      message << std::setprecision(12) << "cached quantized matmul differs from scalar reference: type="
              << ggml_type_name(weight->type) << " row=" << row << " width=" << width
              << " actual=" << actual[row] << " reference=" << reference << " error=" << error
              << " tolerance=" << 1e-5 + 1e-5 * std::abs(reference);
      throw std::runtime_error(message.str());
    }
    max_error = std::max(max_error, error);
  }
  return max_error;
}

double exercise(const std::filesystem::path& directory, const pmi::GgmlBackendOwner& owner,
                ggml_type type, size_t width) {
  auto request = descriptor(type, width, 2);
  const auto first = encode(type, width, 2, 0.2f), second = encode(type, width, 2, 1.4f);
  const auto path = directory / (std::string(ggml_type_name(type)) + ".bin");
  {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(first.data()), static_cast<std::streamsize>(first.size()));
    file.write(reinterpret_cast<const char*>(second.data()), static_cast<std::streamsize>(second.size()));
    require(static_cast<bool>(file), "fixture write");
  }
  pmi::FilePageProvider provider(path, (first.size() + 63) / 64 * 64);
  auto adapter = std::make_shared<pmi::GgmlDeviceBackend>(owner);
  const uint64_t charge = adapter->allocation_bytes(request);
  require(charge >= first.size(), "buffer padding missing from charge");
  pmi::DeviceCache cache(provider, adapter, charge);
  auto a = cache.request(request);
  require(a->charged_bytes() == charge, "reserved buffer charge mismatch");
  require(cache.request(request).get() == a.get(), "device cache hit made another allocation");
  require(provider.stats().misses == 1 && provider.stats().hits == 0, "device hit touched host provider");
  auto other = request;
  other.source.absolute_offset = first.size();
  rejects([&] { cache.request(other); });
  require(provider.stats().io_bytes == first.size(), "capacity rejection performed host I/O");
  pmi::GgmlDeviceBackend foreign(owner);
  rejects([&] { foreign.tensor(*a); });
  double max_error = projection(*adapter, a, first);
  a.reset();
  auto b = cache.request(other);
  max_error = std::max(max_error, projection(*adapter, b, second));
  b.reset();
  a = cache.request(request);
  max_error = std::max(max_error, projection(*adapter, a, first));
  require(cache.stats().uploads == 3 && cache.stats().hits == 1 && cache.stats().evictions == 2,
          "device reload counters disagree");
  require(cache.stats().upload_bytes == 3 * first.size() && cache.stats().current_bytes == charge &&
          cache.stats().peak_bytes == charge, "device allocation accounting disagrees");
  require(provider.stats().misses == 3 && provider.stats().active_leases == 0,
          "upload retained the host page or failed to reload");

  const uint64_t before = provider.stats().io_bytes;
  auto invalid = request;
  invalid.format = "unknown";
  rejects([&] { cache.request(invalid); });
  invalid = request; invalid.layout = "strided";
  rejects([&] { cache.request(invalid); });
  invalid = request; invalid.source.shape.push_back(512);
  rejects([&] { cache.request(invalid); });
  invalid = request; invalid.source.byte_length--;
  rejects([&] { cache.request(invalid); });
  invalid = request; invalid.source.ggml_type = GGML_TYPE_COUNT;
  rejects([&] { cache.request(invalid); });
  invalid = request; invalid.source.shape[0] = 0;
  rejects([&] { cache.request(invalid); });
  invalid = request; invalid.source.shape = {UINT64_MAX, 2};
  rejects([&] { cache.request(invalid); });
  if (type != GGML_TYPE_F32) {
    invalid = request; invalid.source.shape[0]--;
    rejects([&] { cache.request(invalid); });
  }
  rejects([&] { adapter->allocate(request, charge + 1); });
  require(provider.stats().io_bytes == before && cache.stats().current_bytes == charge,
          "invalid layout performed I/O or damaged cached allocation");
  return max_error;
}

void lifetime(const std::filesystem::path& directory) {
  pmi::DeviceLease survivor;
  std::weak_ptr<ggml_backend> weak;
  ggml_tensor* saved = nullptr;
  const float values[] = {1, 2, 3, 4};
  const auto path = directory / "lifetime.bin";
  { std::ofstream file(path, std::ios::binary); file.write(reinterpret_cast<const char*>(values), sizeof(values)); }
  {
    // Explicit CPU initialization only. No backend registry/discovery call.
    pmi::GgmlBackendOwner owner(ggml_backend_cpu_init(), ggml_backend_free);
    require(owner != nullptr, "CPU backend init");
    ggml_backend_cpu_set_n_threads(owner.get(), 1);
    weak = owner;
    auto adapter = std::make_shared<pmi::GgmlDeviceBackend>(owner);
    pmi::FilePageProvider provider(path, 64);
    auto request = descriptor(GGML_TYPE_F32, 4, 1);
    pmi::DeviceCache cache(provider, adapter, adapter->allocation_bytes(request));
    survivor = cache.request(request);
    saved = adapter->tensor(*survivor);
  }
  require(!weak.expired(), "device lease lost backend lifetime");
  float readback[4]{};
  ggml_backend_tensor_get(saved, readback, 0, sizeof(readback));
  require(std::memcmp(values, readback, sizeof(values)) == 0, "lease lost allocation after cache destruction");
  survivor.reset();
  require(weak.expired(), "last lease leaked backend ownership");
}
} // namespace

int main() {
  try {
    Directory directory;
    pmi::GgmlBackendOwner owner(ggml_backend_cpu_init(), ggml_backend_free);
    require(owner != nullptr, "CPU backend init");
    ggml_backend_cpu_set_n_threads(owner.get(), 1);
    double max_error = 0;
    max_error = std::max(max_error, exercise(directory.path, owner, GGML_TYPE_F32, 4));
    max_error = std::max(max_error, exercise(directory.path, owner, GGML_TYPE_Q8_0, 32));
    max_error = std::max(max_error, exercise(directory.path, owner, GGML_TYPE_IQ4_NL, 160));
    lifetime(directory.path);
    rejects([&] { pmi::GgmlDeviceBackend missing(nullptr); });
    std::cout << "ggml device adapter CPU fixtures passed: F32/Q8_0/IQ4_NL, 9 matmuls, 18 values; max_abs_error="
              << max_error << "; upload/readback, hit/eviction/reload, validation, lifetime\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
