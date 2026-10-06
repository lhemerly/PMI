#include "pmi/device_projection.hpp"
#include "pmi/ggml_device.hpp"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class F> void rejects(F f) {
  try { f(); } catch (const std::exception&) { return; }
  throw std::runtime_error("expected projection failure");
}
struct Directory {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("pmi-device-projection-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  Directory() { std::filesystem::create_directories(path); }
  ~Directory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};
struct Fixture {
  pmi::TensorInfo tensor;
  std::vector<uint8_t> bytes;
  std::vector<float> input, reference;
  std::filesystem::path path;
  Fixture(const std::filesystem::path& directory, ggml_type type, size_t width) {
    tensor.name = "compact.matrix";
    tensor.shape = {width, 2};
    tensor.ggml_type = static_cast<uint32_t>(type);
    tensor.byte_length = pmi::ggml_tensor_nbytes(tensor.ggml_type, tensor.shape);
    bytes.resize(static_cast<size_t>(tensor.byte_length));
    std::vector<float> dense(width * 2);
    for (size_t i = 0; i < dense.size(); ++i) dense[i] = 0.3f * std::sin(static_cast<float>(i) * 0.41f + 0.2f);
    if (type == GGML_TYPE_F32) std::memcpy(bytes.data(), dense.data(), bytes.size());
    else {
      check(ggml_quantize_chunk(type, dense.data(), bytes.data(), 0, 2, width, nullptr) == bytes.size(),
            "fixture quantization length");
      const auto* traits = ggml_get_type_traits(type);
      for (size_t row = 0; row < 2; ++row)
        traits->to_float(bytes.data() + row * ggml_row_size(type, width), dense.data() + row * width, width);
    }
    input.resize(width);
    // Q8_0 activation scales are exactly 1 for every block of 0/+127/-127.
    const auto* cpu = ggml_get_type_traits_cpu(type);
    check(cpu && cpu->vec_dot_type == (type == GGML_TYPE_F32 ? GGML_TYPE_F32 : GGML_TYPE_Q8_0),
          "fixture dot input type changed");
    for (size_t i = 0; i < width; ++i) input[i] = i % 3 == 0 ? 0 : (i % 2 ? 127.0f : -127.0f);
    reference.resize(2);
    for (size_t row = 0; row < 2; ++row) {
      double sum = 0;
      for (size_t column = 0; column < width; ++column) sum += double(dense[row * width + column]) * input[column];
      reference[row] = static_cast<float>(sum);
    }
    path = directory / (std::string(ggml_type_name(type)) + ".bin");
    write();
  }
  void write() const {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    check(static_cast<bool>(file), "fixture write failed");
  }
};

double compare(const std::vector<float>& actual, const Fixture& f) {
  check(actual.size() == f.reference.size(), "output shape");
  double maximum = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    const double error = std::abs(double(actual[i]) - f.reference[i]);
    if (!std::isfinite(actual[i]) || error > 1e-5 + 1e-5 * std::abs(f.reference[i]))
      throw std::runtime_error("projection parity: type=" + pmi::ggml_type_name(f.tensor.ggml_type) +
          " row=" + std::to_string(i) + " actual=" + std::to_string(actual[i]) +
          " reference=" + std::to_string(f.reference[i]) + " error=" + std::to_string(error));
    maximum = std::max(maximum, error);
  }
  return maximum;
}

double exercise(const Fixture& f, const std::shared_ptr<pmi::GgmlDeviceBackend>& adapter, bool profile) {
  pmi::FilePageProvider provider(f.path, (f.bytes.size() + 63) / 64 * 64);
  pmi::DevicePageRequest request{f.tensor, adapter->format, adapter->layout};
  const uint64_t charge = adapter->allocation_bytes(request);
  pmi::GgmlDeviceProjection executor(provider, adapter, charge, 8192, profile);
  double maximum = compare(executor.project(f.tensor, f.input), f);
  const auto host = provider.stats();
  maximum = std::max(maximum, compare(executor.project(f.tensor, f.input), f));
  check(provider.stats().io_bytes == host.io_bytes && provider.stats().hits == host.hits &&
        provider.stats().misses == host.misses, "device hit read or requested host weights");
  auto other = f.tensor;
  other.absolute_offset = f.bytes.size();
  maximum = std::max(maximum, compare(executor.project(other, f.input), f));
  maximum = std::max(maximum, compare(executor.project(f.tensor, f.input), f));
  const auto stats = executor.stats();
  check(stats.calls == 4 && stats.completed == 4 && stats.failures == 0, "projection call counters");
  check(stats.cache.hits == 1 && stats.cache.uploads == 3 && stats.cache.evictions == 2 &&
        stats.cache.provider_requests == 3, "device cache request counters");
  check(stats.cache.upload_bytes == 3 * f.bytes.size() && stats.cache.peak_bytes == charge &&
        stats.cache.active_leases == 0 && provider.stats().active_leases == 0, "weight accounting/lifetime");
  check(stats.graph_current_bytes == 0 && stats.graph_peak_bytes > 0 && stats.graph_peak_bytes <= 8192 &&
        stats.graph_capacity_bytes == 8192, "graph allocation accounting/lifetime");
  check(stats.activation_upload_bytes == 4 * f.input.size() * sizeof(float) &&
        stats.output_readback_bytes == 4 * 2 * sizeof(float), "activation/readback transfer accounting");
  check(stats.profile_enabled == profile, "profile flag");
  check(stats.cache.profile_enabled == profile, "cache service profile flag");
  if (!profile) {
    check(stats.cache.provider_read_ms == 0 && stats.cache.allocation_ms == 0 &&
          stats.cache.upload_ms == 0 && stats.cache.eviction_ms == 0, "disabled cache clocks");
  }
  const double times[] = {stats.cache_request_ms, stats.graph_setup_ms, stats.activation_upload_ms,
                         stats.kernel_ms, stats.output_readback_ms, stats.graph_teardown_ms};
  for (double time : times) check(profile ? time >= 0 && std::isfinite(time) : time == 0, "profile timing");
  executor.clear();
  check(executor.stats().cache.current_bytes == 0 && executor.stats().cache.evictions == 3 &&
        executor.stats().completed == 4, "clear retained allocation or reset counters");
  return maximum;
}

void resident_case(const Fixture& f, const std::shared_ptr<pmi::GgmlDeviceBackend>& adapter) {
  pmi::FilePageProvider provider(f.path,4096);
  const uint64_t charge = adapter->allocation_bytes({f.tensor,adapter->format,adapter->layout});
  pmi::GgmlDeviceProjection executor(provider,adapter,charge,8192,true);
  check(!executor.resident(f.tensor,false),"lookup admitted absent weight");
  auto lease = executor.resident(f.tensor,true); check(bool(lease),"seed failed");
  const auto before = executor.stats().cache;
  compare(executor.project_resident(lease,f.input),f);
  auto other = f.tensor; other.absolute_offset += f.bytes.size();
  check(!executor.resident(other,true) && !executor.resident(other,false),"seed evicted a live resident");
  check(executor.stats().cache.uploads == before.uploads && executor.stats().cache.provider_requests == before.provider_requests,
        "resident projection/missing lookup requested weights");
  rejects([&]{executor.project_resident({},f.input);});
  executor.clear(); check(executor.stats().cache.current_bytes == charge,"clear invalidated live resident lease");
  lease.reset(); executor.clear(); check(executor.stats().cache.current_bytes == 0,"resident lease retained storage");
}

void failures(const Fixture& f, const std::shared_ptr<pmi::GgmlDeviceBackend>& adapter) {
  pmi::FilePageProvider provider(f.path, 4096);
  pmi::GgmlDeviceProjection small(provider, adapter, 4096, 1);
  rejects([&] { small.project(f.tensor, f.input); });
  check(provider.stats().io_bytes == 0 && small.stats().cache.requests == 0 &&
        small.stats().graph_peak_bytes == 0 && small.stats().failures == 1,
        "graph capacity preflight touched weights or allocated a graph");
  pmi::GgmlDeviceProjection executor(provider, adapter, 4096, 8192);
  rejects([&] { executor.project(f.tensor, {}); });
  auto integer = f.tensor;
  integer.ggml_type = GGML_TYPE_I32;
  integer.byte_length = pmi::ggml_tensor_nbytes(integer.ggml_type, integer.shape);
  rejects([&] { executor.project(integer, f.input); });
  check(provider.stats().io_bytes == 0 && executor.stats().cache.requests == 0,
        "invalid shape or missing CPU dot kernel touched provider");
  std::filesystem::resize_file(f.path, 1);
  rejects([&] { executor.project(f.tensor, f.input); });
  check(executor.stats().graph_current_bytes == 0 && executor.stats().graph_peak_bytes == 0 &&
        executor.stats().cache.current_bytes == 0 && executor.stats().cache.provider_requests == 1 &&
        executor.stats().failures == 3, "failed host read retained graph storage");
  f.write();
  compare(executor.project(f.tensor, f.input), f);
  check(executor.stats().completed == 1 && executor.stats().graph_current_bytes == 0,
        "executor did not recover after host read error");
  rejects([&] { pmi::GgmlDeviceProjection zero(provider, adapter, 0, 8192); });
  rejects([&] { pmi::GgmlDeviceProjection zero(provider, adapter, 4096, 0); });
  rejects([&] { pmi::GgmlDeviceProjection absent(provider, nullptr, 4096, 8192); });
}
} // namespace

int main() {
  try {
    Directory directory;
    pmi::GgmlBackendOwner owner(ggml_backend_cpu_init(), ggml_backend_free);
    check(bool(owner), "explicit CPU backend initialization failed");
    ggml_backend_cpu_set_n_threads(owner.get(), 1);
    auto adapter = std::make_shared<pmi::GgmlDeviceBackend>(owner);
    double maximum = 0;
    Fixture f32(directory.path, GGML_TYPE_F32, 4);
    Fixture q8(directory.path, GGML_TYPE_Q8_0, 32);
    Fixture iq4(directory.path, GGML_TYPE_IQ4_NL, 160);
    for (bool profile : {false, true})
      for (const auto* f : {&f32, &q8, &iq4}) maximum = std::max(maximum, exercise(*f, adapter, profile));
    resident_case(f32, adapter);
    failures(f32, adapter);
    std::cout << "device projection CPU fixtures passed: F32/Q8_0/IQ4_NL, 25 parity projections/50 values"
              << "; max_abs_error=" << maximum << "; cache bypass, eviction/reload, separate budgets,"
              << " profile on/off, input/type/I/O failure cleanup and recovery\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
