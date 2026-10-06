// Strict native-CPU parity for opt-in Vulkan activation preparation.
// Small fractional-input fixtures; full target-layer coverage uses the replay utility.
#include "pmi/device_projection.hpp"
#include "pmi/ggml_device.hpp"
#include "pmi/telemetry.hpp"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::array<size_t, 2> widths = {256, 2560};
constexpr size_t rows = 16, format_count = 4, expected_cases = widths.size() * format_count;
constexpr size_t projections_per_case = 4, expected_projections = expected_cases * projections_per_case;
constexpr size_t expected_compared_values = expected_projections * rows;
constexpr uint64_t max_fixture_payload_bytes = 163840;
constexpr uint64_t graph_capacity = 65536;
constexpr double absolute_tolerance = 2e-5, relative_tolerance = 1e-4, nmse_limit = 5e-4;
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
void emit_memory() {
  const auto memory = pmi::process_memory();
  auto field = [](const char* name, const std::optional<uint64_t>& value) {
    std::cout << ",\"" << name << "\":";
    if (value) std::cout << *value; else std::cout << "null";
  };
  field("rss_bytes", memory.rss_bytes);
  field("peak_rss_bytes", memory.peak_rss_bytes);
  field("process_storage_read_bytes", memory.storage_read_bytes);
}
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
std::string quoted(const std::string& text) {
  std::ostringstream out;
  out << '"';
  for (unsigned char character : text) {
    if (character == '"' || character == '\\') out << '\\' << character;
    else if (character < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(character) << std::dec;
    else out << character;
  }
  out << '"';
  return out.str();
}
struct Directory {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("pmi-vulkan-activation-parity-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  Directory() { std::filesystem::create_directories(path); }
  ~Directory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};
struct QuantizationCleanup { ~QuantizationCleanup() { ggml_quantize_free(); } };
struct Fixture {
  size_t width;
  pmi::TensorInfo tensor;
  std::filesystem::path path;
  std::vector<float> input, changed_input;
  Fixture(const Directory& directory, ggml_type type, size_t fixture_width) : width(fixture_width) {
    tensor.name = "tiny.single-token.matrix";
    tensor.shape = {width, rows};
    tensor.ggml_type = static_cast<uint32_t>(type);
    tensor.byte_length = pmi::ggml_tensor_nbytes(tensor.ggml_type, tensor.shape);
    require(tensor.byte_length <= max_fixture_payload_bytes, "fixture exceeds its declared tiny payload bound");
    std::vector<float> dense(width * rows), importance(width, 1.0f);
    std::vector<uint8_t> bytes(static_cast<size_t>(tensor.byte_length));
    input.resize(width);
    changed_input.resize(width);
    // Fractional activations expose CPU RHS quantization differences.
    for (size_t i = 0; i < width; ++i) input[i] = 1.2f * std::sin(float(i) * 0.173f) + 0.37f * std::cos(float(i) * 0.007f);
    for (size_t i = 0; i < width; ++i) changed_input[i] = 0.75f * std::cos(float(i) * 0.139f) + 0.23f * std::sin(float(i) * 0.011f);
    path = directory.path / (std::string(ggml_type_name(type)) + "-" + std::to_string(width) + ".bin");
    std::ofstream out(path, std::ios::binary);
    for (unsigned copy = 0; copy < 2; ++copy) {
      for (size_t i = 0; i < dense.size(); ++i)
        dense[i] = 0.3f * std::sin(float(i) * 0.41f + 0.2f + float(copy)) +
                   0.07f * std::cos(float(i) * 0.13f + 0.7f * float(copy));
      if (type == GGML_TYPE_F32) std::memcpy(bytes.data(), dense.data(), bytes.size());
      else require(ggml_quantize_chunk(type, dense.data(), bytes.data(), 0, rows, width, importance.data()) == bytes.size(),
                   "fixture quantization produced an incorrect extent");
      out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    require(static_cast<bool>(out), "cannot write tiny synthetic weight fixture");
  }
};
struct Comparison { double maximum_error{}, maximum_scaled_error{}, nmse{}; };
Comparison compare(const std::vector<float>& reference, const std::vector<float>& actual) {
  require(reference.size() == rows && actual.size() == rows, "projection output extent mismatch");
  Comparison result;
  double error_squared = 0, reference_squared = 0;
  for (size_t i = 0; i < rows; ++i) {
    require(std::isfinite(reference[i]) && std::isfinite(actual[i]), "nonfinite CPU or Vulkan output");
    const double error = std::abs(double(actual[i]) - reference[i]);
    result.maximum_error = std::max(result.maximum_error, error);
    const double scaled = error / (absolute_tolerance + relative_tolerance * std::abs(reference[i]));
    result.maximum_scaled_error = std::max(result.maximum_scaled_error, scaled);
    error_squared += error * error;
    reference_squared += double(reference[i]) * reference[i];
  }
  result.nmse = error_squared / std::max(reference_squared, std::numeric_limits<double>::min());
  require(std::isfinite(result.nmse), "nonfinite normalized error");
  return result;
}
void emit_stats(const pmi::ProjectionStats& stats) {
#define FIELD(name) std::cout << ",\"" #name "\":" << stats.name
  std::cout << "\"calls\":" << stats.calls;
  FIELD(completed); FIELD(failures); FIELD(graph_current_bytes); FIELD(graph_peak_bytes); FIELD(graph_capacity_bytes);
  FIELD(activation_upload_bytes); FIELD(output_readback_bytes); FIELD(cache_request_ms); FIELD(graph_setup_ms);
  FIELD(activation_upload_ms); FIELD(kernel_ms); FIELD(output_readback_ms); FIELD(graph_teardown_ms);
#undef FIELD
  std::cout << ",\"cache\":{\"requests\":" << stats.cache.requests;
#define FIELD(name) std::cout << ",\"" #name "\":" << stats.cache.name
  FIELD(requested_bytes); FIELD(hits); FIELD(misses); FIELD(failures); FIELD(provider_requests);
  FIELD(upload_bytes); FIELD(uploads); FIELD(evictions); FIELD(current_bytes); FIELD(peak_bytes);
  FIELD(capacity_bytes); FIELD(entries); FIELD(active_leases);
#undef FIELD
  std::cout << '}';
}
pmi::ProjectionStats delta(pmi::ProjectionStats after, const pmi::ProjectionStats& before) {
#define DELTA(name) after.name -= before.name
  DELTA(calls); DELTA(completed); DELTA(failures); DELTA(activation_upload_bytes); DELTA(output_readback_bytes);
  DELTA(cache_request_ms); DELTA(graph_setup_ms); DELTA(activation_upload_ms); DELTA(kernel_ms);
  DELTA(output_readback_ms); DELTA(graph_teardown_ms); DELTA(cache.requests); DELTA(cache.requested_bytes);
  DELTA(cache.hits); DELTA(cache.misses); DELTA(cache.failures); DELTA(cache.provider_requests);
  DELTA(cache.upload_bytes); DELTA(cache.uploads); DELTA(cache.evictions);
#undef DELTA
  return after;
}
void exercise(const Fixture& fixture, const std::shared_ptr<pmi::GgmlDeviceBackend>& cpu,
              const std::shared_ptr<pmi::GgmlDeviceBackend>& vulkan) {
  const uint64_t host_capacity = (fixture.tensor.byte_length + 63) / 64 * 64;
  pmi::FilePageProvider cpu_provider(fixture.path, host_capacity);
  pmi::DevicePageRequest descriptor{fixture.tensor, cpu->format, cpu->layout};
  const uint64_t cpu_charge = cpu->allocation_bytes(descriptor);
  pmi::GgmlDeviceProjection cpu_projection(cpu_provider, cpu, cpu_charge, graph_capacity);
  pmi::FilePageProvider provider(fixture.path, host_capacity);
  const uint64_t device_charge = vulkan->allocation_bytes(descriptor);
  require(device_charge <= 1024 * 1024, "tiny weight quote unexpectedly exceeds one MiB");
  pmi::GgmlDeviceProjection projection(provider, vulkan, device_charge, graph_capacity, true, pmi::ProjectionActivationMode::CpuCompatible);
  Comparison maximum;
  std::vector<float> first_reference;
  auto project = [&](const pmi::TensorInfo& tensor, const std::vector<float>& input, const char* label) {
    const auto cpu_start = Clock::now();
    const auto reference = cpu_projection.project(tensor, input);
    const double cpu_wall = elapsed_ms(cpu_start);
    if (first_reference.empty()) first_reference = reference;
    else if (std::string(label) != "reload-original")
      require(reference != first_reference, "changed activation/weights did not change CPU reference output");
    const auto before = projection.stats();
    const auto start = Clock::now();
    const auto actual = projection.project(tensor, input);
    const double wall = elapsed_ms(start);
    const auto measured = compare(reference, actual);
    std::cout << "{\"type\":\"projection\",\"quantization\":" << quoted(ggml_type_name(static_cast<ggml_type>(fixture.tensor.ggml_type)))
              << ",\"shape\":[" << fixture.width << ',' << rows << "]"
              << ",\"case\":" << quoted(label) << ",\"elapsed_ms\":" << wall << ",\"cpu_elapsed_ms\":" << cpu_wall
              << ",\"max_abs_error\":" << measured.maximum_error << ",\"max_scaled_error\":" << measured.maximum_scaled_error
              << ",\"nmse\":" << measured.nmse << ",\"delta\":{";
    emit_stats(delta(projection.stats(), before));
    std::cout << "}}" << std::endl;
    maximum.maximum_error = std::max(maximum.maximum_error, measured.maximum_error);
    maximum.maximum_scaled_error = std::max(maximum.maximum_scaled_error, measured.maximum_scaled_error);
    maximum.nmse = std::max(maximum.nmse, measured.nmse);
  };
  project(fixture.tensor, fixture.input, "cold-original");
  const auto first_host = provider.stats();
  project(fixture.tensor, fixture.changed_input, "warm-weight-changed-input");
  require(provider.stats().io_bytes == first_host.io_bytes && provider.stats().hits == first_host.hits &&
          provider.stats().misses == first_host.misses, "Vulkan cache hit touched host weights");
  auto alternate = fixture.tensor;
  alternate.absolute_offset = fixture.tensor.byte_length;
  project(alternate, fixture.input, "cold-alternate-weight");
  project(fixture.tensor, fixture.input, "reload-original");
  const auto stats = projection.stats();
  require(stats.calls == 4 && stats.completed == 4 && stats.failures == 0, "projection call accounting mismatch");
  require(stats.cache.requests == 4 && stats.cache.hits == 1 && stats.cache.misses == 3 &&
          stats.cache.uploads == 3 && stats.cache.provider_requests == 3 && stats.cache.evictions == 2 &&
          stats.cache.failures == 0, "forced Vulkan eviction/reload counters differ");
  require(stats.cache.current_bytes == device_charge && stats.cache.peak_bytes == device_charge &&
          stats.cache.upload_bytes == 3 * fixture.tensor.byte_length && stats.cache.active_leases == 0 &&
          provider.stats().active_leases == 0, "Vulkan weight allocation or lease accounting mismatch");
  require(provider.stats().io_bytes == 3 * fixture.tensor.byte_length, "evicted host bytes were not reloaded");
  require(stats.graph_current_bytes == 0 && stats.graph_peak_bytes > 0 &&
          stats.graph_peak_bytes <= graph_capacity, "Vulkan graph buffer escaped its bound or lifetime");
  require(stats.activation_upload_bytes == 4 * fixture.width * sizeof(float) &&
          stats.output_readback_bytes == 4 * rows * sizeof(float), "activation/readback byte accounting mismatch");
  for (double value : {stats.cache_request_ms, stats.graph_setup_ms, stats.activation_upload_ms,
                       stats.kernel_ms, stats.output_readback_ms, stats.graph_teardown_ms})
    require(std::isfinite(value) && value >= 0, "invalid Vulkan profile timing");
  projection.clear();
  require(projection.stats().cache.current_bytes == 0 && projection.stats().cache.evictions == 3,
          "Vulkan cleanup retained a cached weight allocation");
  const bool passed = maximum.maximum_scaled_error <= 1 && maximum.nmse <= nmse_limit;
  std::cout << "{\"type\":\"case\",\"quantization\":" << quoted(ggml_type_name(static_cast<ggml_type>(fixture.tensor.ggml_type)))
            << ",\"shape\":[" << fixture.width << ',' << rows << "],\"input_columns\":1,\"supported\":true,\"passed\":" << (passed ? "true" : "false")
            << ",\"compared_values\":64,\"max_abs_error\":" << maximum.maximum_error
            << ",\"max_scaled_error\":" << maximum.maximum_scaled_error << ",\"max_nmse\":" << maximum.nmse
            << ",\"absolute_tolerance\":" << absolute_tolerance << ",\"relative_tolerance\":" << relative_tolerance
            << ",\"nmse_limit\":" << nmse_limit << ",\"host_io_bytes\":" << provider.stats().io_bytes << ",\"projection\":{";
  emit_stats(stats);
  std::cout << "}}" << std::endl;
  require(passed, "Vulkan projection differs from the explicit CPU reference; see case error metrics");
}
} // namespace

int main(int argc, char** argv) {
  size_t completed = 0;
  std::cout.imbue(std::locale::classic());
  std::cout << std::setprecision(12);
  try {
    size_t index = 0;
    std::string expected_description, expected_id;
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      require(++i < argc, "missing Vulkan test argument value");
      const std::string value = argv[i];
      if (option == "--device") {
        require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos, "invalid device index");
        index = std::stoull(value);
      } else if (option == "--expect-description") expected_description = value;
      else if (option == "--expect-device-id") expected_id = value;
      else throw std::invalid_argument("unknown Vulkan test option: " + option);
    }
    require(!expected_description.empty(), "--expect-description is required to verify the approved physical GPU");
    Directory directory;
    QuantizationCleanup cleanup;
    pmi::GgmlBackendOwner cpu_owner(ggml_backend_cpu_init(), ggml_backend_free);
    require(bool(cpu_owner), "explicit CPU backend initialization failed");
    ggml_backend_cpu_set_n_threads(cpu_owner.get(), 1);
    auto cpu = std::make_shared<pmi::GgmlDeviceBackend>(cpu_owner);
    const auto initialization = Clock::now();
    const int count = ggml_backend_vk_get_device_count();
    require(count > 0 && index < static_cast<size_t>(count), "requested Vulkan device does not exist; no fallback permitted");
    pmi::GgmlBackendOwner owner(ggml_backend_vk_init(index), ggml_backend_free);
    require(bool(owner) && ggml_backend_is_vk(owner.get()), "explicit Vulkan initialization failed; no fallback permitted");
    auto* device = ggml_backend_get_device(owner.get());
    require(device != nullptr, "Vulkan backend did not expose device identity");
    ggml_backend_dev_props properties{};
    ggml_backend_dev_get_props(device, &properties);
    const double initialization_ms = elapsed_ms(initialization);
    require(properties.type == GGML_BACKEND_DEVICE_TYPE_GPU, "selected Vulkan device is not a dedicated physical GPU");
    const std::string description = properties.description ? properties.description : "";
    const std::string device_id = properties.device_id ? properties.device_id : "";
    require(description.find(expected_description) != std::string::npos, "Vulkan description differs from the approved GPU");
    require(expected_id.empty() || device_id == expected_id, "Vulkan PCI identity differs from the approved GPU");
    std::cout << "{\"type\":\"device\",\"backend\":" << quoted(ggml_backend_name(owner.get()))
              << ",\"index\":" << index << ",\"name\":" << quoted(properties.name ? properties.name : "")
              << ",\"description\":" << quoted(description) << ",\"device_id\":" << quoted(device_id)
              << ",\"dedicated_gpu\":true,\"reported_memory_total\":" << properties.memory_total
              << ",\"reported_memory_free\":" << properties.memory_free
              << ",\"backend_initialization_ms\":" << initialization_ms
              << ",\"cache_temperature\":\"PMI weight cache only; OS and driver caches uncontrolled\""
              << ",\"cache_request_ms_scope\":\"combined host request, admission, weight allocation and upload\""
              << ",\"fixture_widths\":[256,2560],\"fixture_rows\":" << rows << ",\"format_count\":" << format_count
              << ",\"expected_cases\":" << expected_cases << ",\"expected_projections\":" << expected_projections
              << ",\"expected_compared_values\":" << expected_compared_values
              << ",\"max_fixture_payload_bytes\":" << max_fixture_payload_bytes
              << ",\"graph_capacity_bytes\":" << graph_capacity << "}" << std::endl;
    auto vulkan = std::make_shared<pmi::GgmlDeviceBackend>(owner);
    constexpr std::array<ggml_type, format_count> types = {
        GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_S};
    // k=256 covers compact fixtures; k=2560 also exercises the quantized
    // single-vector dispatch selected for the target's hidden width on AMD.
    for (const auto width : widths)
      for (const auto type : types) {
        Fixture fixture(directory, type, width);
        exercise(fixture, cpu, vulkan);
        ++completed;
      }
    require(completed == expected_cases, "incomplete Vulkan type/width coverage");
    std::cout << "{\"type\":\"complete\",\"expected_cases\":" << expected_cases
              << ",\"completed_cases\":" << completed
              << ",\"expected_projections\":" << expected_projections
              << ",\"expected_compared_values\":" << expected_compared_values
              << ",\"success\":true";
    emit_memory();
    std::cout << '}' << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cout << "{\"type\":\"complete\",\"expected_cases\":" << expected_cases
              << ",\"completed_cases\":" << completed
              << ",\"expected_projections\":" << expected_projections
              << ",\"expected_compared_values\":" << expected_compared_values
              << ",\"success\":false,\"error\":" << quoted(error.what());
    emit_memory();
    std::cout << '}' << std::endl;
    std::cerr << error.what() << '\n';
    return 1;
  }
}
