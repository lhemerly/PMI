#include "pmi/device_projection.hpp"
#include "pmi/ggml_device.hpp"
#include "pmi/gguf.hpp"
#include "pmi/telemetry.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef PMI_WITH_VULKAN
#include "ggml-vulkan.h"
#endif

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Diagnostic replay only: reads one extracted, <=64 MiB GGUF. No model loader,
// checkpoint discovery, routing substitution, or tolerance-adjustment options.
namespace {
namespace fs = std::filesystem;
constexpr uint64_t mib = 1024 * 1024, file_limit = 64 * mib, allocation_limit = 128 * mib;
constexpr double atol = 2e-5, rtol = 1e-4;
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
uint64_t integer(const std::string& value) {
  uint64_t result{};
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size(), "invalid unsigned value");
  return result;
}
std::string quoted(const std::string& value) {
  std::ostringstream out; out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
    else out << c;
  }
  out << '"'; return out.str();
}
void finite(const std::vector<float>& values) {
  for (float value : values) require(std::isfinite(value), "nonfinite replay value");
}

struct Fixture {
  pmi::GgufFile file;
  std::map<std::string, pmi::TensorInfo> tensors;
  std::vector<float> input, logits, weights, reference_routed, trial_routed, reference_output, trial_output;
  std::vector<int32_t> ids;
  uint64_t dim{}, hidden{}, shared_hidden{}, total_experts{}, used{};
  float scale{};
  explicit Fixture(const fs::path& path) {
    require(fs::is_regular_file(path) && fs::file_size(path) <= file_limit, "replay must be a regular GGUF <=64 MiB");
    file = pmi::GgufReader(path).inspect();
    require(file.metadata.at("general.architecture").text == "pmi_moe_replay" &&
            file.metadata.at("replay.version").text == "1", "unsupported replay architecture/version");
    require(file.tensors.size() == 16, "replay requires exactly sixteen named tensors");
    for (const auto& tensor : file.tensors) require(tensors.emplace(tensor.name, tensor).second, "duplicate replay tensor");
    size_t parsed = 0;
    const auto& raw_scale = file.metadata.at("replay.expert_weights_scale").text;
    scale = std::stof(raw_scale, &parsed);
    require(parsed == raw_scale.size() && std::isfinite(scale) && scale >= 0, "invalid replay routing scale");
    input = floats("replay.input"); logits = floats("replay.logits"); weights = floats("replay.weights");
    reference_routed = floats("replay.reference_routed"); trial_routed = floats("replay.trial_routed");
    reference_output = floats("replay.reference_output"); trial_output = floats("replay.trial_output");
    const auto& id_tensor = get("replay.ids");
    require(id_tensor.ggml_type == GGML_TYPE_I32 && id_tensor.shape.size() == 1, "replay IDs must be I32 vector");
    ids.resize(id_tensor.shape[0]); read(id_tensor, ids.data());
    dim = input.size(); total_experts = logits.size(); used = ids.size();
    require(dim > 0 && dim <= 16384 && total_experts > 0 && total_experts <= 16384 && used > 0 && used <= 64,
            "replay dimensions exceed fixed bounds");
    require(weights.size() == used && std::set<int32_t>(ids.begin(), ids.end()).size() == used, "invalid replay routes");
    for (auto id : ids) require(id >= 0 && uint64_t(id) < total_experts, "global routing ID outside source expert range");
    for (const auto* values : {&reference_routed, &trial_routed, &reference_output, &trial_output})
      require(values->size() == dim, "saved replay output extent mismatch");
    const auto& gate = get("gate.weight");
    const auto& shared = get("shared_gate.weight");
    require(gate.shape.size() == 3 && shared.shape.size() == 2, "invalid expert matrix ranks");
    hidden = gate.shape[1]; shared_hidden = shared.shape[1];
    require(hidden > 0 && hidden <= 16384 && shared_hidden > 0 && shared_hidden <= 16384, "hidden extent exceeds bound");
    shape("router.weight", {dim, total_experts});
    shape("gate.weight", {dim, hidden, used}); shape("up.weight", {dim, hidden, used});
    shape("down.weight", {hidden, dim, used}); shape("shared_router.weight", {dim, 1});
    shape("shared_gate.weight", {dim, shared_hidden}); shape("shared_up.weight", {dim, shared_hidden});
    shape("shared_down.weight", {shared_hidden, dim});
  }
  const pmi::TensorInfo& get(const std::string& name) const {
    const auto found = tensors.find(name); require(found != tensors.end(), "missing replay tensor: " + name); return found->second;
  }
  void shape(const std::string& name, const std::vector<uint64_t>& dimensions) const {
    const auto& t = get(name);
    require(t.shape == dimensions && t.byte_length <= 32 * mib, "matrix shape/size mismatch: " + name);
    require(t.ggml_type < GGML_TYPE_COUNT && ggml_get_type_traits_cpu(static_cast<ggml_type>(t.ggml_type))->vec_dot,
            "unsupported CPU reference type: " + name);
  }
  void read(const pmi::TensorInfo& t, void* data) const {
    require(t.byte_length <= file_limit && t.absolute_offset <= file.file_size &&
            t.byte_length <= file.file_size - t.absolute_offset, "replay range exceeds compact fixture");
    std::ifstream stream(file.source_path, std::ios::binary);
    require(bool(stream), "cannot open replay fixture");
    stream.seekg(static_cast<std::streamoff>(t.absolute_offset));
    stream.read(static_cast<char*>(data), static_cast<std::streamsize>(t.byte_length));
    require(bool(stream) && uint64_t(stream.gcount()) == t.byte_length, "short replay range read");
  }
  std::vector<float> floats(const std::string& name) const {
    const auto& t = get(name);
    require(t.ggml_type == GGML_TYPE_F32 && t.shape.size() == 1 && t.shape[0] <= 16384,
            "replay activation must be a bounded F32 vector");
    std::vector<float> result(t.shape[0]); read(t, result.data()); finite(result); return result;
  }
  pmi::TensorInfo slice(const std::string& name, size_t slot) const {
    auto result = get(name);
    require(result.shape.size() == 3 && slot < used && result.byte_length % used == 0, "invalid replay expert slice");
    result.byte_length /= used;
    result.absolute_offset += slot * result.byte_length;
    result.offset += slot * result.byte_length;
    result.shape.resize(2);
    result.name += ".global-expert-" + std::to_string(ids[slot]);
    return result;
  }
};

class CpuGraph {
 public:
  CpuGraph() {
    ggml_init_params p{}; p.no_alloc = true;
    p.mem_size = 512 * ggml_tensor_overhead() + ggml_graph_overhead_custom(512, false);
    context = ggml_init(p); require(context != nullptr, "CPU graph descriptor allocation failed");
  }
  ~CpuGraph() { ggml_backend_buffer_free(buffer); ggml_free(context); }
  ggml_tensor* input(const std::vector<float>& values, int64_t rows = 1) {
    require(rows > 0 && values.size() % rows == 0, "invalid input extent");
    auto* result = ggml_new_tensor_2d(context, GGML_TYPE_F32, values.size() / rows, rows);
    inputs.push_back({result, values.data(), values.size() * sizeof(float)}); return result;
  }
  ggml_tensor* input_ids(const std::vector<int32_t>& values) {
    auto* result = ggml_new_tensor_2d(context, GGML_TYPE_I32, values.size(), 1);
    inputs.push_back({result, values.data(), values.size() * sizeof(int32_t)}); return result;
  }
  void compute(ggml_tensor* output, const Fixture* fixture = nullptr,
               const std::vector<std::pair<const pmi::TensorInfo*, ggml_tensor*>>& weights = {}) {
    auto* graph = ggml_new_graph_custom(context, 512, false); ggml_build_forward_expand(graph, output);
    const uint64_t bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(context, ggml_backend_cpu_buffer_type());
    auto plan = ggml_graph_plan(graph, 1, nullptr);
    require(bytes <= allocation_limit && plan.work_size <= allocation_limit - bytes, "CPU graph exceeds128 MiB buffer/workspace bound");
    buffer = ggml_backend_alloc_ctx_tensors_from_buft(context, ggml_backend_cpu_buffer_type());
    require(buffer != nullptr, "CPU graph allocation failed");
    for (const auto& input : inputs) std::memcpy(input.tensor->data, input.data, input.bytes);
    for (const auto& pair : weights) fixture->read(*pair.first, pair.second->data);
    std::vector<uint8_t> work(plan.work_size); plan.work_data = work.data();
    require(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS, "CPU reference graph failed");
  }
  static std::vector<float> column(const ggml_tensor* tensor, size_t length, size_t slot = 0) {
    require(tensor->type == GGML_TYPE_F32 && tensor->nb[0] == sizeof(float) && length == size_t(tensor->ne[0]),
            "invalid CPU output column");
    std::vector<float> result(length);
    std::memcpy(result.data(), static_cast<const uint8_t*>(tensor->data) + slot * tensor->nb[1], length * sizeof(float));
    finite(result); return result;
  }
  ggml_context* context{};
 private:
  struct Input { ggml_tensor* tensor; const void* data; size_t bytes; };
  std::vector<Input> inputs;
  ggml_backend_buffer_t buffer{};
};

ggml_tensor* sum_experts(ggml_context* c, ggml_tensor* down, ggml_tensor* weights, size_t dim, size_t used, float scale) {
  if (scale != 0 && scale != 1) weights = ggml_scale(c, weights, scale);
  auto* weighted = ggml_mul(c, down, ggml_reshape_3d(c, weights, 1, used, 1));
  auto* sum = ggml_view_2d(c, weighted, dim, 1, weighted->nb[2], 0);
  for (size_t i = 1; i < used; ++i)
    sum = ggml_add(c, sum, ggml_view_2d(c, weighted, dim, 1, weighted->nb[2], i * weighted->nb[1]));
  return sum;
}
struct Expert { std::vector<float> gate, up, activation, down; };
struct Native {
  std::vector<float> logits, weights, routed, shared, output;
  std::vector<int32_t> ids;
  std::vector<Expert> experts;
};
Native native(const Fixture& f) {
  CpuGraph graph; auto* c = graph.context;
  std::vector<std::pair<const pmi::TensorInfo*, ggml_tensor*>> loads;
  auto weight = [&](const char* name) {
    const auto& t = f.get(name); std::vector<int64_t> ne(t.shape.begin(), t.shape.end());
    auto* result = ggml_new_tensor(c, static_cast<ggml_type>(t.ggml_type), ne.size(), ne.data());
    require(ggml_nbytes(result) == t.byte_length, "native ggml weight extent differs from serialized matrix");
    loads.emplace_back(&t, result); return result;
  };
  auto* router = weight("router.weight");
  auto* gate_weight = weight("gate.weight"); auto* up_weight = weight("up.weight"); auto* down_weight = weight("down.weight");
  auto* sg_weight = weight("shared_gate.weight"); auto* su_weight = weight("shared_up.weight");
  auto* sd_weight = weight("shared_down.weight"); auto* sr_weight = weight("shared_router.weight");
  auto* input = graph.input(f.input);
  auto* logits = ggml_mul_mat(c, router, input);
  auto* probabilities = ggml_soft_max(c, logits);
  auto* ids = ggml_argsort_top_k(c, probabilities, f.used);
  auto* weights = ggml_get_rows(c, ggml_reshape_3d(c, probabilities, 1, f.total_experts, 1), ids);
  weights = ggml_reshape_2d(c, weights, f.used, 1);
  weights = ggml_div(c, weights, ggml_clamp(c, ggml_sum_rows(c, weights), 6.103515625e-5f, INFINITY));
  std::vector<int32_t> mapping(f.used); std::iota(mapping.begin(), mapping.end(), 0);
  auto* local_ids = graph.input_ids(mapping);
  auto* x3 = ggml_reshape_3d(c, input, f.dim, 1, 1);
  auto* gate = ggml_mul_mat_id(c, gate_weight, x3, local_ids);
  auto* up = ggml_mul_mat_id(c, up_weight, x3, local_ids);
  auto* activation = ggml_swiglu_split(c, gate, up);
  auto* down = ggml_mul_mat_id(c, down_weight, activation, local_ids);
  auto* routed = sum_experts(c, down, weights, f.dim, f.used, f.scale);
  auto* sg = ggml_mul_mat(c, sg_weight, input); auto* su = ggml_mul_mat(c, su_weight, input);
  auto* shared = ggml_mul(c, ggml_mul_mat(c, sd_weight, ggml_swiglu_split(c, sg, su)),
                         ggml_sigmoid(c, ggml_mul_mat(c, sr_weight, input)));
  auto* output = ggml_add(c, routed, shared);
  graph.compute(output, &f, loads);
  Native result;
  result.logits = CpuGraph::column(logits, f.total_experts); result.weights = CpuGraph::column(weights, f.used);
  result.ids.assign(static_cast<int32_t*>(ids->data), static_cast<int32_t*>(ids->data) + f.used);
  require(result.ids == f.ids, "recomputed native global routing IDs differ from saved trace; refusing projection replay");
  for (size_t i = 0; i < f.used; ++i)
    result.experts.push_back({CpuGraph::column(gate, f.hidden, i), CpuGraph::column(up, f.hidden, i),
                              CpuGraph::column(activation, f.hidden, i), CpuGraph::column(down, f.dim, i)});
  result.routed = CpuGraph::column(routed, f.dim); result.shared = CpuGraph::column(shared, f.dim);
  result.output = CpuGraph::column(output, f.dim);
  return result; // Independent resident native weights released before device replay.
}

struct Reporter {
  fs::path directory;
  std::ofstream index;
  size_t comparisons{}, failed_required{}, failed_diagnostic{};
  explicit Reporter(const fs::path& path) : directory(path) {
    require(fs::create_directory(path), "artifact directory must be a new directory");
    index.open(path / "index.jsonl"); require(bool(index), "cannot create replay index");
  }
  void emit(const std::string& row) { index << row << '\n'; index.flush(); require(bool(index), "cannot write replay index"); std::cout << row << std::endl; }
  template<class T> void save(const std::string& name, const std::vector<T>& values) {
    require(values.size() <= 16384, "artifact vector exceeds bound");
    std::ofstream out(directory / name, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    require(bool(out), "cannot write replay vector artifact");
  }
  bool compare(const std::string& name, const std::vector<float>& reference, const std::vector<float>& actual,
               bool required = true, int32_t id = -1, const std::string& quantization = "f32") {
    require(reference.size() == actual.size() && !reference.empty(), "comparison extent mismatch: " + name);
    finite(reference); finite(actual);
    size_t mismatches = 0, worst = 0; double maximum = 0, scaled_maximum = 0, square_error = 0, square_reference = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
      const double error = std::abs(double(actual[i]) - reference[i]);
      const double scaled = error / (atol + rtol * std::abs(double(reference[i])));
      mismatches += scaled > 1;
      maximum = std::max(maximum, error);
      if (scaled > scaled_maximum) { scaled_maximum = scaled; worst = i; }
      square_error += error * error; square_reference += double(reference[i]) * reference[i];
    }
    if (mismatches) { if (required) ++failed_required; else ++failed_diagnostic; }
    const auto stem = std::to_string(comparisons++);
    const auto expected_path = stem + ".reference.f32le", actual_path = stem + ".actual.f32le";
    save(expected_path, reference); save(actual_path, actual);
    std::ostringstream out; out.imbue(std::locale::classic()); out << std::setprecision(12);
    out << "{\"type\":\"comparison\",\"name\":" << quoted(name) << ",\"required\":" << (required ? "true" : "false")
        << ",\"passed\":" << (!mismatches ? "true" : "false") << ",\"expert_id\":" << id
        << ",\"weight_type\":" << quoted(quantization) << ",\"shape\":[" << actual.size() << "],\"values\":" << actual.size()
        << ",\"mismatches\":" << mismatches << ",\"max_abs_error\":" << maximum
        << ",\"max_scaled_error\":" << scaled_maximum << ",\"absolute_tolerance\":" << atol << ",\"relative_tolerance\":" << rtol
        << ",\"nmse_diagnostic\":" << square_error / std::max(square_reference, std::numeric_limits<double>::min())
        << ",\"worst_index\":" << worst << ",\"worst_reference\":" << reference[worst] << ",\"worst_actual\":" << actual[worst]
        << ",\"reference_f32le\":" << quoted(expected_path) << ",\"actual_f32le\":" << quoted(actual_path) << '}';
    emit(out.str()); return mismatches == 0;
  }
};

std::vector<float> activate(const std::vector<float>& gate, const std::vector<float>& up) {
  CpuGraph graph; auto* output = ggml_swiglu_split(graph.context, graph.input(gate), graph.input(up));
  graph.compute(output); return CpuGraph::column(output, gate.size());
}
std::pair<std::vector<float>, std::vector<float>> combine(const Fixture& f, const std::vector<float>& down, const Native& n) {
  CpuGraph graph; auto* c = graph.context;
  auto* packed = ggml_reshape_3d(c, graph.input(down, f.used), f.dim, f.used, 1);
  auto* routed = sum_experts(c, packed, graph.input(f.weights), f.dim, f.used, f.scale);
  auto* output = ggml_add(c, routed, graph.input(n.shared)); graph.compute(output);
  return {CpuGraph::column(routed, f.dim), CpuGraph::column(output, f.dim)};
}
void projection_replay(const Fixture& f, const Native& n, Reporter& report,
                       const std::shared_ptr<pmi::GgmlDeviceBackend>& backend, const std::string& label,
                       uint64_t host_cache, uint64_t device_cache, uint64_t graph_bytes, bool reproduce_trial,
                       pmi::ProjectionActivationMode activation_mode) {
  pmi::FilePageProvider provider(f.file.source_path, host_cache);
  pmi::GgmlDeviceProjection projection(provider, backend, device_cache, graph_bytes, true, activation_mode);
  std::vector<float> down;
  for (size_t slot = 0; slot < f.used; ++slot) {
    const auto gate_info = f.slice("gate.weight", slot), up_info = f.slice("up.weight", slot), down_info = f.slice("down.weight", slot);
    const auto gate = projection.project(gate_info, f.input), up = projection.project(up_info, f.input);
    const auto activation = activate(gate, up);
    const auto isolated_down = projection.project(down_info, n.experts[slot].activation);
    const auto chained_down = projection.project(down_info, activation);
    const int32_t id = f.ids[slot];
    report.compare(label + ".gate", n.experts[slot].gate, gate, true, id, pmi::ggml_type_name(gate_info.ggml_type));
    report.compare(label + ".up", n.experts[slot].up, up, true, id, pmi::ggml_type_name(up_info.ggml_type));
    report.compare(label + ".activation", n.experts[slot].activation, activation, true, id);
    report.compare(label + ".down_native_activation", n.experts[slot].down, isolated_down, true, id, pmi::ggml_type_name(down_info.ggml_type));
    report.compare(label + ".down_chained_activation", n.experts[slot].down, chained_down, true, id, pmi::ggml_type_name(down_info.ggml_type));
    down.insert(down.end(), chained_down.begin(), chained_down.end());
  }
  const auto output = combine(f, down, n);
  report.compare(label + ".routed", n.routed, output.first);
  report.compare(label + ".combined_moe", n.output, output.second);
  report.compare(label + ".saved_trial_routed", f.trial_routed, output.first, reproduce_trial);
  report.compare(label + ".saved_trial_combined_moe", f.trial_output, output.second, reproduce_trial);
  const auto s = projection.stats(); const auto host = provider.stats();
  std::ostringstream out; out.imbue(std::locale::classic()); out << std::setprecision(12);
  out << "{\"type\":\"projection_summary\",\"name\":" << quoted(label)
      << ",\"activation_mode\":\"" << (activation_mode == pmi::ProjectionActivationMode::CpuCompatible ? "cpu-compatible" : "backend-native") << '"'
      << ",\"calls\":" << s.calls << ",\"completed\":" << s.completed << ",\"failures\":" << s.failures
      << ",\"cache_hits\":" << s.cache.hits << ",\"cache_misses\":" << s.cache.misses << ",\"uploads\":" << s.cache.uploads
      << ",\"upload_bytes\":" << s.cache.upload_bytes << ",\"cache_peak_bytes\":" << s.cache.peak_bytes
      << ",\"cache_capacity_bytes\":" << s.cache.capacity_bytes << ",\"graph_peak_bytes\":" << s.graph_peak_bytes
      << ",\"graph_capacity_bytes\":" << s.graph_capacity_bytes << ",\"host_io_bytes\":" << host.io_bytes
      << ",\"activation_preparation_calls\":" << s.activation_preparation_calls
      << ",\"activation_preparation_bytes\":" << s.activation_preparation_bytes
      << ",\"activation_preparation_peak_bytes\":" << s.activation_preparation_peak_bytes
      << ",\"activation_preparation_ms\":" << s.activation_preparation_ms
      << ",\"kernel_ms\":" << s.kernel_ms << ",\"managed_bytes_scope\":\"ggml_buffers_excluding_driver_staging_and_scratch\"} ";
  report.emit(out.str());
  require(s.calls == f.used * 4 && s.completed == s.calls && !s.failures, "projection replay is incomplete");
  projection.clear();
}

} // namespace
int main(int argc, char** argv) {
  try {
    require(argc >= 2, "usage: pmi_moe_projection_replay FIXTURE.gguf --artifact-dir NEW_DIR [--backend cpu|vulkan]");
    const uint16_t one = 1; require(*reinterpret_cast<const uint8_t*>(&one) == 1, "replay artifacts require little-endian host");
    fs::path fixture_path = argv[1], artifacts;
    std::string backend_name = "cpu", expected_device, trial_role = "reproduce", label = "vulkan";
    pmi::ProjectionActivationMode activation_mode = pmi::ProjectionActivationMode::BackendNative;
    uint64_t device_index = 0, host_cache = 2 * mib, device_cache = 2 * mib, graph_bytes = 2 * mib;
    bool device_specified = false;
    for (int i = 2; i < argc; ++i) {
      const std::string option = argv[i]; require(++i < argc, "missing replay option value"); const std::string value = argv[i];
      if (option == "--artifact-dir") artifacts = value;
      else if (option == "--backend") backend_name = value;
      else if (option == "--device") { device_index = integer(value); device_specified = true; }
      else if (option == "--expect-device-id") expected_device = value;
      else if (option == "--label") label = value;
      else if (option == "--trial-role") trial_role = value;
      else if (option == "--activation-mode") {
        if (value == "backend-native") activation_mode = pmi::ProjectionActivationMode::BackendNative;
        else if (value == "cpu-compatible") activation_mode = pmi::ProjectionActivationMode::CpuCompatible;
        else throw std::invalid_argument("activation-mode must be backend-native or cpu-compatible");
      }
      else if (option == "--host-cache-bytes") host_cache = integer(value);
      else if (option == "--device-cache-bytes") device_cache = integer(value);
      else if (option == "--graph-bytes") graph_bytes = integer(value);
      else throw std::invalid_argument("unknown replay option: " + option);
    }
    require(!artifacts.empty() && (backend_name == "cpu" || backend_name == "vulkan"), "invalid artifact/backend choice");
    require(trial_role == "reproduce" || trial_role == "observe", "trial-role must be reproduce or observe");
    require(backend_name == "vulkan" || activation_mode == pmi::ProjectionActivationMode::BackendNative,
            "cpu-compatible activation mode requires Vulkan");
    require(label.size() <= 64 && !label.empty(), "invalid replay label");
    require(host_cache > 0 && host_cache <= file_limit && device_cache > 0 && device_cache <= file_limit &&
            graph_bytes > 0 && graph_bytes <= 8 * mib, "cache/graph budgets exceed replay bounds");
    require(device_index <= UINT32_MAX, "device index exceeds uint32");
    require(backend_name == "vulkan" ? device_specified && !expected_device.empty() : !device_specified && expected_device.empty(),
            "Vulkan requires explicit device index and expected physical device ID; CPU must not request a GPU");
#ifndef PMI_WITH_VULKAN
    require(backend_name != "vulkan", "Vulkan replay is unavailable in this build");
#endif
    Fixture f(fixture_path);
    Reporter report(artifacts);
    ggml_backend_register(ggml_backend_cpu_reg()); ggml_cpu_init();
    const auto n = native(f);
    report.save("routing.native.i32le", n.ids); report.save("routing.saved.i32le", f.ids);
    report.save("input.f32le", f.input);
    std::ostringstream identity;
    identity << "{\"type\":\"routing\",\"global_ids_exact\":true,\"global_ids\":[";
    for (size_t i = 0; i < f.ids.size(); ++i) { if (i) identity << ','; identity << f.ids[i]; }
    identity << "],\"native_local_ids\":[";
    for (size_t i = 0; i < f.ids.size(); ++i) { if (i) identity << ','; identity << i; }
    identity << "],\"selected_experts\":" << f.used << ",\"total_source_experts\":" << f.total_experts
             << ",\"shared_expert_scope\":\"resident_native_cpu_preserved\",\"fixture_bytes\":" << f.file.file_size << '}';
    report.emit(identity.str());
    const bool logits_match = report.compare("native.saved_router_logits", f.logits, n.logits);
    const bool weights_match = report.compare("native.saved_routing_weights", f.weights, n.weights);
    const bool routed_match = report.compare("native.saved_reference_routed", f.reference_routed, n.routed);
    const bool output_match = report.compare("native.saved_reference_combined_moe", f.reference_output, n.output);
    require(logits_match && weights_match && routed_match && output_match,
            "native replay did not reproduce saved reference; device comparisons are inadmissible");
    pmi::GgmlBackendOwner cpu_owner(ggml_backend_cpu_init(), ggml_backend_free);
    require(bool(cpu_owner), "CPU projection backend initialization failed");
    ggml_backend_cpu_set_n_threads(cpu_owner.get(), 1);
    auto cpu = std::make_shared<pmi::GgmlDeviceBackend>(cpu_owner);
    projection_replay(f, n, report, cpu, "cpu", host_cache, device_cache, graph_bytes, false,
                      pmi::ProjectionActivationMode::BackendNative);
#ifdef PMI_WITH_VULKAN
    if (backend_name == "vulkan") {
      const int count = ggml_backend_vk_get_device_count();
      require(count > 0 && device_index < uint64_t(count), "requested Vulkan device is unavailable");
      pmi::GgmlBackendOwner owner(ggml_backend_vk_init(device_index), ggml_backend_free);
      require(bool(owner) && ggml_backend_is_vk(owner.get()), "Vulkan backend failed; no fallback permitted");
      auto* device = ggml_backend_get_device(owner.get()); require(device != nullptr, "Vulkan device identity unavailable");
      ggml_backend_dev_props props{}; ggml_backend_dev_get_props(device, &props);
      require(props.device_id && expected_device == props.device_id && props.type == GGML_BACKEND_DEVICE_TYPE_GPU,
              "Vulkan physical device identity/type differs from approved selection");
      report.emit("{\"type\":\"device\",\"index\":" + std::to_string(device_index) + ",\"device_id\":" + quoted(props.device_id) +
                  ",\"description\":" + quoted(props.description ? props.description : "") + ",\"vram_bytes\":null}");
      auto adapter = std::make_shared<pmi::GgmlDeviceBackend>(owner);
      projection_replay(f, n, report, adapter, label, host_cache, device_cache, graph_bytes, trial_role == "reproduce", activation_mode);
    }
#endif
    std::ostringstream complete;
    complete << "{\"type\":\"complete\",\"passed\":" << (report.failed_required ? "false" : "true")
             << ",\"comparisons\":" << report.comparisons << ",\"failed_required\":" << report.failed_required
             << ",\"failed_diagnostic\":" << report.failed_diagnostic << ",\"peak_rss_bytes\":";
    const auto memory = pmi::process_memory(); if (memory.peak_rss_bytes) complete << *memory.peak_rss_bytes; else complete << "null";
    complete << '}'; report.emit(complete.str());
    return report.failed_required ? 1 : 0;
  } catch (const std::exception& error) {
    std::cerr << "MoE projection replay failed: " << error.what() << '\n'; return 1;
  }
}
