#include "pmi/connector.hpp"
#include "pmi/gguf.hpp"
#include "pmi/telemetry.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <charconv>
#include <chrono>
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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Real-weight validation utility, not a decoder or an inference benchmark.
// The reference reads COMPLETE physical packed GGUF matrices directly into
// native ggml CPU tensors. It never uses PMI expert slices or PageProvider for
// reference weights, and runs native MUL_MAT_ID for all tokens together.
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr uint64_t accounted_limit = uint64_t{2} * 1024 * 1024 * 1024;
constexpr uint64_t auxiliary_allowance = uint64_t{64} * 1024 * 1024;
constexpr double absolute_tolerance = 3e-7;
constexpr double relative_tolerance = 3e-5;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
uint64_t add(uint64_t a, uint64_t b) {
  require(b <= UINT64_MAX - a, "size accounting overflow"); return a + b;
}
uint64_t integer(const std::string& text) {
  uint64_t value{};
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size(), "invalid unsigned integer: " + text);
  return value;
}
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct PhysicalWeight {
  const pmi::TensorInfo* info{};
  fs::path path;
};
PhysicalWeight find_physical(const std::vector<pmi::GgufFile>& files, const std::string& name) {
  PhysicalWeight result;
  for (const auto& file : files) for (const auto& tensor : file.tensors) if (tensor.name == name) {
    require(result.info == nullptr, "duplicate physical tensor: " + name);
    result = {&tensor, file.source_path};
  }
  require(result.info != nullptr, "missing physical tensor: " + name); return result;
}

struct NativeResult {
  std::vector<pmi::MoeLayerTrace> tokens;
  uint64_t weight_bytes{}, buffer_bytes{}, workspace_bytes{}, accounted_bytes{};
  double load_ms{}, compute_ms{};
  pmi::ProcessMemory memory;
};

struct ContextDeleter { void operator()(ggml_context* context) const { ggml_free(context); } };
struct BufferDeleter { void operator()(ggml_backend_buffer* buffer) const { ggml_backend_buffer_free(buffer); } };

NativeResult native_reference(const std::vector<pmi::GgufFile>& files,
                              const pmi::Qwen4ExpMoeConfig& config,
                              const std::vector<float>& input, uint64_t cache_bytes) {
  const auto tokens = static_cast<int64_t>(input.size() / config.embedding_length);
  const auto dim = static_cast<int64_t>(config.embedding_length);
  const auto hidden = static_cast<int64_t>(config.expert_hidden_length);
  const auto shared_hidden = static_cast<int64_t>(config.shared_hidden_length);
  const auto experts = static_cast<int64_t>(config.expert_count);
  const auto used = static_cast<int64_t>(config.experts_used);
  NativeResult result;
  std::vector<std::pair<PhysicalWeight, ggml_tensor*>> loads;
  const size_t graph_capacity = static_cast<size_t>(128 + 4 * used);
  ggml_init_params params{};
  params.mem_size = (graph_capacity + 16) * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_capacity, false);
  params.no_alloc = true;
  std::unique_ptr<ggml_context, ContextDeleter> context(ggml_init(params));
  require(context != nullptr, "cannot allocate native reference descriptors");
  auto* c = context.get();
  const auto prefix = "blk." + std::to_string(config.layer) + ".";
  auto weight = [&](const std::string& suffix, const std::vector<int64_t>& shape) {
    const auto physical = find_physical(files, prefix + suffix + ".weight");
    const std::vector<uint64_t> expected(shape.begin(), shape.end());
    require(physical.info->shape == expected, "reference physical shape mismatch: " + physical.info->name);
    result.weight_bytes = add(result.weight_bytes, physical.info->byte_length);
    require(add(add(result.weight_bytes, cache_bytes), auxiliary_allowance) <= accounted_limit,
            "complete resident layer exceeds the 2 GiB reference/cache plan; refusing allocation");
    auto* tensor = ggml_new_tensor(c, static_cast<ggml_type>(physical.info->ggml_type),
                                   static_cast<int>(shape.size()), shape.data());
    ggml_set_name(tensor, physical.info->name.c_str());
    require(ggml_nbytes(tensor) == physical.info->byte_length, "reference physical ggml byte count mismatch");
    loads.push_back({physical, tensor}); return tensor;
  };
  auto* router = weight("ffn_gate_inp", {dim, experts});
  auto* gates = weight("ffn_gate_exps", {dim, hidden, experts});
  auto* ups = weight("ffn_up_exps", {dim, hidden, experts});
  auto* downs = weight("ffn_down_exps", {hidden, dim, experts});
  auto* shared_gate_weight = weight("ffn_gate_inp_shexp", {dim});
  auto* shared_gate_matrix = weight("ffn_gate_shexp", {dim, shared_hidden});
  auto* shared_up_matrix = weight("ffn_up_shexp", {dim, shared_hidden});
  auto* shared_down_matrix = weight("ffn_down_shexp", {shared_hidden, dim});

  auto* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, dim, tokens);
  auto* logits = ggml_mul_mat(c, router, x);
  auto* probabilities = ggml_soft_max(c, logits);
  auto* ids = ggml_argsort_top_k(c, probabilities, used);
  auto* weights = ggml_get_rows(c, ggml_reshape_3d(c, probabilities, 1, experts, tokens), ids);
  weights = ggml_reshape_2d(c, weights, used, tokens);
  auto* denominator = ggml_clamp(c, ggml_sum_rows(c, weights), 6.103515625e-5f, INFINITY);
  weights = ggml_div(c, weights, denominator);
  if (config.expert_weights_scale != 0.0f && config.expert_weights_scale != 1.0f)
    weights = ggml_scale(c, weights, config.expert_weights_scale);
  auto* x3 = ggml_reshape_3d(c, x, dim, 1, tokens);
  auto* gate = ggml_mul_mat_id(c, gates, x3, ids);
  auto* up = ggml_mul_mat_id(c, ups, x3, ids);
  auto* activation = ggml_swiglu_split(c, gate, up);
  auto* down = ggml_mul_mat_id(c, downs, activation, ids);
  auto* weighted = ggml_mul(c, down, ggml_reshape_3d(c, weights, 1, used, tokens));
  auto* routed = ggml_view_2d(c, weighted, dim, tokens, weighted->nb[2], 0);
  for (int64_t slot = 1; slot < used; ++slot)
    routed = ggml_add(c, routed, ggml_view_2d(c, weighted, dim, tokens, weighted->nb[2], slot * weighted->nb[1]));
  auto* sg = ggml_mul_mat(c, shared_gate_matrix, x);
  auto* su = ggml_mul_mat(c, shared_up_matrix, x);
  auto* sa = ggml_swiglu_split(c, sg, su);
  auto* sd = ggml_mul_mat(c, shared_down_matrix, sa);
  auto* shared_gate_logit = ggml_mul_mat(c, shared_gate_weight, x);
  auto* shared_gate = ggml_sigmoid(c, shared_gate_logit);
  auto* output = ggml_add(c, routed, ggml_mul(c, sd, shared_gate));
  auto* graph = ggml_new_graph_custom(c, graph_capacity, false);
  ggml_build_forward_expand(graph, output);

  auto plan = ggml_graph_plan(graph, 1, nullptr);
  result.buffer_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(c, ggml_backend_cpu_buffer_type());
  result.workspace_bytes = plan.work_size;
  result.accounted_bytes = add(add(add(result.buffer_bytes, result.workspace_bytes), cache_bytes), auxiliary_allowance);
  require(result.buffer_bytes > 0 && result.accounted_bytes <= accounted_limit,
          "reference backend buffer, workspace, cache and allowance exceed 2 GiB; refusing allocation");
  std::unique_ptr<ggml_backend_buffer, BufferDeleter> buffer(
      ggml_backend_alloc_ctx_tensors_from_buft(c, ggml_backend_cpu_buffer_type()));
  require(buffer != nullptr, "cannot allocate bounded native reference CPU buffer");
  const auto load_start = Clock::now();
  for (const auto& load : loads) {
    const auto& source = load.first;
    require(source.info->absolute_offset <= static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()) &&
            source.info->byte_length <= static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()),
            "reference file range cannot be represented by stream I/O");
    std::ifstream file(source.path, std::ios::binary);
    require(static_cast<bool>(file), "cannot open reference shard: " + source.path.string());
    file.seekg(static_cast<std::streamoff>(source.info->absolute_offset));
    file.read(static_cast<char*>(load.second->data), static_cast<std::streamsize>(source.info->byte_length));
    require(static_cast<bool>(file) && static_cast<uint64_t>(file.gcount()) == source.info->byte_length,
            "short direct read of reference weight: " + source.info->name);
  }
  std::memcpy(x->data, input.data(), input.size() * sizeof(float));
  result.load_ms = elapsed(load_start);
  std::vector<uint8_t> workspace(plan.work_size);
  plan.work_data = workspace.data();
  const auto compute_start = Clock::now();
  require(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS, "native resident MoE graph failed");
  result.compute_ms = elapsed(compute_start);

  auto column = [](const ggml_tensor* tensor, int64_t length, int64_t i1, int64_t i2 = 0) {
    require(tensor->type == GGML_TYPE_F32 && tensor->nb[0] == sizeof(float), "reference output is not a float column");
    std::vector<float> values(static_cast<size_t>(length));
    const auto* data = static_cast<const uint8_t*>(tensor->data) + i1 * tensor->nb[1] + i2 * tensor->nb[2];
    std::memcpy(values.data(), data, values.size() * sizeof(float));
    for (float value : values) require(std::isfinite(value), "non-finite native reference output");
    return values;
  };
  for (int64_t token = 0; token < tokens; ++token) {
    pmi::MoeLayerTrace trace;
    trace.router_logits = column(logits, experts, token);
    trace.router_probabilities = column(probabilities, experts, token);
    trace.expert_weights = column(weights, used, token);
    for (int64_t slot = 0; slot < used; ++slot) {
      int32_t id;
      std::memcpy(&id, static_cast<const uint8_t*>(ids->data) + slot * ids->nb[0] + token * ids->nb[1], sizeof(id));
      require(id >= 0 && id < experts, "native reference router returned an invalid expert");
      trace.expert_ids.push_back(id);
      trace.experts.push_back({id, column(gate, hidden, slot, token), column(up, hidden, slot, token),
                              column(activation, hidden, slot, token), column(down, dim, slot, token)});
    }
    trace.routed_output = column(routed, dim, token);
    trace.shared = {-1, column(sg, shared_hidden, token), column(su, shared_hidden, token),
                    column(sa, shared_hidden, token), column(sd, dim, token)};
    trace.shared_gate_logit = column(shared_gate_logit, 1, token)[0];
    trace.shared_gate = column(shared_gate, 1, token)[0];
    trace.output = column(output, dim, token);
    result.tokens.push_back(std::move(trace));
  }
  result.memory = pmi::process_memory();
  return result; // Full resident weights are freed before the paged comparison.
}

struct Difference {
  uint64_t values{}, mismatches{}, id_mismatches{};
  double max_absolute_error{}, max_tolerance_fraction{};
};
using Differences = std::map<std::string, Difference>;
void compare_values(Differences& report, const std::string& name,
                    const std::vector<float>& expected, const std::vector<float>& actual) {
  require(expected.size() == actual.size(), "comparison size mismatch: " + name);
  auto& item = report[name];
  for (size_t i = 0; i < expected.size(); ++i) {
    require(std::isfinite(expected[i]) && std::isfinite(actual[i]), "non-finite comparison value: " + name);
    const double error = std::abs(static_cast<double>(actual[i]) - expected[i]);
    const double tolerance = absolute_tolerance + relative_tolerance * std::abs(expected[i]);
    ++item.values; item.mismatches += error > tolerance;
    item.max_absolute_error = std::max(item.max_absolute_error, error);
    item.max_tolerance_fraction = std::max(item.max_tolerance_fraction, error / tolerance);
  }
}
void compare_trace(Differences& report, const pmi::MoeLayerTrace& reference, const pmi::MoeLayerTrace& paged) {
  require(reference.expert_ids.size() == paged.expert_ids.size(), "routing comparison size mismatch");
  for (size_t slot = 0; slot < reference.expert_ids.size(); ++slot)
    report["routing"].id_mismatches += reference.expert_ids[slot] != paged.expert_ids[slot];
  compare_values(report, "router_logits", reference.router_logits, paged.router_logits);
  compare_values(report, "router_probabilities", reference.router_probabilities, paged.router_probabilities);
  compare_values(report, "expert_weights", reference.expert_weights, paged.expert_weights);
  require(reference.experts.size() == paged.experts.size(), "expert comparison size mismatch");
  auto expert = [&](const std::string& prefix, const pmi::MoeExpertTrace& a, const pmi::MoeExpertTrace& b) {
    compare_values(report, prefix + "gate", a.gate, b.gate);
    compare_values(report, prefix + "up", a.up, b.up);
    compare_values(report, prefix + "activation", a.activation, b.activation);
    compare_values(report, prefix + "down", a.down, b.down);
  };
  for (size_t slot = 0; slot < reference.experts.size(); ++slot) expert("expert_", reference.experts[slot], paged.experts[slot]);
  expert("shared_", reference.shared, paged.shared);
  compare_values(report, "routed_output", reference.routed_output, paged.routed_output);
  compare_values(report, "shared_gate", {reference.shared_gate_logit, reference.shared_gate}, {paged.shared_gate_logit, paged.shared_gate});
  compare_values(report, "output", reference.output, paged.output);
}

template<class T> void array(std::ostream& out, const std::vector<T>& values) {
  out << '['; for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; } out << ']';
}
void expert_json(std::ostream& out, const pmi::MoeExpertTrace& expert) {
  out << "{\"id\":" << expert.expert_id << ",\"gate\":"; array(out, expert.gate);
  out << ",\"up\":"; array(out, expert.up); out << ",\"activation\":"; array(out, expert.activation);
  out << ",\"down\":"; array(out, expert.down); out << '}';
}
void trace_json(std::ostream& out, const pmi::MoeLayerTrace& trace, bool intermediates) {
  out << "{\"expert_ids\":"; array(out, trace.expert_ids);
  out << ",\"router_logits\":"; array(out, trace.router_logits);
  out << ",\"router_probabilities\":"; array(out, trace.router_probabilities);
  out << ",\"expert_weights\":"; array(out, trace.expert_weights);
  out << ",\"output\":"; array(out, trace.output);
  if (intermediates) {
    out << ",\"routed_output\":"; array(out, trace.routed_output);
    out << ",\"experts\":[";
    for (size_t i = 0; i < trace.experts.size(); ++i) { if (i) out << ','; expert_json(out, trace.experts[i]); }
    out << "],\"shared_gate_logit\":" << trace.shared_gate_logit << ",\"shared_gate\":" << trace.shared_gate << ",\"shared\":";
    expert_json(out, trace.shared);
  }
  out << '}';
}
void optional(std::ostream& out, const std::optional<uint64_t>& value) {
  if (value) out << *value; else out << "null";
}
} // namespace

int main(int argc, char** argv) {
  try {
    std::vector<fs::path> paths;
    fs::path input_path;
    uint64_t layer = 0, capacity = 64 * 1024 * 1024;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument.rfind("--", 0) != 0) paths.emplace_back(argument);
      else {
        require(++i < argc, "missing option value: " + argument);
        if (argument == "--layer") layer = integer(argv[i]);
        else if (argument == "--input") input_path = argv[i];
        else if (argument == "--cache-bytes") capacity = integer(argv[i]);
        else throw std::runtime_error("unknown option: " + argument);
      }
    }
    require(!paths.empty() && !input_path.empty() && layer <= INT32_MAX && capacity > 0 && capacity <= SIZE_MAX &&
            capacity < accounted_limit - auxiliary_allowance,
            "usage: pmi_target_moe_reference SHARD.gguf [SHARD.gguf ...] --layer N --input FLOATS.txt [--cache-bytes N]");
    std::vector<pmi::GgufFile> files;
    for (const auto& path : paths) files.push_back(pmi::GgufReader(path).inspect());
    const auto manifest = pmi::build_manifest(files);
    paths.clear(); for (const auto& shard : manifest.shards) paths.push_back(shard.source_path);
    const auto prefix = "blk." + std::to_string(layer) + ".";
    const auto router = find_physical(files, prefix + "ffn_gate_inp.weight");
    const auto gate = find_physical(files, prefix + "ffn_gate_exps.weight");
    const auto shared = find_physical(files, prefix + "ffn_up_shexp.weight");
    require(router.info->shape.size() == 2 && gate.info->shape.size() == 3 && shared.info->shape.size() == 2,
            "unsupported native MoE physical ranks");
    pmi::Qwen4ExpMoeConfig config;
    config.layer = static_cast<int>(layer);
    config.embedding_length = router.info->shape[0]; config.expert_count = router.info->shape[1];
    config.expert_hidden_length = gate.info->shape[1]; config.shared_hidden_length = shared.info->shape[1];
    for (const auto& file : files) if (file.source_path == manifest.shards.front().source_path) {
      const auto used = file.metadata.find("qwen4exp.expert_used_count");
      require(used != file.metadata.end(), "GGUF metadata must declare qwen4exp.expert_used_count");
      config.experts_used = integer(used->second.text);
      if (const auto scale = file.metadata.find("qwen4exp.expert_weights_scale"); scale != file.metadata.end()) {
        size_t consumed{}; config.expert_weights_scale = std::stof(scale->second.text, &consumed);
        require(consumed == scale->second.text.size(), "invalid expert weight scale metadata");
      }
      if (const auto groups = file.metadata.find("qwen4exp.expert_group_count"); groups != file.metadata.end())
        config.expert_groups = integer(groups->second.text);
    }
    pmi::Qwen4ExpMoeConnector connector(manifest, config);
    require(config.embedding_length <= 1024 * 1024, "input embedding exceeds validation harness bound");
    const auto minimum_capacity = (connector.largest_weight_page_bytes() + 63) & ~uint64_t{63};
    require(capacity >= minimum_capacity, "cache must fit the largest MoE weight page: " + std::to_string(minimum_capacity));
    std::ifstream input_file(input_path);
    require(static_cast<bool>(input_file), "cannot open input vectors");
    std::vector<float> input;
    float value;
    for (;;) {
      input_file >> std::ws;
      if (input_file.eof()) break;
      require(static_cast<bool>(input_file >> value), "malformed float in input vectors");
      require(std::isfinite(value) && input.size() < 2 * config.embedding_length, "input must contain one or two finite token vectors");
      input.push_back(value);
    }
    require(input_file.eof() && !input.empty() && input.size() % config.embedding_length == 0,
            "input must contain one or two complete whitespace-separated token columns");

    ggml_cpu_init();
    const auto reference = native_reference(files, config, input, capacity);
    pmi::FilePageProvider provider(paths, static_cast<size_t>(capacity));
    std::vector<pmi::MoeLayerTrace> paged;
    Differences differences;
    const auto paged_start = Clock::now();
    for (size_t token = 0; token < reference.tokens.size(); ++token) {
      const auto start = input.begin() + token * config.embedding_length;
      paged.push_back(connector.forward(provider, {start, start + config.embedding_length}));
      compare_trace(differences, reference.tokens[token], paged.back());
    }
    const double paged_ms = elapsed(paged_start);
    const auto cache = provider.stats();
    require(cache.active_leases == 0 && cache.peak_bytes <= capacity, "paged validation leaked leases or exceeded cache capacity");
    const auto memory = pmi::process_memory();
    Difference total;
    for (const auto& entry : differences) {
      total.values += entry.second.values; total.mismatches += entry.second.mismatches; total.id_mismatches += entry.second.id_mismatches;
      total.max_absolute_error = std::max(total.max_absolute_error, entry.second.max_absolute_error);
      total.max_tolerance_fraction = std::max(total.max_tolerance_fraction, entry.second.max_tolerance_fraction);
    }
    const bool match = total.mismatches == 0 && total.id_mismatches == 0;
    std::cout.imbue(std::locale::classic());
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    std::cout << "{\"scope\":\"real_weight_moe_sublayer_given_input\",\"reference\":\"native_resident_ggml_mul_mat_id\",\"match\":"
              << (match ? "true" : "false") << ",\"layer\":" << layer << ",\"token_count\":" << reference.tokens.size()
              << ",\"absolute_tolerance\":" << absolute_tolerance << ",\"relative_tolerance\":" << relative_tolerance
              << ",\"compared_floats\":" << total.values << ",\"float_mismatches\":" << total.mismatches
              << ",\"id_mismatches\":" << total.id_mismatches << ",\"max_abs_error\":" << total.max_absolute_error
              << ",\"max_tolerance_fraction\":" << total.max_tolerance_fraction
              << ",\"reference_weight_bytes\":" << reference.weight_bytes << ",\"reference_backend_buffer_bytes\":" << reference.buffer_bytes
              << ",\"reference_workspace_bytes\":" << reference.workspace_bytes << ",\"planned_reference_cache_bytes\":" << reference.accounted_bytes
              << ",\"plan_limit_bytes\":" << accounted_limit << ",\"auxiliary_allowance_bytes\":" << auxiliary_allowance
              << ",\"reference_load_ms\":" << reference.load_ms << ",\"reference_compute_ms\":" << reference.compute_ms
              << ",\"paged_compute_ms\":" << paged_ms << ",\"paged_io_bytes\":" << cache.io_bytes
              << ",\"cache_hits\":" << cache.hits << ",\"cache_misses\":" << cache.misses
              << ",\"cache_evictions\":" << cache.evictions << ",\"cache_capacity_bytes\":" << capacity
              << ",\"cache_peak_reserved_bytes\":" << cache.peak_bytes << ",\"reference_rss_bytes\":";
    optional(std::cout, reference.memory.rss_bytes);
    std::cout << ",\"process_rss_bytes\":"; optional(std::cout, memory.rss_bytes);
    std::cout << ",\"process_peak_rss_bytes\":"; optional(std::cout, memory.peak_rss_bytes);
    std::cout << ",\"sections\":{";
    bool first = true;
    for (const auto& entry : differences) {
      if (!first) std::cout << ',';
      first = false;
      std::cout << '"' << entry.first << "\":{\"values\":" << entry.second.values
                << ",\"mismatches\":" << entry.second.mismatches << ",\"id_mismatches\":" << entry.second.id_mismatches
                << ",\"max_abs_error\":" << entry.second.max_absolute_error << '}';
    }
    std::cout << "},\"tokens\":[";
    for (size_t token = 0; token < reference.tokens.size(); ++token) {
      if (token) std::cout << ',';
      std::cout << "{\"index\":" << token << ",\"native\":"; trace_json(std::cout, reference.tokens[token], true);
      std::cout << ",\"paged\":"; trace_json(std::cout, paged[token], false); std::cout << '}';
    }
    std::cout << "]}\n";
    require(static_cast<bool>(std::cout), "failed writing validation JSON");
    return match ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "target MoE reference failed: " << error.what() << '\n'; return 1;
  }
}
