#include "pmi/connector.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#ifdef PMI_WITH_LLAMA
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#endif

namespace pmi {
namespace {

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("qwen4exp MoE connector: " + what);
}

void require_finite(const std::vector<float>& values, const char* name) {
  for (float value : values) if (!std::isfinite(value)) fail(std::string(name) + " contains non-finite values");
}

void validate_tensor(const ManifestTensor& tensor, const std::vector<uint64_t>& shape) {
  if (tensor.shape != shape) fail("unexpected shape for " + tensor.name);
  if (tensor.byte_length != ggml_tensor_nbytes(tensor.ggml_type, shape)) fail("unexpected byte length for " + tensor.name);
  if (tensor.absolute_offset > UINT64_MAX - tensor.byte_length) fail("offset overflow for " + tensor.name);
#ifdef PMI_WITH_LLAMA
  if (tensor.ggml_type >= GGML_TYPE_COUNT) fail("unsupported ggml type for " + tensor.name);
  const auto type = static_cast<ggml_type>(tensor.ggml_type);
  const auto* traits = ggml_get_type_traits(type);
  const auto* cpu = ggml_get_type_traits_cpu(type);
  if (!traits || traits->blck_size <= 0 || !traits->type_size || !cpu || !cpu->vec_dot) {
    fail("ggml CPU matrix multiplication does not support " + tensor.name);
  }
  if (shape[0] % static_cast<uint64_t>(traits->blck_size)) fail("quantized row is not block aligned: " + tensor.name);
#else
  (void) tensor;
#endif
}

#ifdef PMI_WITH_LLAMA
// Tensor descriptors and activations belong to ggml. Weight storage belongs to
// PageProvider and is never copied into a ggml allocation or mutated by a graph.
class CpuGraph {
 public:
  explicit CpuGraph(int threads) : threads_(threads) {
    ggml_cpu_init();
    ggml_init_params params{};
    params.mem_size = 256 * ggml_tensor_overhead() + ggml_graph_overhead_custom(256, false);
    params.no_alloc = true;
    ctx_ = ggml_init(params);
    if (!ctx_) fail("cannot allocate ggml graph context");
  }
  ~CpuGraph() { ggml_backend_buffer_free(buffer_); ggml_free(ctx_); }
  CpuGraph(const CpuGraph&) = delete;
  CpuGraph& operator=(const CpuGraph&) = delete;
  ggml_context* ctx() const { return ctx_; }
  ggml_tensor* input(const std::vector<float>& values) {
    auto* tensor = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, static_cast<int64_t>(values.size()), 1);
    tensor->data = const_cast<float*>(values.data());
    return tensor;
  }
  void compute(ggml_tensor* output) {
    auto* graph = ggml_new_graph_custom(ctx_, 256, false);
    ggml_build_forward_expand(graph, output);
    buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_, ggml_backend_cpu_buffer_type());
    if (!buffer_) fail("cannot allocate ggml activations");
    auto plan = ggml_graph_plan(graph, threads_, nullptr);
    std::vector<uint8_t> workspace(plan.work_size);
    plan.work_data = workspace.data();
    if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) fail("ggml CPU graph failed");
  }
  static std::vector<float> floats(const ggml_tensor* tensor) {
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) fail("noncontiguous ggml result");
    const auto* data = static_cast<const float*>(tensor->data);
    std::vector<float> values(data, data + ggml_nelements(tensor));
    require_finite(values, "ggml result");
    return values;
  }
 private:
  int threads_;
  ggml_context* ctx_{};
  ggml_backend_buffer_t buffer_{};
};

std::vector<float> project(PageProvider& provider, const ManifestTensor& tensor,
                           const std::vector<float>& input, int threads,
                           MoeLayerTrace& trace) {
  auto page = provider.request(tensor).get();
  if (!page || page->size() != tensor.byte_length) fail("provider returned incorrect page length for " + tensor.name);
  if (reinterpret_cast<uintptr_t>(page->data()) % alignof(float)) fail("provider returned unaligned page for " + tensor.name);
  trace.largest_weight_page_bytes = std::max(trace.largest_weight_page_bytes, tensor.byte_length);
  ++trace.weight_page_requests;
  CpuGraph graph(threads);
  const int64_t rows = tensor.shape.size() == 1 ? 1 : static_cast<int64_t>(tensor.shape[1]);
  auto* weight = ggml_new_tensor_2d(graph.ctx(), static_cast<ggml_type>(tensor.ggml_type),
                                  static_cast<int64_t>(tensor.shape[0]), rows);
  weight->data = const_cast<uint8_t*>(page->data());
  auto* result = ggml_mul_mat(graph.ctx(), weight, graph.input(input));
  graph.compute(result);
  return CpuGraph::floats(result);
  // Graph is destroyed before page; no graph retains a dangling weight pointer.
}

void route(MoeLayerTrace& trace, const Qwen4ExpMoeConfig& config) {
  CpuGraph graph(config.cpu_threads);
  auto* ctx = graph.ctx();
  auto* probabilities = ggml_soft_max(ctx, graph.input(trace.router_logits));
  auto* selected = ggml_argsort_top_k(ctx, probabilities, static_cast<int>(config.experts_used));
  auto* weights = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probabilities, 1,
                              static_cast<int64_t>(config.expert_count), 1), selected);
  weights = ggml_reshape_2d(ctx, weights, static_cast<int64_t>(config.experts_used), 1);
  auto* denominator = ggml_clamp(ctx, ggml_sum_rows(ctx, weights), 6.103515625e-5f,
                                 std::numeric_limits<float>::infinity());
  weights = ggml_div(ctx, weights, denominator);
  if (config.expert_weights_scale != 0.0f && config.expert_weights_scale != 1.0f) {
    weights = ggml_scale(ctx, weights, config.expert_weights_scale);
  }
  graph.compute(weights);
  trace.router_probabilities = CpuGraph::floats(probabilities);
  trace.expert_weights = CpuGraph::floats(weights);
  const auto* ids = static_cast<const int32_t*>(selected->data);
  trace.expert_ids.assign(ids, ids + config.experts_used);
  for (auto id : trace.expert_ids) if (id < 0 || static_cast<uint64_t>(id) >= config.expert_count) fail("invalid ggml routing result");
}

std::vector<float> swiglu(const std::vector<float>& gate, const std::vector<float>& up, int threads) {
  CpuGraph graph(threads);
  auto* result = ggml_swiglu_split(graph.ctx(), graph.input(gate), graph.input(up));
  graph.compute(result);
  return CpuGraph::floats(result);
}

std::vector<float> weighted_add(const std::vector<float>& sum, const std::vector<float>& next,
                                float weight, int threads) {
  CpuGraph graph(threads);
  auto* scaled = ggml_scale(graph.ctx(), graph.input(next), weight);
  auto* result = sum.empty() ? scaled : ggml_add(graph.ctx(), graph.input(sum), scaled);
  graph.compute(result);
  return CpuGraph::floats(result);
}

float sigmoid(float logit, int threads) {
  const std::vector<float> value{logit};
  CpuGraph graph(threads);
  auto* result = ggml_sigmoid(graph.ctx(), graph.input(value));
  graph.compute(result);
  return CpuGraph::floats(result)[0];
}
#endif

} // namespace

Qwen4ExpMoeConnector::Qwen4ExpMoeConnector(const ModelManifest& manifest, Qwen4ExpMoeConfig config)
    : config_(std::move(config)) {
  if (manifest.model_architecture != "qwen4exp") fail("unsupported architecture: " + manifest.model_architecture);
  if (config_.layer < 0 || !config_.embedding_length || !config_.expert_count ||
      !config_.experts_used || !config_.expert_hidden_length || !config_.shared_hidden_length ||
      config_.experts_used > config_.expert_count || config_.expert_count > INT32_MAX ||
      config_.embedding_length > INT32_MAX || config_.expert_hidden_length > INT32_MAX ||
      config_.shared_hidden_length > INT32_MAX || config_.cpu_threads < 1) fail("invalid layer dimensions or CPU thread count");
  if (config_.activation != "silu" || config_.routing != "softmax" ||
      config_.expert_groups != 1 || !config_.normalize_topk || config_.swiglu_clamp != 0.0f) {
    fail("unsupported activation, routing, expert groups, normalization, or SwiGLU clamp");
  }
  if (!std::isfinite(config_.expert_weights_scale) || config_.expert_weights_scale < 0.0f) fail("invalid expert weight scale");
  const std::string prefix = "blk." + std::to_string(config_.layer) + ".";
  const std::unordered_set<std::string> expected{
    prefix + "ffn_gate_inp.weight", prefix + "ffn_gate_exps.weight", prefix + "ffn_up_exps.weight",
    prefix + "ffn_down_exps.weight", prefix + "ffn_gate_inp_shexp.weight", prefix + "ffn_gate_shexp.weight",
    prefix + "ffn_up_shexp.weight", prefix + "ffn_down_shexp.weight"};
  std::unordered_map<std::string, std::unordered_map<int, const ManifestTensor*>> index;
  for (const auto& tensor : manifest.tensors) {
    if ((tensor.source_tensor_name.rfind(prefix + "ffn_", 0) == 0 ||
         tensor.source_tensor_name.rfind(prefix + "exp_probs", 0) == 0) &&
        !expected.count(tensor.source_tensor_name)) fail("unsupported FFN tensor/layout: " + tensor.source_tensor_name);
    if (expected.count(tensor.source_tensor_name)) {
      if (!index[tensor.source_tensor_name].emplace(tensor.expert_id, &tensor).second) {
        fail("duplicate tensor slice: " + tensor.source_tensor_name);
      }
      if (!manifest.shards.empty()) {
        if (tensor.shard_id >= manifest.shards.size()) fail("tensor shard is absent from manifest: " + tensor.name);
        const auto& shard = manifest.shards[tensor.shard_id];
        if (shard.shard_id != tensor.shard_id || tensor.absolute_offset < shard.tensor_data_offset ||
            tensor.offset != tensor.absolute_offset - shard.tensor_data_offset || tensor.absolute_offset > shard.file_size ||
            tensor.byte_length > shard.file_size - tensor.absolute_offset) fail("tensor range is outside its manifest shard: " + tensor.name);
      }
    }
  }
  auto find = [&](const std::string& suffix, int expert_id, const std::vector<uint64_t>& shape) {
    const auto source = prefix + suffix;
    const auto source_it = index.find(source);
    if (source_it == index.end() || !source_it->second.count(expert_id)) fail("missing tensor/expert slice: " + source + " expert=" + std::to_string(expert_id));
    const ManifestTensor* result = source_it->second.at(expert_id);
    if (result->layer != config_.layer) fail("incorrect tensor layer: " + source);
    validate_tensor(*result, shape);
    return *result;
  };
  // Reject extra IDs instead of silently ignoring malformed manifest slices.
  for (const auto& tensor : manifest.tensors) if (expected.count(tensor.source_tensor_name)) {
    const bool routed = tensor.source_tensor_name.find("_exps.weight") != std::string::npos;
    if ((routed && (tensor.expert_id < 0 || static_cast<uint64_t>(tensor.expert_id) >= config_.expert_count)) ||
        (!routed && tensor.expert_id != -1)) fail("unexpected expert ID for " + tensor.source_tensor_name);
  }
  const auto dim = config_.embedding_length;
  const auto hidden = config_.expert_hidden_length;
  router_ = find("ffn_gate_inp.weight", -1, {dim, config_.expert_count});
  shared_gate_ = find("ffn_gate_inp_shexp.weight", -1, {dim});
  experts_.reserve(static_cast<size_t>(config_.expert_count));
  for (uint64_t id = 0; id < config_.expert_count; ++id) {
    experts_.push_back({find("ffn_gate_exps.weight", static_cast<int>(id), {dim, hidden}),
                        find("ffn_up_exps.weight", static_cast<int>(id), {dim, hidden}),
                        find("ffn_down_exps.weight", static_cast<int>(id), {hidden, dim})});
  }
  shared_ = {find("ffn_gate_shexp.weight", -1, {dim, config_.shared_hidden_length}),
             find("ffn_up_shexp.weight", -1, {dim, config_.shared_hidden_length}),
             find("ffn_down_shexp.weight", -1, {config_.shared_hidden_length, dim})};
}

MoeLayerTrace Qwen4ExpMoeConnector::forward(PageProvider& provider, const std::vector<float>& input) const {
#ifndef PMI_WITH_LLAMA
  (void) provider;
  (void) input;
  fail("execution requires PMI_WITH_LLAMA and the retained ggml CPU backend");
#else
  if (input.size() != config_.embedding_length) fail("input must contain exactly one token embedding");
  require_finite(input, "input");
  MoeLayerTrace trace;
  trace.router_logits = project(provider, router_, input, config_.cpu_threads, trace);
  route(trace, config_);
  auto expert = [&](const Expert& weights, int32_t id) {
    MoeExpertTrace out;
    out.expert_id = id;
    out.gate = project(provider, weights.gate, input, config_.cpu_threads, trace);
    out.up = project(provider, weights.up, input, config_.cpu_threads, trace);
    out.activation = swiglu(out.gate, out.up, config_.cpu_threads);
    out.down = project(provider, weights.down, out.activation, config_.cpu_threads, trace);
    return out;
  };
  for (size_t i = 0; i < trace.expert_ids.size(); ++i) {
    trace.experts.push_back(expert(experts_[static_cast<size_t>(trace.expert_ids[i])], trace.expert_ids[i]));
    trace.routed_output = weighted_add(trace.routed_output, trace.experts.back().down,
                                      trace.expert_weights[i], config_.cpu_threads);
  }
  trace.shared = expert(shared_, -1);
  trace.shared_gate_logit = project(provider, shared_gate_, input, config_.cpu_threads, trace)[0];
  trace.shared_gate = sigmoid(trace.shared_gate_logit, config_.cpu_threads);
  trace.output = weighted_add(trace.routed_output, trace.shared.down, trace.shared_gate, config_.cpu_threads);
  return trace;
#endif
}

uint64_t Qwen4ExpMoeConnector::largest_weight_page_bytes() const {
  uint64_t largest = std::max(router_.byte_length, shared_gate_.byte_length);
  auto include = [&](const Expert& expert) {
    largest = std::max({largest, expert.gate.byte_length, expert.up.byte_length, expert.down.byte_length});
  };
  for (const auto& expert : experts_) include(expert);
  include(shared_);
  return largest;
}

uint64_t Qwen4ExpMoeConnector::total_weight_bytes() const {
  uint64_t total = 0;
  auto add = [&](uint64_t bytes) {
    if (total > UINT64_MAX - bytes) fail("total weight size overflow");
    total += bytes;
  };
  add(router_.byte_length);
  add(shared_gate_.byte_length);
  auto include = [&](const Expert& expert) { add(expert.gate.byte_length); add(expert.up.byte_length); add(expert.down.byte_length); };
  for (const auto& expert : experts_) include(expert);
  include(shared_);
  return total;
}

} // namespace pmi
