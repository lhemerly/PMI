#include "pmi/connector.hpp"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
using pmi::MoeLayerTrace;

struct ComparisonMetrics {
  double max_absolute_error{};
  double max_tolerance_fraction{};
  uint64_t scalar_values_compared{};
} metrics;

void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
void expect_error(const std::function<void()>& fn, const std::string& message) {
  try { fn(); } catch (const std::exception&) { return; }
  throw std::runtime_error("expected error: " + message);
}
void close(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& name,
           float atol = 3e-7f, float rtol = 3e-5f) {
  check(actual.size() == expected.size(), name + " length");
  for (size_t i = 0; i < actual.size(); ++i) {
    const double error = std::abs(static_cast<double>(actual[i]) - expected[i]);
    const double tolerance = atol + rtol * std::abs(expected[i]);
    metrics.max_absolute_error = std::max(metrics.max_absolute_error, error);
    metrics.max_tolerance_fraction = std::max(metrics.max_tolerance_fraction, error / tolerance);
    ++metrics.scalar_values_compared;
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
        std::abs(actual[i] - expected[i]) > atol + rtol * std::abs(expected[i])) {
      throw std::runtime_error(name + " at " + std::to_string(i) + ": " + std::to_string(actual[i]) +
                               " != " + std::to_string(expected[i]));
    }
  }
}
void compare(const MoeLayerTrace& actual, const MoeLayerTrace& expected) {
  check(actual.expert_ids == expected.expert_ids, "selected expert IDs");
  close(actual.router_logits, expected.router_logits, "router logits");
  close(actual.router_probabilities, expected.router_probabilities, "router probabilities");
  close(actual.expert_weights, expected.expert_weights, "normalized scaled top-k weights");
  check(actual.experts.size() == expected.experts.size(), "expert trace count");
  auto expert = [](const auto& a, const auto& b) {
    check(a.expert_id == b.expert_id, "expert trace ID");
    close(a.gate, b.gate, "gate projection");
    close(a.up, b.up, "up projection");
    close(a.activation, b.activation, "SwiGLU activation");
    close(a.down, b.down, "down projection");
  };
  for (size_t i = 0; i < actual.experts.size(); ++i) expert(actual.experts[i], expected.experts[i]);
  expert(actual.shared, expected.shared);
  close(actual.routed_output, expected.routed_output, "weighted routed output");
  close({actual.shared_gate_logit, actual.shared_gate}, {expected.shared_gate_logit, expected.shared_gate}, "shared sigmoid gate");
  close(actual.output, expected.output, "final MoE output");
}

template<class T> void write(std::ostream& out, T value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
void string(std::ostream& out, const std::string& value) {
  write<uint64_t>(out, value.size()); out.write(value.data(), static_cast<std::streamsize>(value.size()));
}
template<class T> void metadata(std::ostream& out, const std::string& name, pmi::GgufType type, T value) {
  string(out, name); write<uint32_t>(out, static_cast<uint32_t>(type)); write<T>(out, value);
}
uint64_t align(uint64_t value) { return (value + 4095) & ~uint64_t{4095}; }

struct FixtureTensor {
  std::string name;
  std::vector<uint64_t> shape;
  ggml_type type{};
  std::vector<float> original;
  std::vector<uint8_t> encoded;
  uint32_t shard{};
  uint64_t offset{};
};
struct Fixture {
  pmi::Qwen4ExpMoeConfig config;
  std::vector<FixtureTensor> tensors;
  std::vector<fs::path> paths;
  std::vector<float> input;
  pmi::ModelManifest manifest;
  const FixtureTensor& get(const std::string& suffix) const {
    const auto name = "blk.0." + suffix + ".weight";
    for (const auto& tensor : tensors) if (tensor.name == name) return tensor;
    throw std::runtime_error("fixture tensor missing " + name);
  }
};

Fixture make_fixture(const fs::path& directory, ggml_type type, bool target_mix = false) {
  fs::create_directories(directory);
  Fixture f;
  // IQ4_XS and IQ3_S need 256-value input-row blocks.
  const uint64_t dim = std::max<int64_t>(32, ggml_blck_size(type));
  f.config.embedding_length = dim;
  f.config.expert_hidden_length = target_mix ? 128 : dim;
  f.config.shared_hidden_length = target_mix ? 64 : dim;
  f.config.expert_count = 4;
  f.config.experts_used = 2;
  f.config.expert_weights_scale = 1.7f;
  f.input.resize(dim);
  for (uint64_t i = 0; i < dim; ++i) f.input[i] = 0.5f * std::cos(static_cast<float>(i) * 0.37f);
  f.input[0] = 1.25f;
  auto add = [&](const std::string& suffix, const std::vector<uint64_t>& shape, uint32_t shard, ggml_type storage) {
    FixtureTensor tensor;
    tensor.name = "blk.0." + suffix + ".weight";
    tensor.shape = shape; tensor.shard = shard; tensor.type = storage;
    const auto count = std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<uint64_t>{});
    tensor.original.resize(count);
    const float phase = static_cast<float>(f.tensors.size() + 1) * 0.21f;
    for (uint64_t i = 0; i < count; ++i) {
      tensor.original[i] = (std::sin(static_cast<float>(i % 997) * 0.79f + phase) +
                            0.5f * std::cos(static_cast<float>(i % 331) * 0.17f - phase)) /
                           std::sqrt(static_cast<float>(dim));
    }
    if (suffix == "ffn_gate_inp") {
      std::fill(tensor.original.begin(), tensor.original.end(), 0.0f);
      const float router[] = {-0.2f, 1.4f, 0.7f, -0.8f};
      for (size_t expert = 0; expert < 4; ++expert) tensor.original[expert * dim] = router[expert];
    }
    tensor.encoded.resize(pmi::ggml_tensor_nbytes(storage, shape));
    if (storage == GGML_TYPE_F32) {
      std::memcpy(tensor.encoded.data(), tensor.original.data(), tensor.encoded.size());
    } else {
      const auto bytes = ggml_quantize_chunk(storage, tensor.original.data(), tensor.encoded.data(), 0,
                                             static_cast<int64_t>(count / shape[0]), static_cast<int64_t>(shape[0]), nullptr);
      check(bytes == tensor.encoded.size(), "quantized fixture length");
    }
    f.tensors.push_back(std::move(tensor));
  };
  // Both shards begin with a packed expert tensor at absolute byte 4096.
  add("ffn_gate_exps", {dim, f.config.expert_hidden_length, 4}, 0, type);
  add("ffn_up_exps", {dim, f.config.expert_hidden_length, 4}, 1, type);
  add("ffn_down_exps", {f.config.expert_hidden_length, dim, 4}, 1, target_mix ? GGML_TYPE_IQ4_NL : type);
  add("ffn_gate_inp", {dim, 4}, 0, GGML_TYPE_F32);
  add("ffn_gate_inp_shexp", {dim}, 0, GGML_TYPE_F32);
  const auto shared_type = target_mix ? GGML_TYPE_Q8_0 : type;
  add("ffn_gate_shexp", {dim, f.config.shared_hidden_length}, 1, shared_type);
  add("ffn_up_shexp", {dim, f.config.shared_hidden_length}, 0, shared_type);
  add("ffn_down_shexp", {f.config.shared_hidden_length, dim}, 1, shared_type);
  f.paths = {directory / "qwen-moe-00001-of-00002.gguf", directory / "qwen-moe-00002-of-00002.gguf"};
  for (uint32_t shard = 0; shard < 2; ++shard) {
    uint64_t offset = 0, count = 0;
    for (auto& tensor : f.tensors) if (tensor.shard == shard) {
      tensor.offset = offset; offset = align(offset + tensor.encoded.size()); ++count;
    }
    std::ofstream out(f.paths[shard], std::ios::binary);
    check(static_cast<bool>(out), "create fixture GGUF");
    write<uint32_t>(out, 0x46554747); write<uint32_t>(out, 3);
    write<uint64_t>(out, count); write<uint64_t>(out, 11);
    string(out, "general.architecture"); write<uint32_t>(out, 8); string(out, "qwen4exp");
    metadata(out, "general.alignment", pmi::GgufType::UINT32, uint32_t{4096});
    metadata(out, "split.no", pmi::GgufType::UINT16, static_cast<uint16_t>(shard));
    metadata(out, "split.count", pmi::GgufType::UINT16, uint16_t{2});
    metadata(out, "split.tensors.count", pmi::GgufType::INT32, int32_t{8});
    metadata(out, "qwen4exp.embedding_length", pmi::GgufType::UINT32, static_cast<uint32_t>(dim));
    metadata(out, "qwen4exp.expert_count", pmi::GgufType::UINT32, uint32_t{4});
    metadata(out, "qwen4exp.expert_used_count", pmi::GgufType::UINT32, uint32_t{2});
    metadata(out, "qwen4exp.expert_feed_forward_length", pmi::GgufType::UINT32, static_cast<uint32_t>(f.config.expert_hidden_length));
    metadata(out, "qwen4exp.expert_shared_feed_forward_length", pmi::GgufType::UINT32, static_cast<uint32_t>(f.config.shared_hidden_length));
    metadata(out, "qwen4exp.expert_weights_scale", pmi::GgufType::FLOAT32, f.config.expert_weights_scale);
    for (const auto& tensor : f.tensors) if (tensor.shard == shard) {
      string(out, tensor.name); write<uint32_t>(out, static_cast<uint32_t>(tensor.shape.size()));
      for (auto dimension : tensor.shape) write<uint64_t>(out, dimension);
      write<uint32_t>(out, static_cast<uint32_t>(tensor.type)); write<uint64_t>(out, tensor.offset);
    }
    check(static_cast<uint64_t>(out.tellp()) <= 4096, "fixture header fits first page");
    for (const auto& tensor : f.tensors) if (tensor.shard == shard) {
      out.seekp(static_cast<std::streamoff>(4096 + tensor.offset));
      out.write(reinterpret_cast<const char*>(tensor.encoded.data()), static_cast<std::streamsize>(tensor.encoded.size()));
    }
    check(static_cast<bool>(out), "write fixture GGUF");
  }
  std::ofstream input_file(directory / "input.txt");
  input_file << std::setprecision(9);
  for (float value : f.input) input_file << value << '\n';
  // Intentionally shuffled; manifest must canonicalize split IDs to provider paths.
  f.manifest = pmi::build_manifest({pmi::GgufReader(f.paths[1]).inspect(), pmi::GgufReader(f.paths[0]).inspect()});
  check(f.manifest.shards.size() == 2 && f.manifest.shards[0].source_path == fs::absolute(f.paths[0]), "canonical shard order");
  return f;
}

// Independent resident reference: all packed expert weights are ggml tensors,
// and mul_mat_id executes the selected experts together as in llama-graph.cpp.
// It never calls the PMI connector, PageProvider, or connector math helpers.
MoeLayerTrace resident(const Fixture& f, const std::vector<float>& input) {
  ggml_init_params params{};
  params.mem_size = 16 * 1024 * 1024;
  std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
  check(static_cast<bool>(ctx), "resident ggml context");
  auto* c = ctx.get();
  auto tensor = [&](const std::string& name) {
    const auto& source = f.get(name);
    std::vector<int64_t> shape(source.shape.begin(), source.shape.end());
    auto* value = ggml_new_tensor(c, source.type, static_cast<int>(shape.size()), shape.data());
    std::memcpy(value->data, source.encoded.data(), source.encoded.size());
    return value;
  };
  const auto dim = static_cast<int64_t>(f.config.embedding_length);
  const auto hidden = static_cast<int64_t>(f.config.expert_hidden_length);
  const auto used = static_cast<int64_t>(f.config.experts_used);
  auto* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, dim, 1);
  std::memcpy(x->data, input.data(), input.size() * sizeof(float));
  auto* logits = ggml_mul_mat(c, tensor("ffn_gate_inp"), x);
  auto* probabilities = ggml_soft_max(c, logits);
  auto* ids = ggml_argsort_top_k(c, probabilities, used);
  auto* probabilities_3d = ggml_reshape_3d(c, probabilities, 1, static_cast<int64_t>(f.config.expert_count), 1);
  auto* weights = ggml_reshape_2d(c, ggml_get_rows(c, probabilities_3d, ids), used, 1);
  weights = ggml_div(c, weights, ggml_clamp(c, ggml_sum_rows(c, weights), 6.103515625e-5f, INFINITY));
  if (f.config.expert_weights_scale != 0.0f && f.config.expert_weights_scale != 1.0f) {
    weights = ggml_scale(c, weights, f.config.expert_weights_scale);
  }
  auto* x3 = ggml_reshape_3d(c, x, dim, 1, 1);
  auto* gate = ggml_mul_mat_id(c, tensor("ffn_gate_exps"), x3, ids);
  auto* up = ggml_mul_mat_id(c, tensor("ffn_up_exps"), x3, ids);
  auto* activation = ggml_swiglu_split(c, gate, up);
  auto* down = ggml_mul_mat_id(c, tensor("ffn_down_exps"), activation, ids);
  auto* weighted = ggml_mul(c, down, ggml_reshape_3d(c, weights, 1, used, 1));
  auto* routed = ggml_view_2d(c, weighted, dim, 1, weighted->nb[2], 0);
  for (int64_t id = 1; id < used; ++id) {
    routed = ggml_add(c, routed, ggml_view_2d(c, weighted, dim, 1, weighted->nb[2], id * weighted->nb[1]));
  }
  auto* sg = ggml_mul_mat(c, tensor("ffn_gate_shexp"), x);
  auto* su = ggml_mul_mat(c, tensor("ffn_up_shexp"), x);
  auto* sa = ggml_swiglu_split(c, sg, su);
  auto* sd = ggml_mul_mat(c, tensor("ffn_down_shexp"), sa);
  auto* sgl = ggml_mul_mat(c, tensor("ffn_gate_inp_shexp"), x);
  auto* sgate = ggml_sigmoid(c, sgl);
  auto* output = ggml_add(c, routed, ggml_mul(c, sd, sgate));
  auto* graph = ggml_new_graph_custom(c, 256, false);
  ggml_build_forward_expand(graph, output);
  check(ggml_graph_compute_with_ctx(c, graph, 1) == GGML_STATUS_SUCCESS, "resident compute");
  auto values = [](const ggml_tensor* value, size_t first = 0, size_t length = 0) {
    const auto* data = static_cast<const float*>(value->data);
    if (!length) length = static_cast<size_t>(ggml_nelements(value));
    return std::vector<float>(data + first, data + first + length);
  };
  MoeLayerTrace out;
  out.router_logits = values(logits); out.router_probabilities = values(probabilities);
  out.expert_weights = values(weights);
  const auto* selected = static_cast<const int32_t*>(ids->data);
  out.expert_ids.assign(selected, selected + used);
  for (int64_t i = 0; i < used; ++i) {
    out.experts.push_back({selected[i], values(gate, i * hidden, hidden), values(up, i * hidden, hidden),
                          values(activation, i * hidden, hidden), values(down, i * dim, dim)});
  }
  out.routed_output = values(routed);
  out.shared = {-1, values(sg), values(su), values(sa), values(sd)};
  out.shared_gate_logit = values(sgl)[0]; out.shared_gate = values(sgate)[0]; out.output = values(output);
  return out;
}

// A scalar F32 oracle independently checks matrix orientation, gate/up order,
// router normalization/scale, and shared gating rather than only comparing two
// ggml execution arrangements with the same possible formula mistake.
MoeLayerTrace scalar(const Fixture& f) {
  auto multiply = [&](const std::string& name, const std::vector<float>& input, int expert = 0) {
    const auto& matrix = f.get(name);
    const auto columns = matrix.shape[0], rows = matrix.shape.size() == 1 ? 1 : matrix.shape[1];
    std::vector<float> result(rows);
    for (uint64_t row = 0; row < rows; ++row) {
      double sum = 0;
      for (uint64_t col = 0; col < columns; ++col) sum += static_cast<double>(matrix.original[expert * columns * rows + row * columns + col]) * input[col];
      result[row] = static_cast<float>(sum);
    }
    return result;
  };
  MoeLayerTrace out;
  out.router_logits = multiply("ffn_gate_inp", f.input);
  const float max = *std::max_element(out.router_logits.begin(), out.router_logits.end());
  float sum = 0;
  for (float logit : out.router_logits) { out.router_probabilities.push_back(std::exp(logit - max)); sum += out.router_probabilities.back(); }
  for (float& value : out.router_probabilities) value /= sum;
  out.expert_ids.resize(f.config.expert_count); std::iota(out.expert_ids.begin(), out.expert_ids.end(), 0);
  std::sort(out.expert_ids.begin(), out.expert_ids.end(), [&](int a, int b) { return out.router_probabilities[a] > out.router_probabilities[b]; });
  out.expert_ids.resize(f.config.experts_used);
  sum = 0; for (int id : out.expert_ids) sum += out.router_probabilities[id];
  const float scale = f.config.expert_weights_scale == 0 ? 1 : f.config.expert_weights_scale;
  for (int id : out.expert_ids) out.expert_weights.push_back(out.router_probabilities[id] / std::max(sum, 6.103515625e-5f) * scale);
  auto expert = [&](int id) {
    const auto suffix = id == -1 ? "_shexp" : "_exps";
    pmi::MoeExpertTrace e;
    e.expert_id = id;
    e.gate = multiply("ffn_gate" + std::string(suffix), f.input, std::max(id, 0));
    e.up = multiply("ffn_up" + std::string(suffix), f.input, std::max(id, 0));
    for (size_t i = 0; i < e.gate.size(); ++i) e.activation.push_back(e.gate[i] / (1 + std::exp(-e.gate[i])) * e.up[i]);
    e.down = multiply("ffn_down" + std::string(suffix), e.activation, std::max(id, 0));
    return e;
  };
  out.routed_output.resize(f.config.embedding_length);
  for (size_t i = 0; i < out.expert_ids.size(); ++i) {
    out.experts.push_back(expert(out.expert_ids[i]));
    for (size_t j = 0; j < out.routed_output.size(); ++j) out.routed_output[j] += out.experts.back().down[j] * out.expert_weights[i];
  }
  out.shared = expert(-1); out.shared_gate_logit = multiply("ffn_gate_inp_shexp", f.input)[0];
  out.shared_gate = 1 / (1 + std::exp(-out.shared_gate_logit)); out.output = out.routed_output;
  for (size_t i = 0; i < out.output.size(); ++i) out.output[i] += out.shared.down[i] * out.shared_gate;
  return out;
}

void exercise(Fixture& f) {
  metrics = {};
  pmi::Qwen4ExpMoeConnector connector(f.manifest, f.config);
  auto baseline = resident(f, f.input);
  check(baseline.expert_ids == std::vector<int32_t>({1, 2}), "nontrivial known routing");
  if (f.get("ffn_gate_exps").type == GGML_TYPE_F32) compare(baseline, scalar(f));
  // A cache large enough for all tensors measures actual warm hits. Include
  // per-allocation 64-byte padding required by the Rust storage contract.
  pmi::FilePageProvider warm(f.paths, static_cast<size_t>(connector.total_weight_bytes() + f.manifest.tensors.size() * 64));
  auto cold = connector.forward(warm, f.input); compare(cold, baseline);
  const auto after_cold = warm.stats();
  check(cold.weight_page_requests == 11 && after_cold.misses == 11, "only router, selected experts, and shared tensors are read");
  check(after_cold.active_leases == 0, "all graph page leases released after forward");
  auto hot = connector.forward(warm, f.input); compare(hot, baseline);
  const auto after_hot = warm.stats();
  check(after_hot.io_bytes == after_cold.io_bytes && after_hot.hits - after_cold.hits == 11, "warm graph uses cached weight pages without new reads");
  const auto capacity = (connector.largest_weight_page_bytes() + 63) & ~uint64_t{63};
  pmi::FilePageProvider evict(f.paths, static_cast<size_t>(capacity));
  compare(connector.forward(evict, f.input), baseline);
  auto threaded_config = f.config; threaded_config.cpu_threads = 2;
  pmi::Qwen4ExpMoeConnector threaded(f.manifest, threaded_config);
  compare(threaded.forward(evict, f.input), baseline);
  const auto first = evict.stats();
  compare(connector.forward(evict, f.input), baseline);
  const auto second = evict.stats();
  check(second.evictions > 0 && second.io_bytes > first.io_bytes, "forced eviction reloads weights");
  check(second.peak_bytes <= capacity && second.active_leases == 0, "bounded leased weight payload");
  auto changed_input = f.input; changed_input[0] = -1.25f;
  auto changed = connector.forward(evict, changed_input);
  check(changed.expert_ids == std::vector<int32_t>({3, 0}), "routing changes request different experts");
  compare(changed, resident(f, changed_input));
  compare(connector.forward(evict, f.input), baseline);

  // 0 is the upstream sentinel for identity scaling, rather than zero output.
  auto unscaled = f; unscaled.config.expert_weights_scale = 0;
  pmi::Qwen4ExpMoeConnector no_scale(unscaled.manifest, unscaled.config);
  compare(no_scale.forward(evict, f.input), resident(unscaled, f.input));
  // The connector fails through the provider, without retaining failed leases.
  pmi::FilePageProvider too_small(f.paths, 1);
  expect_error([&] { connector.forward(too_small, f.input); }, "capacity failure");
  check(too_small.stats().active_leases == 0, "capacity failure does not leak leases");
  expect_error([&] { connector.forward(warm, {}); }, "wrong embedding dimension");
  auto nan = f.input; nan[0] = std::numeric_limits<float>::quiet_NaN();
  expect_error([&] { connector.forward(warm, nan); }, "non-finite input");
  auto bad = f.manifest; bad.model_architecture = "qwen3next";
  expect_error([&] { pmi::Qwen4ExpMoeConnector other(bad, f.config); }, "architecture mismatch");
  bad = f.manifest; bad.tensors.pop_back();
  expect_error([&] { pmi::Qwen4ExpMoeConnector other(bad, f.config); }, "incomplete shared expert");
  bad = f.manifest; bad.tensors.push_back(bad.tensors.front());
  expect_error([&] { pmi::Qwen4ExpMoeConnector other(bad, f.config); }, "duplicate expert slice");
  bad = f.manifest; bad.tensors.front().shape[0] += 1;
  expect_error([&] { pmi::Qwen4ExpMoeConnector other(bad, f.config); }, "shape mismatch");
  bad = f.manifest; bad.tensors.front().byte_length -= 1;
  expect_error([&] { pmi::Qwen4ExpMoeConnector other(bad, f.config); }, "length mismatch");
  bad = f.manifest; auto extra = bad.tensors.front(); extra.source_tensor_name = "blk.0.ffn_gate_exps.scale"; bad.tensors.push_back(extra);
  expect_error([&] { pmi::Qwen4ExpMoeConnector other(bad, f.config); }, "external scale unsupported");
  auto bad_config = f.config; bad_config.routing = "sigmoid";
  expect_error([&] { pmi::Qwen4ExpMoeConnector other(f.manifest, bad_config); }, "wrong router semantics");
  bad_config = f.config; bad_config.activation = "gelu";
  expect_error([&] { pmi::Qwen4ExpMoeConnector other(f.manifest, bad_config); }, "wrong activation");
  // Read errors must also unwind every graph/page lease. Corrupt a copied file,
  // leaving preserved runnable fixture files intact when a directory is given.
  const auto truncated = f.paths[1].parent_path() / "truncated.gguf";
  fs::copy_file(f.paths[1], truncated, fs::copy_options::overwrite_existing);
  fs::resize_file(truncated, 4097);
  auto broken_paths = f.paths; broken_paths[1] = truncated;
  pmi::FilePageProvider broken(broken_paths, static_cast<size_t>(capacity));
  expect_error([&] { connector.forward(broken, f.input); }, "short weight read");
  check(broken.stats().active_leases == 0, "read failure does not leak leases");
  fs::remove(truncated);
  std::ofstream golden(f.paths[0].parent_path() / "expected-output.txt");
  golden << std::setprecision(9); for (float value : baseline.output) golden << value << '\n';
  std::cout << f.paths[0].parent_path().filename().string()
            << ": IDs=[1,2], logits/weights/gate/up/SwiGLU/down/shared/output match resident ggml; "
            << "cold_io=" << after_cold.io_bytes << " warm_extra_io=" << after_hot.io_bytes - after_cold.io_bytes
            << " warm_hits=" << after_hot.hits - after_cold.hits << " eviction_peak=" << second.peak_bytes
            << " capacity=" << capacity << " reload_io=" << second.io_bytes - first.io_bytes
            << " max_abs_error=" << metrics.max_absolute_error
            << " max_tolerance_fraction=" << metrics.max_tolerance_fraction
            << " values_compared=" << metrics.scalar_values_compared << '\n';
}
} // namespace

int main(int argc, char** argv) {
  fs::path directory;
  const bool preserve = argc == 2;
  try {
    check(argc <= 2, "usage: pmi_connector_test [preserved-fixture-directory]");
    if (preserve) directory = fs::absolute(argv[1]);
    else directory = fs::temp_directory_path() / ("pmi-qwen-moe-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ggml_cpu_init();
    for (const auto type : {GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL}) {
      auto fixture = make_fixture(directory / ggml_type_name(type), type);
      exercise(fixture);
    }
    auto mixed = make_fixture(directory / "mixed-iq3_s-iq4_nl-q8_0", GGML_TYPE_IQ3_S, true);
    exercise(mixed);
    if (preserve) std::cout << "Fixtures preserved at " << directory << '\n';
    else fs::remove_all(directory);
    std::cout << "Qwen4exp single-token MoE boundary verified; this is not full-model generation.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "connector test failed: " << error.what() << '\n';
    if (!preserve && !directory.empty()) fs::remove_all(directory);
    return 1;
  }
}
