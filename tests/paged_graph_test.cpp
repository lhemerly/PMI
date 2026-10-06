#include "pmi/paged_graph.hpp"
#include "pmi/gguf.hpp"
#include "pmi/ggml_device.hpp"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
namespace fs = std::filesystem;
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
void rejects(const std::function<void()>& action, const std::string& message) {
  try { action(); } catch (const std::exception&) { return; }
  throw std::runtime_error("expected rejection: " + message);
}
double maximum_error = 0;
uint64_t compared_values = 0;
void compare(const std::vector<float>& actual, const std::vector<float>& expected,
             const std::string& message) {
  require(actual.size() == expected.size(), message + ": result size");
  for (size_t i = 0; i < actual.size(); ++i) {
    const double error = std::abs(static_cast<double>(actual[i]) - expected[i]);
    maximum_error = std::max(maximum_error, error);
    ++compared_values;
    require(std::isfinite(actual[i]) && std::isfinite(expected[i]) &&
            error <= 3e-6 + 3e-5 * std::abs(expected[i]),
            message + ": element " + std::to_string(i) + " differs: " +
            std::to_string(actual[i]) + " vs " + std::to_string(expected[i]));
  }
}

struct Directory {
  fs::path path = fs::temp_directory_path() /
      ("pmi-paged-graph-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  Directory() { fs::create_directories(path); }
  ~Directory() { std::error_code ignored; fs::remove_all(path, ignored); }
};

template<class T> void write(std::ostream& out, T value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
void string(std::ostream& out, const std::string& value) {
  write<uint64_t>(out, value.size()); out.write(value.data(), static_cast<std::streamsize>(value.size()));
}
template<class T> void metadata(std::ostream& out, const std::string& key, pmi::GgufType type, T value) {
  string(out, key); write<uint32_t>(out, static_cast<uint32_t>(type)); write<T>(out, value);
}
uint64_t align64(uint64_t value) { return (value + 63) & ~uint64_t{63}; }
uint64_t align4096(uint64_t value) { return (value + 4095) & ~uint64_t{4095}; }

struct Weight {
  std::string name;
  std::vector<int64_t> shape;
  ggml_type type{};
  std::vector<uint8_t> bytes;
  uint32_t shard{};
  uint64_t offset{};
};
struct Fixture {
  int64_t columns{}, rows{}, experts{7}, lookup_width{}, lookup_rows{17};
  std::vector<Weight> weights;
  std::vector<fs::path> paths;
  pmi::ModelManifest manifest;
  uint64_t expert_page_bytes{}, lookup_page_bytes{}, total_bytes{};
  const Weight& gate() const { return weights.at(0); }
  const Weight& lookup() const { return weights.at(3); }
};

Fixture make_fixture(const fs::path& directory, ggml_type type) {
  fs::create_directories(directory);
  Fixture f;
  f.columns = std::max<int64_t>(32, ggml_blck_size(type));
  f.rows = std::max<int64_t>(64, ggml_blck_size(type));
  // The verified target's Q4_0 PLE row is 160 elements / 90 payload bytes.
  f.lookup_width = type == GGML_TYPE_Q4_0 ? 160 : f.columns;
  auto add = [&](const std::string& name, const std::vector<int64_t>& shape, uint32_t shard) {
    Weight w; w.name = name; w.shape = shape; w.shard = shard; w.type = type;
    const auto count = std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>{});
    std::vector<float> values(static_cast<size_t>(count));
    const float phase = static_cast<float>(f.weights.size() + 1) * 0.43f;
    for (int64_t i = 0; i < count; ++i) {
      values[static_cast<size_t>(i)] = (std::sin(static_cast<float>(i % 997) * 0.41f + phase) +
                                       0.3f * std::cos(static_cast<float>(i % 199) * 0.23f - phase)) /
                                      std::sqrt(static_cast<float>(shape[0]));
    }
    w.bytes.resize(static_cast<size_t>(ggml_row_size(type, shape[0]) * (count / shape[0])));
    if (type == GGML_TYPE_F32) std::memcpy(w.bytes.data(), values.data(), w.bytes.size());
    else require(ggml_quantize_chunk(type, values.data(), w.bytes.data(), 0, count / shape[0], shape[0], nullptr) == w.bytes.size(),
                 "fixture quantization byte count");
    f.total_bytes += w.bytes.size();
    f.weights.push_back(std::move(w));
  };
  add("blk.0.ffn_gate_exps.weight", {f.columns, f.rows, f.experts}, 0);
  add("blk.0.ffn_up_exps.weight", {f.columns, f.rows, f.experts}, 1);
  add("blk.0.ffn_down_exps.weight", {f.rows, f.columns, f.experts}, 1);
  add("per_layer_token_embd.weight", {f.lookup_width, f.lookup_rows}, 1);
  f.expert_page_bytes = f.gate().bytes.size() / static_cast<uint64_t>(f.experts);
  f.lookup_page_bytes = f.lookup().bytes.size() / static_cast<uint64_t>(f.lookup_rows);
  f.paths = {directory / "ops-00001-of-00002.gguf", directory / "ops-00002-of-00002.gguf"};
  for (uint32_t shard = 0; shard < 2; ++shard) {
    uint64_t position = 0, count = 0;
    for (auto& weight : f.weights) if (weight.shard == shard) {
      weight.offset = position; position = align4096(position + weight.bytes.size()); ++count;
    }
    std::ofstream out(f.paths[shard], std::ios::binary);
    write<uint32_t>(out, 0x46554747); write<uint32_t>(out, 3);
    write<uint64_t>(out, count); write<uint64_t>(out, shard == 0 ? 6 : 4);
    metadata(out, "general.alignment", pmi::GgufType::UINT32, uint32_t{4096});
    metadata(out, "split.no", pmi::GgufType::UINT16, static_cast<uint16_t>(shard));
    metadata(out, "split.count", pmi::GgufType::UINT16, uint16_t{2});
    metadata(out, "split.tensors.count", pmi::GgufType::INT32, int32_t{4});
    if (shard == 0) {
      string(out, "general.architecture"); write<uint32_t>(out, 8); string(out, "qwen4exp");
      metadata(out, "qwen4exp.expert_count", pmi::GgufType::UINT32, static_cast<uint32_t>(f.experts));
    }
    for (const auto& weight : f.weights) if (weight.shard == shard) {
      string(out, weight.name); write<uint32_t>(out, static_cast<uint32_t>(weight.shape.size()));
      for (auto dimension : weight.shape) write<uint64_t>(out, static_cast<uint64_t>(dimension));
      write<uint32_t>(out, static_cast<uint32_t>(weight.type)); write<uint64_t>(out, weight.offset);
    }
    require(static_cast<uint64_t>(out.tellp()) < 4096, "fixture header size");
    for (const auto& weight : f.weights) if (weight.shard == shard) {
      out.seekp(static_cast<std::streamoff>(4096 + weight.offset));
      out.write(reinterpret_cast<const char*>(weight.bytes.data()), static_cast<std::streamsize>(weight.bytes.size()));
    }
    require(static_cast<bool>(out), "write graph fixture");
  }
  f.manifest = pmi::build_manifest({pmi::GgufReader(f.paths[1]).inspect(), pmi::GgufReader(f.paths[0]).inspect()});
  return f;
}

class Arena {
 public:
  explicit Arena(bool descriptors_only = false) {
    ggml_init_params params{};
    params.mem_size = descriptors_only ? 65536 : 2 * 1024 * 1024;
    params.no_alloc = descriptors_only;
    ctx_ = ggml_init(params); require(ctx_ != nullptr, "ggml test arena allocation");
  }
  ~Arena() { ggml_free(ctx_); }
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;
  ggml_context* ctx() { return ctx_; }
  ggml_tensor* weight(const Weight& weight, bool resident) {
    auto* tensor = ggml_new_tensor(ctx_, weight.type, static_cast<int>(weight.shape.size()), weight.shape.data());
    ggml_set_name(tensor, weight.name.c_str());
    if (resident) std::memcpy(tensor->data, weight.bytes.data(), weight.bytes.size());
    else require(tensor->data == nullptr, "deferred descriptor unexpectedly owns payload");
    return tensor;
  }
  ggml_cgraph* graph(ggml_tensor* output) {
    auto* graph = ggml_new_graph_custom(ctx_, 128, false);
    ggml_build_forward_expand(graph, output); return graph;
  }
 private:
  ggml_context* ctx_{};
};

ggml_status compute(ggml_cgraph* graph, int threads = 1, pmi::PagedQwenGraph* pager = nullptr) {
  auto plan = ggml_graph_plan(graph, threads, nullptr);
  std::vector<uint8_t> workspace(plan.work_size);
  plan.work_data = workspace.data();
  if (pager) { plan.abort_callback = pmi::PagedQwenGraph::abort; plan.abort_callback_data = pager; }
  return ggml_graph_compute(graph, &plan);
}
std::vector<float> values(const ggml_tensor* tensor) {
  require(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor), "expected contiguous float graph output");
  const auto* data = static_cast<const float*>(tensor->data);
  return {data, data + ggml_nelements(tensor)};
}
void excludes_weight(ggml_tensor* root, const ggml_tensor* weight) {
  std::unordered_set<ggml_tensor*> seen;
  std::function<void(ggml_tensor*)> inspect = [&](ggml_tensor* node) {
    if (!node || !seen.insert(node).second) return;
    require(node != weight && std::string(node->name) != weight->name, "deferred weight leaked into scheduler graph");
    for (auto* src : node->src) inspect(src);
    inspect(node->view_src);
  };
  inspect(root);
}

constexpr int64_t slots = 4, tokens = 3;
const std::vector<int32_t> expert_ids{6, 1, 6, 0, 2, 6, 1, 2, 0, 4, 6, 1};
const std::vector<int32_t> row_ids{16, 0, 7, 16, 1, 8};

struct ExpertInputs { ggml_tensor* input{}; ggml_tensor* ids{}; };
void set_ids(ggml_tensor* ids, const std::vector<int32_t>& values) {
  require(static_cast<size_t>(ggml_nelements(ids)) == values.size(), "index shape");
  for (int64_t i1 = 0; i1 < ids->ne[1]; ++i1) for (int64_t i0 = 0; i0 < ids->ne[0]; ++i0) {
    const auto value = values[static_cast<size_t>(i1 * ids->ne[0] + i0)];
    std::memcpy(static_cast<uint8_t*>(ids->data) + i0 * ids->nb[0] + i1 * ids->nb[1], &value, sizeof(value));
  }
}
ExpertInputs expert_inputs(Arena& arena, int64_t columns, int64_t input_slots, bool strided) {
  auto* ctx = arena.ctx();
  const size_t nb1 = static_cast<size_t>(columns + (strided ? 8 : 0)) * sizeof(float);
  const size_t nb2 = static_cast<size_t>(input_slots + (strided ? 1 : 0)) * nb1;
  const size_t offset = strided ? 12 * sizeof(float) : 0;
  auto* backing = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, static_cast<int64_t>((nb2 * tokens + offset) / sizeof(float)));
  std::fill_n(static_cast<float*>(backing->data), ggml_nelements(backing), std::numeric_limits<float>::quiet_NaN());
  auto* input = ggml_view_3d(ctx, backing, columns, input_slots, tokens, nb1, nb2, offset);
  for (int64_t token = 0; token < tokens; ++token) for (int64_t slot = 0; slot < input_slots; ++slot) {
    auto* row = reinterpret_cast<float*>(static_cast<uint8_t*>(input->data) + slot * input->nb[1] + token * input->nb[2]);
    for (int64_t column = 0; column < columns; ++column) {
      row[column] = std::cos(static_cast<float>(column) * 0.19f + static_cast<float>(token) * 0.7f) +
                    0.41f * static_cast<float>(slot) - 0.17f * static_cast<float>(token);
    }
  }
  ggml_tensor* ids = nullptr;
  if (strided) {
    // Transpose a padded view, giving nb[0] > sizeof(int32_t), not merely a
    // noncontiguous token stride. Sentinel padding must never be read as IDs.
    auto* storage = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (tokens + 3) * slots + 4);
    std::fill_n(static_cast<int32_t*>(storage->data), ggml_nelements(storage), int32_t{-999});
    auto* view = ggml_view_2d(ctx, storage, tokens, slots, static_cast<size_t>(tokens + 3) * sizeof(int32_t), 4 * sizeof(int32_t));
    ids = ggml_transpose(ctx, view);
  } else ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, slots, tokens);
  set_ids(ids, expert_ids);
  return {input, ids};
}
ggml_tensor* lookup_indices(Arena& arena, bool strided) {
  ggml_tensor* ids = nullptr;
  if (strided) {
    auto* backing = ggml_new_tensor_1d(arena.ctx(), GGML_TYPE_I32, static_cast<int64_t>(row_ids.size() * 3 + 2));
    std::fill_n(static_cast<int32_t*>(backing->data), ggml_nelements(backing), int32_t{-999});
    auto* view = ggml_view_2d(arena.ctx(), backing, 1, static_cast<int64_t>(row_ids.size()), 3 * sizeof(int32_t), 2 * sizeof(int32_t));
    ids = ggml_transpose(arena.ctx(), view);
  } else ids = ggml_new_tensor_1d(arena.ctx(), GGML_TYPE_I32, static_cast<int64_t>(row_ids.size()));
  set_ids(ids, row_ids); return ids;
}

std::shared_ptr<pmi::GgmlDeviceProjection> cpu_device_executor(
    pmi::FilePageProvider& provider, const Fixture& f, bool tiny_cache,
    uint64_t graph_capacity = 1024 * 1024, uint64_t cache_capacity_override = 0) {
  pmi::GgmlBackendOwner owner(ggml_backend_cpu_init(), ggml_backend_free);
  require(owner != nullptr, "CPU device executor backend");
  ggml_backend_cpu_set_n_threads(owner.get(), 1);
  auto adapter = std::make_shared<pmi::GgmlDeviceBackend>(std::move(owner));
  pmi::DevicePageRequest request;
  request.source = *std::find_if(f.manifest.tensors.begin(), f.manifest.tensors.end(),
      [](const auto& item) { return item.source_tensor_name == "blk.0.ffn_gate_exps.weight" && item.expert_id == 0; });
  request.format = pmi::GgmlDeviceBackend::format;
  request.layout = pmi::GgmlDeviceBackend::layout;
  const auto charge = adapter->allocation_bytes(request);
  const auto cache_capacity = cache_capacity_override ? cache_capacity_override :
      charge * (tiny_cache ? 1 : static_cast<uint64_t>(f.experts) * 3);
  return std::make_shared<pmi::GgmlDeviceProjection>(provider, std::move(adapter),
      cache_capacity, graph_capacity, true);
}

void experts_case(const Fixture& f, size_t weight_index, int64_t input_slots, bool strided,
                  bool tiny_cache, bool device_projection = false) {
  const auto& weight = f.weights.at(weight_index);
  const auto columns = weight.shape[0], rows = weight.shape[1];
  const auto capacity = tiny_cache || device_projection ? align64(f.expert_page_bytes) :
      f.total_bytes + f.manifest.tensors.size() * 64;
  pmi::FilePageProvider provider(f.paths, static_cast<size_t>(capacity));
  pmi::PagedGraphExecutionOptions options;
  if (device_projection) {
    options.projection_executor = cpu_device_executor(provider, f, tiny_cache);
    options.profile = true;
  }
  pmi::PagedQwenGraph pager(f.manifest, provider, {true, 1024, 1024}, options);
  Arena descriptors(true), paged, resident;
  auto* deferred = descriptors.weight(weight, false);
  require(pmi::PagedQwenGraph::select(deferred, &pager), "expert descriptor was not selected");
  pager.throw_if_failed();
  auto args = expert_inputs(paged, columns, input_slots, strided);
  auto reference_args = expert_inputs(resident, columns, input_slots, strided);
  auto* output = pmi::PagedQwenGraph::build(paged.ctx(), GGML_OP_MUL_MAT_ID, deferred, args.input, args.ids, &pager);
  pager.throw_if_failed(); require(output != nullptr, "expert custom node is null");
  excludes_weight(output, deferred);
  require(output->ne[0] == rows && output->ne[1] == slots && output->ne[2] == tokens, "expert output shape");
  auto* expected = ggml_mul_mat_id(resident.ctx(), resident.weight(weight, true), reference_args.input, reference_args.ids);
  auto* graph = paged.graph(output); auto* reference_graph = resident.graph(expected);
  require(compute(reference_graph, 2) == GGML_STATUS_SUCCESS, "resident expert graph");
  const auto expected_values = values(expected);
  require(compute(graph, 2, &pager) == GGML_STATUS_SUCCESS, "paged expert graph");
  pager.throw_if_failed(); compare(values(output), expected_values, "multi-token expert outputs");
  const auto cold = provider.stats();
  require(cold.active_leases == 0 && cold.peak_bytes <= capacity, "expert pages exceeded payload capacity or leaked leases");
  if (!tiny_cache) require(cold.io_bytes == 5 * f.expert_page_bytes, "read payload other than the five selected experts");
  require(deferred->data == nullptr, "deferred expert descriptor acquired a full payload");
  const auto events = pager.trace();
  require(events.size() == static_cast<size_t>(slots * tokens), "one trace event per token/expert slot");
  std::unordered_set<int64_t> seen;
  for (const auto& event : events) {
    require(event.tensor_name == weight.name && event.token >= 0 && event.token < tokens &&
            event.slot >= 0 && event.slot < slots, "expert trace coordinates");
    const auto index = event.token * slots + event.slot;
    require(seen.insert(index).second && event.id == expert_ids[static_cast<size_t>(index)], "expert trace global ID");
    compare(event.output, {expected_values.begin() + index * rows, expected_values.begin() + (index + 1) * rows}, "expert trace values");
  }
  require(compute(graph, 1, &pager) == GGML_STATUS_SUCCESS, "repeat paged expert graph");
  pager.throw_if_failed(); compare(values(output), expected_values, "repeat expert outputs");
  const auto warm = provider.stats();
  if (tiny_cache) require(warm.evictions > 0 && warm.io_bytes > cold.io_bytes, "forced expert eviction did not reload");
  else if (device_projection) {
    const auto stats = pager.stats().device_projection;
    require(warm.io_bytes == cold.io_bytes && warm.hits == cold.hits &&
            stats.cache.provider_requests == 5 && stats.cache.hits == 2 * slots * tokens - 5,
            "warm device hits touched host provider");
  } else require(warm.io_bytes == cold.io_bytes && warm.hits > cold.hits, "warm experts performed extra reads");
  // Graph reuse must read current IDs, rather than a routing decision captured
  // while constructing the node. Previously unused expert 5 is included here.
  auto changed = expert_ids; changed[0] = 5; changed[5] = 3;
  set_ids(args.ids, changed); set_ids(reference_args.ids, changed);
  for (auto* input : {args.input, reference_args.input}) {
    auto* row = reinterpret_cast<float*>(static_cast<uint8_t*>(input->data) +
                 (input_slots - 1) * input->nb[1] + (tokens - 1) * input->nb[2]);
    for (int64_t column = 0; column < columns; ++column) row[column] = row[column] * 1.5f + 0.25f;
  }
  require(compute(reference_graph, 1) == GGML_STATUS_SUCCESS && compute(graph, 2, &pager) == GGML_STATUS_SUCCESS,
          "expert graph routing reuse");
  pager.throw_if_failed(); compare(values(output), values(expected), "changed global expert IDs");
  require(provider.stats().active_leases == 0 && provider.stats().peak_bytes <= capacity,
          "reused expert graph escaped lease accounting");
  require(pager.stats().expert_requests > 0 && pager.stats().page_requests > 0 && pager.stats().executed_nodes >= 3,
          "expert operation telemetry");
  require(!pager.trace().empty(), "expert output trace missing");
  const auto stats = pager.stats();
  require(stats.device_projection_enabled == device_projection, "device projection enable telemetry");
  if (device_projection) {
    require(stats.page_requests == 3 * slots * tokens && stats.expert_requests == stats.page_requests &&
            stats.device_projection.calls == stats.expert_requests &&
            stats.device_projection.completed == stats.device_projection.calls,
            "logical graph and executor projection counters differ");
    require(stats.profile_provider_requests == 0 && stats.profile_expert_requests == 0 &&
            stats.profile_projection_calls == stats.expert_requests,
            "device mode reported direct graph provider requests");
    require(stats.device_projection.cache.provider_requests == provider.stats().hits + provider.stats().misses &&
            stats.device_projection.cache.active_leases == 0 &&
            stats.device_projection.cache.peak_bytes <= stats.device_projection.cache.capacity_bytes &&
            stats.device_projection.graph_current_bytes == 0 &&
            stats.device_projection.graph_peak_bytes <= stats.device_projection.graph_capacity_bytes,
            "device graph escaped cache, graph-buffer, or host-request accounting");
    if (tiny_cache) require(stats.device_projection.cache.evictions > 0 &&
                            stats.device_projection.cache.uploads > static_cast<uint64_t>(f.experts),
                            "tiny device cache failed to evict and upload again");
  }
}

// A failing resident branch must finish before the connector clears its cache.
class SplitFault final : public pmi::ProjectionExecutor {
 public:
  explicit SplitFault(std::shared_ptr<pmi::ProjectionExecutor> d) : delegate(std::move(d)) {}
  std::vector<float> project(const pmi::TensorInfo& d, const std::vector<float>& x) override { return delegate->project(d,x); }
  bool supports_resident_split() const noexcept override { return true; }
  pmi::DeviceLease resident(const pmi::TensorInfo& d, bool seed) override { return delegate->resident(d,seed); }
  std::vector<float> project_resident(const pmi::DeviceLease& lease, const std::vector<float>& x) override {
    auto result = delegate->project_resident(lease,x);
    finished = true;
    throw std::runtime_error("injected split worker failure");
  }
  pmi::ProjectionStats stats() const override { return delegate->stats(); }
  void clear() override { cleared_after_finish = finished.load(); ++clears; delegate->clear(); }
  std::atomic<bool> finished{false}; bool cleared_after_finish{}; size_t clears{};
 private: std::shared_ptr<pmi::ProjectionExecutor> delegate;
};

void split_case(const Fixture& f, bool fault = false, bool cpu_error = false) {
  pmi::FilePageProvider provider(f.paths, align64(f.expert_page_bytes));
  auto executor = cpu_device_executor(provider, f, true);
  auto failing = std::make_shared<SplitFault>(executor);
  pmi::PagedGraphExecutionOptions options;
  options.projection_executor = fault ? std::static_pointer_cast<pmi::ProjectionExecutor>(failing) : executor;
  options.profile = true; options.split_cpu_misses = true;
  pmi::PagedQwenGraph pager(f.manifest, provider, {true, 1024, 1024}, options);
  Arena descriptors(true), paged, reference;
  const auto& w = f.gate(); auto* deferred = descriptors.weight(w,false);
  require(pmi::PagedQwenGraph::select(deferred,&pager), "split selection");
  auto args = expert_inputs(paged,f.columns,2,true);
  auto expected_args = expert_inputs(reference,f.columns,2,true);
  auto* out = pmi::PagedQwenGraph::build(paged.ctx(),GGML_OP_MUL_MAT_ID,deferred,args.input,args.ids,&pager);
  pager.throw_if_failed(); require(out != nullptr,"split graph node");
  auto* graph = paged.graph(out);
  if (cpu_error) {
    const auto& page = *std::find_if(f.manifest.tensors.begin(),f.manifest.tensors.end(),
      [](const auto& t){return t.source_tensor_name == "blk.0.ffn_gate_exps.weight" && t.expert_id == 6;});
    { auto lease = executor->resident(page,true); require(bool(lease),"CPU error resident setup"); }
    pager.finish_device_seed();
    fs::resize_file(f.paths.at(page.shard_id),0);
  }
  if (fault) {
    require(compute(graph,2,&pager) == GGML_STATUS_ABORTED,"split failure did not abort");
    rejects([&]{pager.throw_if_failed();},"split failure was not latched");
    require(failing->clears == 1 && failing->cleared_after_finish,"cache cleared before worker finished");
    require(executor->stats().cache.active_leases == 0 && executor->stats().cache.current_bytes == 0 &&
            provider.stats().active_leases == 0,"split error leaked leases/storage");
    return;
  }
  auto* expected = ggml_mul_mat_id(reference.ctx(),reference.weight(w,true),expected_args.input,expected_args.ids);
  require(compute(reference.graph(expected),2) == GGML_STATUS_SUCCESS,"split resident reference");
  require(compute(graph,2,&pager) == GGML_STATUS_SUCCESS,"split first evaluation"); pager.throw_if_failed();
  compare(values(out),values(expected),"split mixed strided rows");
  pager.finish_device_seed();
  const auto seeded = executor->stats().cache;
  require(seeded.uploads == 1 && seeded.evictions == 0,"split seed did not remain bounded/frozen");
  for (unsigned repeat=0;repeat<3;++repeat) {
    require(compute(graph,2,&pager) == GGML_STATUS_SUCCESS,"split reused graph"); pager.throw_if_failed();
    compare(values(out),values(expected),"split repeated rows");
  }
  const auto stats = pager.stats(); const auto cache = executor->stats().cache;
  require(stats.split_cpu_projections > 0 && stats.split_device_projections > 0 &&
          stats.split_cpu_projections + stats.split_device_projections == 4 * slots * tokens,"split dispatch accounting");
  require(cache.uploads == seeded.uploads && cache.provider_requests == seeded.provider_requests &&
          cache.evictions == 0,"frozen misses uploaded or evicted weights");
  require(cache.active_leases == 0 && provider.stats().active_leases == 0 && provider.stats().evictions > 0,
          "split leases/CPU eviction accounting");
  require(stats.split_critical_ms >= stats.split_overlap_ms && stats.split_cpu_ms >= stats.split_overlap_ms &&
          stats.split_device_ms >= stats.split_overlap_ms,"split timing interval accounting");
}

void profile_stats(const pmi::PagedGraphStats&, bool enabled);

void lookup_case(const Fixture& f, bool strided, bool tiny_cache, bool profiling = false) {
  const auto capacity = tiny_cache ? align64(f.lookup_page_bytes) : f.total_bytes + f.manifest.tensors.size() * 64;
  pmi::FilePageProvider provider(f.paths, static_cast<size_t>(capacity));
  pmi::PagedGraphExecutionOptions options; options.profile = profiling;
  pmi::PagedQwenGraph pager(f.manifest, provider, {true, 1024, 1024}, options);
  Arena descriptors(true), paged, resident;
  auto* deferred = descriptors.weight(f.lookup(), false);
  require(pmi::PagedQwenGraph::select(deferred, &pager), "PLE descriptor was not selected");
  pager.throw_if_failed();
  auto* ids = lookup_indices(paged, strided); auto* expected_ids = lookup_indices(resident, strided);
  auto* output = pmi::PagedQwenGraph::build(paged.ctx(), GGML_OP_GET_ROWS, deferred, ids, nullptr, &pager);
  pager.throw_if_failed(); require(output != nullptr, "PLE custom node is null");
  excludes_weight(output, deferred);
  auto* expected = ggml_get_rows(resident.ctx(), resident.weight(f.lookup(), true), expected_ids);
  auto* graph = paged.graph(output); auto* reference_graph = resident.graph(expected);
  require(compute(reference_graph, 2) == GGML_STATUS_SUCCESS, "resident PLE graph");
  require(compute(graph, 2, &pager) == GGML_STATUS_SUCCESS, "paged PLE graph");
  pager.throw_if_failed(); compare(values(output), values(expected), "global PLE lookup rows");
  const auto expected_values = values(expected);
  const auto events = pager.trace();
  require(events.size() == row_ids.size(), "one PLE trace event per requested row");
  std::unordered_set<int64_t> seen;
  for (const auto& event : events) {
    require(event.tensor_name == f.lookup().name && event.slot == 0 && event.token >= 0 &&
            static_cast<size_t>(event.token) < row_ids.size(), "PLE trace coordinates");
    require(seen.insert(event.token).second && event.id == row_ids[static_cast<size_t>(event.token)], "PLE trace global row ID");
    compare(event.output, {expected_values.begin() + event.token * f.lookup_width,
                          expected_values.begin() + (event.token + 1) * f.lookup_width}, "PLE trace values");
  }
  const auto cold = provider.stats();
  require(cold.active_leases == 0 && cold.peak_bytes <= capacity, "PLE row pages exceeded payload capacity or leaked leases");
  require(cold.io_bytes < f.lookup().bytes.size(), "entire sparse lookup table was eagerly read");
  require(deferred->data == nullptr, "deferred PLE descriptor acquired a full payload");
  require(compute(graph, 1, &pager) == GGML_STATUS_SUCCESS, "repeat PLE graph");
  pager.throw_if_failed(); compare(values(output), values(expected), "repeat global PLE lookup rows");
  const auto warm = provider.stats();
  if (tiny_cache) require(warm.evictions > 0 && warm.io_bytes > cold.io_bytes, "forced PLE row eviction did not reload");
  else require(warm.io_bytes == cold.io_bytes && warm.hits > cold.hits, "warm PLE rows performed extra reads");
  auto changed = row_ids; changed[0] = 15; changed[5] = 2;
  set_ids(ids, changed); set_ids(expected_ids, changed);
  require(compute(reference_graph, 1) == GGML_STATUS_SUCCESS && compute(graph, 2, &pager) == GGML_STATUS_SUCCESS,
          "PLE graph index reuse");
  pager.throw_if_failed(); compare(values(output), values(expected), "changed global PLE indices");
  require(provider.stats().active_leases == 0 && provider.stats().peak_bytes <= capacity,
          "reused PLE graph escaped lease accounting");
  require(pager.stats().lookup_requests > 0 && !pager.trace().empty(), "PLE telemetry/trace missing");
  profile_stats(pager.stats(), profiling);
  pager.clear_trace(); require(pager.trace().empty(), "trace clearing");
}

void failure_case(const Fixture& f, bool lookup, int32_t invalid_id, bool truncate = false,
                  bool capacity_error = false, bool partial_progress = false) {
  const auto page_bytes = lookup ? f.lookup_page_bytes : f.expert_page_bytes;
  auto paths = f.paths;
  fs::path broken;
  if (truncate) {
    const auto shard = lookup ? f.lookup().shard : f.gate().shard;
    broken = paths[shard].parent_path() / (lookup ? "broken-lookup.gguf" : "broken-experts.gguf");
    fs::copy_file(paths[shard], broken, fs::copy_options::overwrite_existing);
    paths[shard] = broken;
  }
  pmi::FilePageProvider provider(paths, capacity_error ? 1 : static_cast<size_t>(align64(page_bytes)));
  pmi::PagedQwenGraph pager(f.manifest, provider);
  Arena descriptors(true), arena;
  auto* deferred = descriptors.weight(lookup ? f.lookup() : f.gate(), false);
  ggml_tensor* output = nullptr;
  if (lookup) {
    auto* ids = lookup_indices(arena, true);
    if (!truncate && !capacity_error) { auto bad_ids = row_ids; bad_ids[0] = invalid_id; set_ids(ids, bad_ids); }
    if (partial_progress) { auto changed = row_ids; changed[0] = 0; changed[1] = 16; set_ids(ids, changed); }
    output = pmi::PagedQwenGraph::build(arena.ctx(), GGML_OP_GET_ROWS, deferred, ids, nullptr, &pager);
  } else {
    auto args = expert_inputs(arena, f.columns, 2, true);
    if (!truncate && !capacity_error) { auto bad_ids = expert_ids; bad_ids[0] = invalid_id; set_ids(args.ids, bad_ids); }
    if (partial_progress) { auto changed = expert_ids; changed[0] = 0; set_ids(args.ids, changed); }
    output = pmi::PagedQwenGraph::build(arena.ctx(), GGML_OP_MUL_MAT_ID, deferred, args.input, args.ids, &pager);
  }
  pager.throw_if_failed(); require(output != nullptr, "failure-test custom node is null");
  if (truncate) fs::resize_file(broken, partial_progress ?
      4096 + (lookup ? f.lookup().offset : f.gate().offset) + page_bytes : 4097); // after opening shard handles
  // A following graph node makes it observable that the latched callback error
  // aborts the decoder before a consumer can use the failed operator output.
  auto* consumer = ggml_scale(arena.ctx(), output, 2.0f);
  constexpr float sentinel = -98765.25f;
  std::fill_n(static_cast<float*>(consumer->data), ggml_nelements(consumer), sentinel);
  auto* graph = arena.graph(consumer);
  require(compute(graph, 2, &pager) == GGML_STATUS_ABORTED, "failed page operation did not abort the enclosing graph");
  for (float value : values(consumer)) require(value == sentinel, "a consumer executed after the custom operation failed");
  require(pmi::PagedQwenGraph::abort(&pager), "failed operator error was not latched");
  rejects([&] { pager.throw_if_failed(); }, "failed operator error propagation");
  require(provider.stats().active_leases == 0, "failed custom operation leaked a page lease");
  if (partial_progress) require(provider.stats().io_bytes >= page_bytes && provider.stats().failures > 0,
                                "partial-progress failure did not read an earlier valid page");
  const auto failed_io = provider.stats().io_bytes;
  require(compute(graph, 1, &pager) == GGML_STATUS_ABORTED, "latched failure allowed graph reuse");
  require(provider.stats().io_bytes == failed_io, "latched failed graph performed additional I/O");
  if (!broken.empty()) fs::remove(broken);
}

// Faults occur after successful compact projections populated a real CPU
// device cache. This checks the connector's failure boundary independently of
// backend transfer-failure injection in the adapter/cache unit tests.
class FaultProjection final : public pmi::ProjectionExecutor {
 public:
  FaultProjection(std::shared_ptr<pmi::ProjectionExecutor> delegate, bool wrong_size)
      : delegate_(std::move(delegate)), wrong_size_(wrong_size) {}
  std::vector<float> project(const pmi::TensorInfo& descriptor,
                             const std::vector<float>& input) override {
    ++calls;
    auto result = delegate_->project(descriptor, input);
    if (calls == 3) {
      if (wrong_size_) result.pop_back();
      else throw std::runtime_error("injected late projection failure");
    }
    return result;
  }
  pmi::ProjectionStats stats() const override { return delegate_->stats(); }
  void clear() override { ++clears; delegate_->clear(); }
  size_t calls{}, clears{};
 private:
  std::shared_ptr<pmi::ProjectionExecutor> delegate_;
  bool wrong_size_{};
};

void device_projection_failure(const Fixture& f, bool wrong_size) {
  pmi::FilePageProvider provider(f.paths, static_cast<size_t>(align64(f.expert_page_bytes)));
  auto executor = std::make_shared<FaultProjection>(cpu_device_executor(provider, f, false), wrong_size);
  {
    pmi::PagedGraphExecutionOptions options;
    options.projection_executor = executor;
    pmi::PagedQwenGraph pager(f.manifest, provider, {}, options);
    Arena descriptors(true), arena;
    auto* deferred = descriptors.weight(f.gate(), false);
    auto args = expert_inputs(arena, f.columns, 2, true);
    auto* output = pmi::PagedQwenGraph::build(arena.ctx(), GGML_OP_MUL_MAT_ID,
                                              deferred, args.input, args.ids, &pager);
    pager.throw_if_failed(); require(output != nullptr, "device failure custom node");
    auto* consumer = ggml_scale(arena.ctx(), output, 2.0f);
    constexpr float sentinel = -98765.25f;
    std::fill_n(static_cast<float*>(consumer->data), ggml_nelements(consumer), sentinel);
    auto* graph = arena.graph(consumer);
    require(compute(graph, 2, &pager) == GGML_STATUS_ABORTED,
            "late executor failure did not abort graph");
    require(executor->calls == 3 && executor->clears == 1,
            "late executor failure did not clear once after partial work");
    for (float value : values(output)) require(value == 0, "partial executor output was not zeroed");
    for (float value : values(consumer)) require(value == sentinel, "failed projection consumer executed");
    rejects([&] { pager.throw_if_failed(); }, "executor failure was not propagated");
    const auto stats = pager.stats();
    require(stats.expert_requests == 3 && stats.device_projection.cache.current_bytes == 0 &&
            stats.device_projection.cache.active_leases == 0 &&
            stats.device_projection.graph_current_bytes == 0 && provider.stats().active_leases == 0,
            "failed device graph retained cache storage or consumer leases");
    const auto io = provider.stats().io_bytes;
    require(compute(graph, 1, &pager) == GGML_STATUS_ABORTED && executor->calls == 3 &&
            provider.stats().io_bytes == io, "latched failure allowed another executor or storage call");
  }
  require(executor->clears == 2 && executor->stats().cache.current_bytes == 0,
          "graph destruction did not safely clear its dedicated executor");
}

void device_projection_configuration(const Fixture& f) {
  pmi::FilePageProvider provider(f.paths, static_cast<size_t>(align64(f.expert_page_bytes)));
  auto executor = cpu_device_executor(provider, f, false);
  pmi::PagedGraphExecutionOptions options;
  options.projection_executor = executor;
  options.retain_experts_per_layer = 4;
  rejects([&] { pmi::PagedQwenGraph pager(f.manifest, provider, {}, options); },
          "host previous-top4 retention with device executor");
  require(provider.stats().io_bytes == 0 && executor->stats().calls == 0,
          "incompatible policy performed work before rejection");
  options.retain_experts_per_layer = 0;
  {
    pmi::PagedQwenGraph pager(f.manifest, provider, {}, options);
    Arena descriptors(true), arena, resident;
    auto* deferred = descriptors.weight(f.lookup(), false);
    auto* ids = lookup_indices(arena, true);
    auto* output = pmi::PagedQwenGraph::build(arena.ctx(), GGML_OP_GET_ROWS, deferred, ids, nullptr, &pager);
    pager.throw_if_failed(); require(output != nullptr, "device-configured PLE node");
    auto* expected = ggml_get_rows(resident.ctx(), resident.weight(f.lookup(), true), lookup_indices(resident, true));
    require(compute(resident.graph(expected), 1) == GGML_STATUS_SUCCESS &&
            compute(arena.graph(output), 2, &pager) == GGML_STATUS_SUCCESS,
            "device-configured PLE execution");
    pager.throw_if_failed(); compare(values(output), values(expected), "PLE stays on CPU with executor enabled");
    require(executor->stats().calls == 0 && pager.stats().lookup_requests == row_ids.size() &&
            pager.stats().expert_requests == 0, "PLE lookup called expert projection executor");
  }
}

void device_projection_capacity(const Fixture& f, bool graph_limit) {
  pmi::FilePageProvider provider(f.paths, static_cast<size_t>(align64(f.expert_page_bytes)));
  auto executor = cpu_device_executor(provider, f, true, graph_limit ? 1 : 1024 * 1024,
                                       graph_limit ? 0 : 1);
  pmi::PagedGraphExecutionOptions options;
  options.projection_executor = executor;
  pmi::PagedQwenGraph pager(f.manifest, provider, {}, options);
  Arena descriptors(true), arena;
  auto args = expert_inputs(arena, f.columns, 1, false);
  auto* output = pmi::PagedQwenGraph::build(arena.ctx(), GGML_OP_MUL_MAT_ID,
      descriptors.weight(f.gate(), false), args.input, args.ids, &pager);
  pager.throw_if_failed(); require(output != nullptr, "bounded device projection node");
  auto* consumer = ggml_scale(arena.ctx(), output, 2.0f);
  require(compute(arena.graph(consumer), 2, &pager) == GGML_STATUS_ABORTED,
          "device allocation budget failure did not abort graph");
  rejects([&] { pager.throw_if_failed(); }, "device allocation budget failure propagation");
  for (float value : values(output)) require(value == 0, "budget failure output was not zeroed");
  const auto stats = executor->stats();
  require(provider.stats().io_bytes == 0 && stats.cache.provider_requests == 0 &&
          stats.cache.current_bytes == 0 && stats.graph_current_bytes == 0 &&
          stats.cache.active_leases == 0 && provider.stats().active_leases == 0,
          "rejected device allocation budget read weights or leaked allocations");
}

void tracing_limits(const Fixture& f) {
  pmi::FilePageProvider provider(f.paths, static_cast<size_t>(align64(f.expert_page_bytes)));
  pmi::PagedQwenGraph pager(f.manifest, provider, {true, 2, 3});
  Arena descriptors(true), arena;
  auto* weight = descriptors.weight(f.gate(), false);
  auto args = expert_inputs(arena, f.columns, 2, true);
  auto* output = pmi::PagedQwenGraph::build(arena.ctx(), GGML_OP_MUL_MAT_ID, weight, args.input, args.ids, &pager);
  pager.throw_if_failed(); require(output != nullptr, "trace-limited custom node");
  require(compute(arena.graph(output), 2, &pager) == GGML_STATUS_SUCCESS, "trace-limited custom node execution");
  pager.throw_if_failed();
  const auto events = pager.trace();
  require(events.size() <= 2 && !events.empty() && pager.stats().trace_dropped > 0, "trace event count bound");
  for (const auto& event : events) require(event.output.size() <= 3 && event.output_truncated, "trace output bound");
}

void planning_and_callback_errors(const Fixture& f) {
  pmi::FilePageProvider provider(f.paths, static_cast<size_t>(align64(f.expert_page_bytes)));
  {
    auto shuffled = f.manifest;
    std::reverse(shuffled.tensors.begin(), shuffled.tensors.end());
    pmi::PagedQwenGraph pager(shuffled, provider);
    require(pager.stats().candidate_weight_bytes == f.total_bytes && pager.stats().deferred_weight_bytes == 0,
            "candidate payload plan differs from actual selected deferred bytes");
    require(provider.stats().io_bytes == 0, "planning eagerly read weight payload");
    Arena metadata(true);
    auto* dense = ggml_new_tensor_2d(metadata.ctx(), GGML_TYPE_F32, 32, 32);
    ggml_set_name(dense, "token_embd.weight");
    require(!pmi::PagedQwenGraph::select(dense, &pager), "dense embedding was incorrectly deferred");
    pager.throw_if_failed();
    require(pager.stats().deferred_weight_bytes == 0, "dense selection changed deferred-byte accounting");
    uint64_t selected_bytes = 0;
    for (const auto& weight : f.weights) {
      auto* descriptor = metadata.weight(weight, false);
      require(pmi::PagedQwenGraph::select(descriptor, &pager), "candidate weight selection");
      pager.throw_if_failed(); selected_bytes += weight.bytes.size();
      require(pager.stats().deferred_weight_bytes == selected_bytes, "selected weight payload accounting");
      require(pmi::PagedQwenGraph::select(descriptor, &pager) && pager.stats().deferred_weight_bytes == selected_bytes,
              "repeated selection double-counted deferred bytes");
    }
  }
  {
    auto malformed = f.manifest; malformed.model_architecture = "qwen3next";
    rejects([&] { pmi::PagedQwenGraph pager(malformed, provider); }, "unsupported architecture");
    malformed = f.manifest; malformed.tensors.erase(malformed.tensors.begin());
    rejects([&] { pmi::PagedQwenGraph pager(malformed, provider); }, "missing expert slice");
    malformed = f.manifest; malformed.tensors.push_back(malformed.tensors.front());
    rejects([&] { pmi::PagedQwenGraph pager(malformed, provider); }, "duplicate expert slice");
    malformed = f.manifest;
    auto scale = malformed.tensors.back();
    scale.name = scale.source_tensor_name = "per_layer_token_embd.weight_scale";
    malformed.tensors.push_back(scale);
    rejects([&] { pmi::PagedQwenGraph pager(malformed, provider); }, "unsupported external PLE scale");
  }
  {
    pmi::PagedQwenGraph pager(f.manifest, provider);
    Arena metadata(true);
    auto* malformed = metadata.weight(f.gate(), false);
    ++malformed->ne[1];
    // A known-but-malformed name must remain deferred so the loader cannot
    // silently fall back to loading a large resident payload after the error.
    require(pmi::PagedQwenGraph::select(malformed, &pager), "malformed deferred metadata allowed resident fallback");
    require(pmi::PagedQwenGraph::abort(&pager), "metadata mismatch was not latched");
    rejects([&] { pager.throw_if_failed(); }, "metadata shape mismatch");
  }
  for (int mode = 0; mode < 3; ++mode) {
    pmi::PagedQwenGraph pager(f.manifest, provider);
    Arena metadata(true), arena;
    auto* weight = metadata.weight(f.gate(), false);
    auto args = expert_inputs(arena, f.columns, mode == 1 ? 3 : 2, true);
    if (mode == 2) args.ids = ggml_new_tensor_2d(arena.ctx(), GGML_TYPE_F32, slots, tokens);
    auto* output = pmi::PagedQwenGraph::build(arena.ctx(), mode == 0 ? GGML_OP_MUL_MAT : GGML_OP_MUL_MAT_ID,
                                              weight, args.input, args.ids, &pager);
    require(output == nullptr && pmi::PagedQwenGraph::abort(&pager), "unsupported operation/type/broadcast was not rejected safely");
    rejects([&] { pager.throw_if_failed(); }, "graph construction error");
  }
  require(provider.stats().io_bytes == 0 && provider.stats().active_leases == 0,
          "planning or rejected graph construction acquired weight pages");
}

// A complete routed FFN is required to exercise layer-scoped retention. The
// original operator tests deliberately keep exercising standalone LRU nodes.
class LayerGraph {
 public:
  LayerGraph(const Fixture& fixture, pmi::PagedQwenGraph& pager, int64_t token_count,
             int64_t route_count)
      : pager_(pager), descriptors_(true), token_count_(token_count),
        route_count_(route_count) {
    auto inputs = [&](Arena& arena, ggml_tensor*& ids, ggml_tensor*& input) {
      ids = ggml_new_tensor_2d(arena.ctx(), GGML_TYPE_I32, route_count, token_count);
      input = ggml_new_tensor_3d(arena.ctx(), GGML_TYPE_F32, fixture.columns, 1, token_count);
      auto* data = static_cast<float*>(input->data);
      for (int64_t i = 0; i < ggml_nelements(input); ++i)
        data[i] = std::sin(static_cast<float>(i) * 0.23f) + 0.125f;
    };
    ggml_tensor *input = nullptr, *reference_input = nullptr;
    inputs(paged_, ids_, input);
    inputs(resident_, reference_ids_, reference_input);
    ggml_tensor *deferred[3], *native[3];
    for (size_t i = 0; i < 3; ++i) {
      deferred[i] = descriptors_.weight(fixture.weights[i], false);
      require(pmi::PagedQwenGraph::select(deferred[i], &pager), "retention fixture source selection");
      native[i] = resident_.weight(fixture.weights[i], true);
    }
    auto project = [&](size_t index, ggml_tensor* x) {
      auto* output = pmi::PagedQwenGraph::build(paged_.ctx(), GGML_OP_MUL_MAT_ID,
                                               deferred[index], x, ids_, &pager);
      pager.throw_if_failed();
      require(output != nullptr, "retention projection construction");
      return output;
    };
    auto* gate = project(0, input);
    gate_graph_ = paged_.graph(gate);
    auto* up = project(1, input);
    auto* activation = ggml_mul(paged_.ctx(), ggml_silu(paged_.ctx(), gate), up);
    output_ = project(2, activation);
    auto* reference_gate = ggml_mul_mat_id(resident_.ctx(), native[0], reference_input, reference_ids_);
    auto* reference_up = ggml_mul_mat_id(resident_.ctx(), native[1], reference_input, reference_ids_);
    auto* reference_activation = ggml_mul(resident_.ctx(), ggml_silu(resident_.ctx(), reference_gate), reference_up);
    expected_ = ggml_mul_mat_id(resident_.ctx(), native[2], reference_activation, reference_ids_);
    graph_ = paged_.graph(output_);
    reference_graph_ = resident_.graph(expected_);
  }

  void run(const std::vector<int32_t>& routes) {
    require(routes.size() == static_cast<size_t>(token_count_ * route_count_), "retention route extent");
    set_ids(ids_, routes); set_ids(reference_ids_, routes);
    require(compute(reference_graph_, 1) == GGML_STATUS_SUCCESS, "retention resident FFN execution");
    require(compute(graph_, 2, &pager_) == GGML_STATUS_SUCCESS, "retention paged FFN execution");
    pager_.throw_if_failed();
    compare(values(output_), values(expected_), "retained complete FFN numerical parity");
  }

  void fail_with(const std::vector<int32_t>& routes) {
    set_ids(ids_, routes);
    require(compute(graph_, 2, &pager_) == GGML_STATUS_ABORTED, "invalid retained route did not abort");
    rejects([&] { pager_.throw_if_failed(); }, "invalid retained route propagation");
  }

  void run_gate_only(const std::vector<int32_t>& routes) {
    set_ids(ids_, routes);
    require(compute(gate_graph_, 1, &pager_) == GGML_STATUS_SUCCESS,
            "partially completed retained group failed its first projection");
    pager_.throw_if_failed();
  }

 private:
  pmi::PagedQwenGraph& pager_;
  Arena descriptors_, paged_, resident_;
  int64_t token_count_{}, route_count_{};
  ggml_tensor *ids_{}, *reference_ids_{}, *output_{}, *expected_{};
  ggml_cgraph *graph_{}, *reference_graph_{}, *gate_graph_{};
};

void device_projection_ffn(const Fixture& f) {
  pmi::FilePageProvider provider(f.paths, static_cast<size_t>(align64(f.expert_page_bytes)));
  auto executor = cpu_device_executor(provider, f, true);
  {
    pmi::PagedGraphExecutionOptions options;
    options.projection_executor = executor;
    pmi::PagedQwenGraph pager(f.manifest, provider, {}, options);
    LayerGraph graph(f, pager, 3, 4);
    graph.run(expert_ids);
    const auto first = pager.stats().device_projection;
    auto reordered = expert_ids;
    std::reverse(reordered.begin(), reordered.end());
    graph.run(reordered);
    const auto stats = pager.stats();
    require(stats.expert_requests == 2 * 3 * slots * tokens &&
            stats.device_projection.calls == stats.expert_requests &&
            stats.device_projection.cache.evictions > first.cache.evictions &&
            stats.device_projection.cache.peak_bytes <= stats.device_projection.cache.capacity_bytes &&
            stats.device_projection.cache.active_leases == 0 && provider.stats().active_leases == 0,
            "complete FFN device paging escaped bounded ownership or logical counters");
  }
  require(executor->stats().cache.current_bytes == 0 &&
          executor->stats().cache.active_leases == 0 &&
          executor->stats().graph_current_bytes == 0,
          "complete FFN graph destruction retained device allocations");
}

void retained_ids(pmi::PagedQwenGraph& pager, const std::vector<int32_t>& expected) {
  const auto snapshot = pager.retained_experts();
  std::set<int32_t> found;
  for (const auto& layer : snapshot) {
    require(layer.layer == 0, "single-layer fixture retained an unknown layer");
    found.insert(layer.expert_ids.begin(), layer.expert_ids.end());
  }
  require(found == std::set<int32_t>(expected.begin(), expected.end()),
          "retention selected the wrong last-token experts");
}

void retention_bounds(const Fixture& f, const pmi::PagedQwenGraph& pager,
                      const pmi::FilePageProvider& provider, uint64_t quota) {
  const auto stats = pager.stats();
  const auto cache = provider.stats();
  const auto bundle = 3 * align64(f.expert_page_bytes);
  require(stats.retention_requested_experts == 4 && stats.retention_effective_experts == quota,
          "retention effective quota mismatch");
  require(stats.retention_protected_bytes == quota * bundle, "committed protection byte count");
  require(stats.retention_peak_protected_bytes <= 2 * quota * bundle,
          "protection exceeded old/new layer transition allowance");
  require(stats.retention_peak_protected_bytes + stats.retention_service_reserve_bytes <= cache.capacity_bytes,
          "retained pages did not leave counted transient service capacity");
  require(cache.pinned_entries == quota * 3, "committed pin ownership mismatch");
  require(cache.active_leases == 0 && cache.peak_bytes <= cache.capacity_bytes,
          "retention escaped runtime payload/lease accounting");
}

void churn_unprotected(const Fixture& f, pmi::FilePageProvider& provider) {
  auto descriptor = *std::find_if(f.manifest.tensors.begin(), f.manifest.tensors.end(),
      [](const auto& tensor) { return tensor.source_tensor_name == "blk.0.ffn_gate_exps.weight" && tensor.expert_id == 0; });
  const auto original = descriptor.absolute_offset;
  // All ranges remain inside the synthetic first shard; their distinct offsets
  // cannot alias any aligned expert key. They exercise the actual Rust cache.
  for (uint64_t i = 1; i <= 64; ++i) {
    descriptor.absolute_offset = original + i * 64;
    auto page = provider.request(descriptor).get();
    require(page->size() == f.expert_page_bytes, "retention cache-churn page size");
  }
  require(provider.stats().active_leases == 0, "cache churn leaked leases");
}

void profile_stats(const pmi::PagedGraphStats& stats, bool enabled) {
  require(stats.profile_enabled == enabled, "profile enable flag");
  const double durations[] = {stats.provider_request_ms, stats.activation_copy_ms, stats.graph_setup_ms,
      stats.kernel_ms, stats.output_copy_ms, stats.graph_teardown_ms, stats.retention_ms,
      stats.expert_request_ms, stats.lookup_request_ms, stats.lookup_decode_ms};
  for (double duration : durations)
    require(std::isfinite(duration) && duration >= 0 && (enabled || duration == 0),
            "profile duration must be finite or zero when disabled");
  if (enabled) {
    require(stats.profile_projection_calls == stats.expert_requests,
            "profile projection count differs from actual routed requests");
    require(stats.profile_lookup_rows == stats.lookup_requests,
            "profile lookup count differs from actual row requests");
    require(stats.profile_provider_requests == stats.page_requests,
            "profile provider count differs from actual graph requests");
    require(stats.profile_expert_requests == stats.expert_requests &&
            stats.profile_lookup_requests == stats.lookup_requests, "profile request phase counts");
    require(std::abs(stats.provider_request_ms - stats.expert_request_ms - stats.lookup_request_ms) <= 1e-6,
            "profile provider phases do not reconcile");
    require(stats.provider_request_ms > 0, "enabled paging profile recorded no request time");
  } else {
    require(stats.profile_projection_calls == 0 && stats.profile_lookup_rows == 0 &&
            stats.profile_provider_requests == 0 && stats.profile_expert_requests == 0 &&
            stats.profile_lookup_requests == 0, "disabled paging profile collected operation counters");
  }
}

void retained_layers_case(const Fixture& f, bool profiling) {
  pmi::FilePageProvider provider(f.paths, 262144);
  {
    pmi::PagedGraphExecutionOptions options;
    options.retain_experts_per_layer = 4; options.profile = profiling;
    pmi::PagedQwenGraph pager(f.manifest, provider, {}, options);
    LayerGraph graph(f, pager, 1, 6);
    const std::vector<int32_t> initial{0, 1, 2, 3, 4, 5};
    graph.run(initial);
    retained_ids(pager, {0, 1, 2, 3});
    retention_bounds(f, pager, provider, 4);
    churn_unprotected(f, provider);
    auto before = provider.stats();
    graph.run(initial);
    auto after = provider.stats();
    require(after.hits - before.hits == 12 && after.misses - before.misses == 6,
            "repeated routes failed to preserve all twelve protected projection pages");
    require(after.io_bytes - before.io_bytes == 6 * f.expert_page_bytes,
            "repeated retained routes hid rereads");
    churn_unprotected(f, provider);
    before = provider.stats();
    graph.run({4, 5, 6, 3, 0, 1});
    after = provider.stats();
    require(after.hits - before.hits == 9 && after.misses - before.misses == 9,
            "previous top-four experts at current ranks five/six were released prematurely");
    require(after.io_bytes - before.io_bytes == 9 * f.expert_page_bytes,
            "changed retained routes hid rotation rereads");
    retained_ids(pager, {4, 5, 6, 3});
    retention_bounds(f, pager, provider, 4);
    profile_stats(pager.stats(), profiling);
    require(pager.stats().retention_transitions >= 3, "layer retention did not commit each pass");
  }
  require(provider.stats().pinned_entries == 0 && provider.stats().active_leases == 0,
          "retention destructor leaked pins or leases");
}

void retained_prefill_and_fallback(const Fixture& f) {
  const std::vector<int32_t> prefill{0, 1, 2, 3, 4, 5,
                                   1, 2, 3, 4, 5, 6,
                                   6, 5, 4, 3, 2, 1};
  for (const auto capacity_and_quota : {std::pair<size_t, uint64_t>{8192, 0},
                                      {65536, 1}, {131072, 2}, {262144, 4}}) {
    pmi::FilePageProvider provider(f.paths, capacity_and_quota.first);
    {
      pmi::PagedGraphExecutionOptions options; options.retain_experts_per_layer = 4;
      pmi::PagedQwenGraph pager(f.manifest, provider, {}, options);
      LayerGraph graph(f, pager, 3, 6);
      graph.run(prefill);
      const auto quota = capacity_and_quota.second;
      retained_ids(pager, std::vector<int32_t>(prefill.begin() + 12, prefill.begin() + 12 + quota));
      retention_bounds(f, pager, provider, quota);
      if (quota < 4) require(pager.stats().retention_fallbacks > 0,
                            "smaller effective quota was not reported as a fallback");
      profile_stats(pager.stats(), false);
    }
    require(provider.stats().pinned_entries == 0 && provider.stats().active_leases == 0,
            "prefill/fallback destruction leaked retention ownership");
  }
}

void retained_error_cleanup(const Fixture& f) {
  pmi::FilePageProvider provider(f.paths, 262144);
  pmi::PagedGraphExecutionOptions options; options.retain_experts_per_layer = 4;
  pmi::PagedQwenGraph pager(f.manifest, provider, {}, options);
  LayerGraph graph(f, pager, 1, 6);
  graph.run({0, 1, 2, 3, 4, 5});
  require(provider.stats().pinned_entries == 12, "error fixture did not establish protected pages");
  graph.fail_with({0, 1, 2, 3, 4, static_cast<int32_t>(f.experts)});
  require(provider.stats().pinned_entries == 0 && provider.stats().active_leases == 0,
          "failed retained graph did not release pins and leases");
}

void retained_io_cleanup(const Fixture& f) {
  auto paths = f.paths;
  const auto broken = paths[0].parent_path() / "retention-short-read.gguf";
  fs::copy_file(paths[0], broken, fs::copy_options::overwrite_existing);
  paths[0] = broken;
  pmi::FilePageProvider provider(paths, 262144);
  pmi::PagedGraphExecutionOptions options; options.retain_experts_per_layer = 4;
  pmi::PagedQwenGraph pager(f.manifest, provider, {}, options);
  LayerGraph graph(f, pager, 1, 6);
  graph.run({0, 1, 2, 3, 4, 5});
  fs::resize_file(broken, 4097);
  graph.fail_with({6, 5, 4, 3, 2, 1});
  require(provider.stats().failures > 0 && provider.stats().pinned_entries == 0 &&
          provider.stats().active_leases == 0, "short read leaked retained pages or leases");
}

void retained_partial_group_cleanup(const Fixture& f) {
  pmi::FilePageProvider provider(f.paths, 262144);
  {
    pmi::PagedGraphExecutionOptions options; options.retain_experts_per_layer = 4;
    pmi::PagedQwenGraph pager(f.manifest, provider, {}, options);
    LayerGraph graph(f, pager, 1, 6);
    graph.run({0, 1, 2, 3, 4, 5});
    graph.run_gate_only({4, 5, 6, 3, 0, 1});
    require(provider.stats().pinned_entries > 12 && provider.stats().active_leases == 0,
            "partial group did not hold old and arriving protected pages");
  }
  require(provider.stats().pinned_entries == 0 && provider.stats().active_leases == 0,
          "partial layer destruction leaked old or arriving protection");
}
} // namespace

int main() {
  try {
    ggml_cpu_init(); Directory directory;
    for (const auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_XS,
                            GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_0}) {
      maximum_error = 0; compared_values = 0;
      const auto fixture = make_fixture(directory.path / ggml_type_name(type), type);
      experts_case(fixture, 0, 1, false, false);
      experts_case(fixture, 0, 2, true, true);
      experts_case(fixture, 0, slots, true, false);
      experts_case(fixture, 1, 2, true, true);
      experts_case(fixture, 2, 2, true, true);
      lookup_case(fixture, false, false);
      lookup_case(fixture, true, true);
      if (type == GGML_TYPE_F32 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL) {
        experts_case(fixture, 0, 1, false, false, true);
        experts_case(fixture, 0, 2, true, true, true);
        experts_case(fixture, 0, slots, true, false, true);
        experts_case(fixture, 1, 2, true, true, true);
        experts_case(fixture, 2, 2, true, true, true);
        device_projection_ffn(fixture);
      }
      if (type == GGML_TYPE_F32) {
        split_case(fixture);
        split_case(fixture, true);
        split_case(make_fixture(directory.path / "split-cpu-io-failure", GGML_TYPE_F32), true, true);
        planning_and_callback_errors(fixture);
        for (const auto invalid : {int32_t{-1}, static_cast<int32_t>(fixture.experts)}) failure_case(fixture, false, invalid);
        for (const auto invalid : {int32_t{-1}, static_cast<int32_t>(fixture.lookup_rows)}) failure_case(fixture, true, invalid);
        failure_case(fixture, false, 0, true);
        failure_case(fixture, true, 0, true);
        failure_case(fixture, false, 0, true, false, true);
        failure_case(fixture, true, 0, true, false, true);
        failure_case(fixture, false, 0, false, true);
        failure_case(fixture, true, 0, false, true);
        tracing_limits(fixture);
        retained_layers_case(fixture, false);
        retained_layers_case(fixture, true);
        retained_prefill_and_fallback(fixture);
        retained_error_cleanup(fixture);
        retained_io_cleanup(fixture);
        retained_partial_group_cleanup(fixture);
        lookup_case(fixture, true, true, true);
        device_projection_failure(fixture, false);
        device_projection_failure(fixture, true);
        device_projection_configuration(fixture);
        device_projection_capacity(fixture, false);
        device_projection_capacity(fixture, true);
      }
      std::cout << ggml_type_name(type) << ": paged MUL_MAT_ID/GET_ROWS match resident ggml; "
                << "broadcast_slots=1,2,4 tokens=3 strided_indices=yes graph_reuse=yes "
                << "expert_page=" << fixture.expert_page_bytes << " lookup_row=" << fixture.lookup_page_bytes
                << " compared_values=" << compared_values << " max_abs_error=" << maximum_error << '\n';
    }
    std::cout << "Paged graph operators passed resident comparisons, eviction/reload, bounds, I/O/capacity failures, abort propagation, and bounded tracing. "
                 "Opt-in device projection used CPU fixtures only: compact experts, host bypass, full FFN, bounded eviction, late-failure/output-size cleanup; no GPU device was opened.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "paged graph test failed: " << error.what() << '\n'; return 1;
  }
}
