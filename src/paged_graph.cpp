#include "pmi/paged_graph.hpp"
#include "pmi/retention.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#ifdef PMI_WITH_LLAMA
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#endif

namespace pmi {
namespace {
[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("Qwen paged graph: " + message);
}

uint64_t add(uint64_t a, uint64_t b) {
  if (b > UINT64_MAX - a) fail("tensor size/offset overflow");
  return a + b;
}

uint64_t multiply(uint64_t a, uint64_t b) {
  if (a && b > UINT64_MAX / a) fail("tensor size/offset overflow");
  return a * b;
}

#ifdef PMI_WITH_LLAMA
using ProfileClock = std::chrono::steady_clock;
class PhaseTimer final {
 public:
  explicit PhaseTimer(double* destination) noexcept : destination_(destination) {
    if (destination_) start_ = ProfileClock::now();
  }
  ~PhaseTimer() {
    if (destination_) *destination_ += std::chrono::duration<double, std::milli>(ProfileClock::now() - start_).count();
  }
  PhaseTimer(const PhaseTimer&) = delete;
  PhaseTimer& operator=(const PhaseTimer&) = delete;
 private:
  double* destination_;
  ProfileClock::time_point start_{};
};

// The numerical work stays in the retained ggml CPU backend. Only one leased
// expert is attached to this short-lived graph, with no copying/repacking of
// the immutable quantized matrix by PMI. The graph dies before its page lease.
class CpuProjection final {
 public:
  CpuProjection(const ManifestTensor& descriptor, const Page& page,
                const std::vector<float>& input, PagedGraphStats* profile)
      : profile_(profile) {
    try {
      ggml_cgraph* graph;
      ggml_cplan plan{};
      {
        PhaseTimer timer(profile_ ? &profile_->graph_setup_ms : nullptr);
        ggml_init_params params{};
        params.mem_size = 32 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false);
        params.no_alloc = true;
        ctx_ = ggml_init(params);
        if (!ctx_) fail("cannot allocate nested CPU context");
        auto* weight = ggml_new_tensor_2d(ctx_, static_cast<ggml_type>(descriptor.ggml_type),
            static_cast<int64_t>(descriptor.shape[0]), static_cast<int64_t>(descriptor.shape[1]));
        weight->data = const_cast<uint8_t*>(page.data());
        auto* x = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, static_cast<int64_t>(input.size()), 1);
        x->data = const_cast<float*>(input.data());
        output_ = ggml_mul_mat(ctx_, weight, x);
        graph = ggml_new_graph_custom(ctx_, 32, false);
        ggml_build_forward_expand(graph, output_);
        buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_, ggml_backend_cpu_buffer_type());
        if (!buffer_) fail("cannot allocate nested CPU output");
        plan = ggml_graph_plan(graph, 1, nullptr);
        workspace_.resize(plan.work_size);
        plan.work_data = workspace_.data();
      }
      {
        PhaseTimer timer(profile_ ? &profile_->kernel_ms : nullptr);
        if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) fail("nested CPU matrix multiplication failed");
      }
    } catch (...) {
      release();
      throw;
    }
  }
  ~CpuProjection() { release(); }
  CpuProjection(const CpuProjection&) = delete;
  CpuProjection& operator=(const CpuProjection&) = delete;
  const float* data() const { return static_cast<const float*>(output_->data); }
 private:
  void release() noexcept {
    PhaseTimer timer(profile_ ? &profile_->graph_teardown_ms : nullptr);
    ggml_backend_buffer_free(buffer_);
    ggml_free(ctx_);
    std::vector<uint8_t>().swap(workspace_);
  }
  PagedGraphStats* profile_{};
  ggml_context* ctx_{};
  ggml_backend_buffer_t buffer_{};
  ggml_tensor* output_{};
  std::vector<uint8_t> workspace_;
};

int32_t read_id(const ggml_tensor* ids, int64_t i0, int64_t i1 = 0,
                int64_t i2 = 0) {
  int32_t value;
  const auto* p = static_cast<const uint8_t*>(ids->data) + i0 * ids->nb[0] +
      i1 * ids->nb[1] + i2 * ids->nb[2];
  std::memcpy(&value, p, sizeof(value));
  return value;
}
#endif
} // namespace

struct PagedQwenGraph::Impl {
  FilePageProvider& provider;
  PagedGraphTraceOptions options;
  PagedGraphExecutionOptions execution_options;
  mutable std::mutex mutex;
  mutable std::mutex projection_mutex;
  std::atomic<bool> failed{false};
  std::exception_ptr failure;
  PagedGraphStats counters;
  std::vector<PagedGraphEvent> events;
  bool device_seed_finished{}; // Evaluation boundaries serialize with all node/worker work.

  explicit Impl(FilePageProvider& p, PagedGraphTraceOptions o, PagedGraphExecutionOptions e)
      : provider(p), options(o), execution_options(e) {
    if (e.retain_experts_per_layer > 4) fail("retention quota must be between 0 and 4");
    if (e.retain_experts_per_layer && e.projection_executor)
      fail("host expert retention cannot be combined with a device projection executor");
    if (e.split_cpu_misses && (!e.projection_executor || !e.projection_executor->supports_resident_split()))
      fail("split CPU misses requires a resident-capable projection executor");
    counters.retention_requested_experts = e.retain_experts_per_layer;
    counters.profile_enabled = e.profile;
    counters.device_projection_enabled = static_cast<bool>(e.projection_executor);
    counters.split_enabled = e.split_cpu_misses;
  }

  ~Impl() {
    clear_projection();
  }

  void clear_projection() noexcept {
    if (!execution_options.projection_executor) return;
    try {
      std::lock_guard<std::mutex> lock(projection_mutex);
      execution_options.projection_executor->clear();
    } catch (...) { /* Destruction and error latching must remain noexcept. */ }
  }

  void latch() noexcept {
    // The failing project() call has unwound and released its consumer lease.
    // Clear completed uploads as well; no resident or CPU fallback is allowed.
    clear_projection();
    try {
      std::lock_guard<std::mutex> lock(mutex);
      if (!failure) failure = std::current_exception();
#ifdef PMI_WITH_LLAMA
      if (retention) {
        try { retention->clear(); } catch (...) { /* Retried by the helper destructor. */ }
        update_retention_stats();
      }
      for (auto& item : layers) {
        item.second->committed_ids.clear();
        item.second->routing_ids.clear();
        item.second->next_ids.clear();
        item.second->completed = 0;
      }
#endif
    } catch (...) { /* The atomic flag still records a failure. */ }
    failed.store(true, std::memory_order_release);
  }

#ifdef PMI_WITH_LLAMA
  struct ProfileScope {
    Impl& owner;
    PagedGraphStats delta;
    explicit ProfileScope(Impl& value) : owner(value) {}
    PagedGraphStats* get() { return owner.execution_options.profile ? &delta : nullptr; }
    ~ProfileScope() noexcept {
      if (!owner.execution_options.profile) return;
      try {
        std::lock_guard<std::mutex> lock(owner.mutex);
        auto& out = owner.counters;
        out.expert_request_ms += delta.expert_request_ms;
        out.lookup_request_ms += delta.lookup_request_ms;
        out.provider_request_ms = out.expert_request_ms + out.lookup_request_ms;
        out.activation_copy_ms += delta.activation_copy_ms;
        out.graph_setup_ms += delta.graph_setup_ms;
        out.kernel_ms += delta.kernel_ms;
        out.output_copy_ms += delta.output_copy_ms;
        out.graph_teardown_ms += delta.graph_teardown_ms;
        out.retention_ms += delta.retention_ms;
        out.lookup_decode_ms += delta.lookup_decode_ms;
        out.profile_projection_calls += delta.profile_projection_calls;
        out.profile_lookup_rows += delta.profile_lookup_rows;
        out.profile_provider_requests += delta.profile_provider_requests;
        out.profile_expert_requests += delta.profile_expert_requests;
        out.profile_lookup_requests += delta.profile_lookup_requests;
        out.split_critical_ms += delta.split_critical_ms;
        out.split_cpu_ms += delta.split_cpu_ms;
        out.split_device_ms += delta.split_device_ms;
        out.split_join_ms += delta.split_join_ms;
        out.split_overlap_ms += delta.split_overlap_ms;
      } catch (...) { /* Profiling must not escape a computation cleanup path. */ }
    }
  };
  struct Layer;
  struct Source {
    Impl* owner{};
    std::string name;
    bool lookup{};
    std::array<int64_t, 4> shape{};
    uint32_t type{};
    std::vector<ManifestTensor> pages;
    uint64_t row_bytes{};
    uint64_t logical_bytes{};
    bool selected{}; // Protected by owner->mutex; callbacks may repeat a source.
    Layer* layer{};
    unsigned projection{}; // gate=0, up=1, down=2
  };
  struct Layer {
    int id{-1};
    std::array<Source*, 3> sources{};
    uint64_t bundle_charge{};
    std::vector<int32_t> routing_ids, next_ids, committed_ids;
    int64_t slots{}, tokens{};
    unsigned completed{};
  };
  // unique_ptr keeps custom-op userdata stable even during map rehashes.
  std::unordered_map<std::string, std::unique_ptr<Source>> sources;
  std::map<int, std::unique_ptr<Layer>> layers;
  std::unique_ptr<ProtectedPageRetention> retention;
  bool retention_planned{};
  int64_t planned_routed_count{};

  static uint64_t charge(uint64_t bytes) { return add(bytes, 63) & ~uint64_t{63}; }

  void update_retention_stats() noexcept {
    if (!retention) return;
    counters.retention_protected_bytes = retention->protected_bytes();
    counters.retention_peak_protected_bytes = retention->peak_protected_bytes();
  }

  static void validate_page(const ManifestTensor& tensor, const ModelManifest& manifest, bool lookup) {
    if (tensor.shape.size() != 2 || !tensor.shape[0] || !tensor.shape[1] ||
        tensor.shape[0] > INT32_MAX || tensor.shape[1] > INT32_MAX)
      fail("unsupported matrix dimensions: " + tensor.name);
    if (tensor.ggml_type >= GGML_TYPE_COUNT) fail("unsupported tensor type: " + tensor.name);
    const auto type = static_cast<ggml_type>(tensor.ggml_type);
    const auto* traits = ggml_get_type_traits(type);
    const auto* cpu = ggml_get_type_traits_cpu(type);
    if (!traits || traits->blck_size <= 0 || !traits->type_size ||
        tensor.shape[0] % static_cast<uint64_t>(traits->blck_size))
      fail("unsupported or unaligned quantized row: " + tensor.name);
    if (lookup) {
      if (type != GGML_TYPE_F32 && !traits->to_float)
        fail("ggml cannot dequantize lookup rows: " + tensor.name);
    } else if (!cpu || !cpu->vec_dot) {
      fail("ggml CPU cannot multiply expert type: " + tensor.name);
    }
    if (tensor.byte_length != ggml_tensor_nbytes(tensor.ggml_type, tensor.shape))
      fail("matrix byte size disagrees with shape: " + tensor.name);
    add(tensor.absolute_offset, tensor.byte_length);
    add(tensor.offset, tensor.byte_length);
    if (tensor.shard_id >= manifest.shards.size()) fail("missing shard for " + tensor.name);
    const auto& shard = manifest.shards[tensor.shard_id];
    if (shard.shard_id != tensor.shard_id || tensor.absolute_offset < shard.tensor_data_offset ||
        tensor.offset != tensor.absolute_offset - shard.tensor_data_offset ||
        tensor.absolute_offset > shard.file_size || tensor.byte_length > shard.file_size - tensor.absolute_offset)
      fail("matrix range is outside shard: " + tensor.name);
  }

  void initialize(const ModelManifest& manifest) {
    if (manifest.model_architecture != "qwen4exp") fail("expected qwen4exp architecture");
    ggml_cpu_init();
    const std::regex expert_name("^blk\\.([0-9]+)\\.ffn_(gate|up|down)_exps\\.weight$");
    std::map<std::string, std::vector<const ManifestTensor*>> grouped;
    std::map<std::string, std::map<std::string, std::string>> layer_names;
    for (const auto& tensor : manifest.tensors) {
      std::smatch match;
      if (std::regex_match(tensor.source_tensor_name, match, expert_name)) {
        if (tensor.expert_id < 0) fail("expert source is not sliced: " + tensor.source_tensor_name);
        validate_page(tensor, manifest, false);
        grouped[tensor.source_tensor_name].push_back(&tensor);
        layer_names[match[1].str()][match[2].str()] = tensor.source_tensor_name;
      } else if (tensor.source_tensor_name.find("_exps") != std::string::npos) {
        fail("unsupported fused/scaled expert layout: " + tensor.source_tensor_name);
      } else if (tensor.source_tensor_name == "per_layer_token_embd.weight") {
        if (tensor.expert_id != -1) fail("unexpected expert ID on lookup table");
        validate_page(tensor, manifest, true);
        auto source = std::make_unique<Source>();
        source->owner = this;
        source->name = tensor.source_tensor_name;
        source->lookup = true;
        source->shape = {static_cast<int64_t>(tensor.shape[0]), static_cast<int64_t>(tensor.shape[1]), 1, 1};
        source->type = tensor.ggml_type;
        source->pages.push_back(tensor);
        source->row_bytes = ggml_tensor_nbytes(tensor.ggml_type, {tensor.shape[0]});
        source->logical_bytes = tensor.byte_length;
        counters.candidate_weight_bytes = add(counters.candidate_weight_bytes, source->logical_bytes);
        counters.largest_page_bytes = std::max(counters.largest_page_bytes, source->row_bytes);
        if (!sources.emplace(source->name, std::move(source)).second) fail("duplicate lookup source");
      } else if (tensor.source_tensor_name.rfind("per_layer_token_embd", 0) == 0) {
        fail("unsupported scaled/split PLE layout: " + tensor.source_tensor_name);
      }
    }
    for (auto& group : grouped) {
      auto& slices = group.second;
      std::sort(slices.begin(), slices.end(), [](const auto* a, const auto* b) { return a->expert_id < b->expert_id; });
      if (slices.empty() || slices.size() > INT32_MAX) fail("invalid expert count: " + group.first);
      const auto& base = *slices.front();
      auto source = std::make_unique<Source>();
      source->owner = this;
      source->name = group.first;
      source->shape = {static_cast<int64_t>(base.shape[0]), static_cast<int64_t>(base.shape[1]),
                       static_cast<int64_t>(slices.size()), 1};
      source->type = base.ggml_type;
      source->pages.reserve(slices.size());
      for (size_t i = 0; i < slices.size(); ++i) {
        const auto& slice = *slices[i];
        const auto displacement = multiply(i, base.byte_length);
        if (slice.expert_id != static_cast<int>(i) || slice.shape != base.shape ||
            slice.ggml_type != base.ggml_type || slice.shard_id != base.shard_id ||
            slice.byte_length != base.byte_length || slice.layer != base.layer ||
            slice.absolute_offset != add(base.absolute_offset, displacement) ||
            slice.offset != add(base.offset, displacement))
          fail("inconsistent packed expert slices: " + group.first);
        source->pages.push_back(slice);
      }
      source->logical_bytes = multiply(base.byte_length, slices.size());
      counters.candidate_weight_bytes = add(counters.candidate_weight_bytes, source->logical_bytes);
      counters.largest_page_bytes = std::max(counters.largest_page_bytes, base.byte_length);
      sources.emplace(group.first, std::move(source));
    }
    for (const auto& layer : layer_names) {
      const auto& names = layer.second;
      if (names.size() != 3 || !names.count("gate") || !names.count("up") || !names.count("down"))
        fail("requires separate gate/up/down expert tensors in layer " + layer.first);
      const auto& gate = *sources.at(names.at("gate"));
      const auto& up = *sources.at(names.at("up"));
      const auto& down = *sources.at(names.at("down"));
      if (gate.shape != up.shape || gate.shape[0] != down.shape[1] ||
          gate.shape[1] != down.shape[0] || gate.shape[2] != down.shape[2])
        fail("inconsistent gate/up/down dimensions in layer " + layer.first);
      const auto index = std::stoull(layer.first);
      if (index > INT32_MAX) fail("layer index is too large");
      auto layout = std::make_unique<Layer>();
      layout->id = static_cast<int>(index);
      for (unsigned projection = 0; projection < 3; ++projection) {
        const char* suffix = projection == 0 ? "gate" : projection == 1 ? "up" : "down";
        auto* source = sources.at(names.at(suffix)).get();
        if (source->pages.front().layer != layout->id) fail("incorrect manifest layer for " + source->name);
        layout->sources[projection] = source;
        layout->bundle_charge = add(layout->bundle_charge, charge(source->pages.front().byte_length));
        source->layer = layout.get();
        source->projection = projection;
      }
      layers.emplace(layout->id, std::move(layout));
    }
    if (sources.empty()) fail("manifest has no supported deferred tensors");
  }

  void plan_retention(int64_t routed_count) {
    if (retention_planned) {
      if (routed_count != planned_routed_count) fail("routing slot count changed after retention planning");
      return;
    }
    retention_planned = true;
    planned_routed_count = routed_count;
    const auto capacity = provider.stats().capacity_bytes;
    const auto largest = charge(counters.largest_page_bytes);
    constexpr uint64_t target_reserve = 8 * 1024 * 1024;
    const auto service = std::max(largest, std::min(target_reserve, std::max(largest, capacity / 16)));
    counters.retention_service_reserve_bytes = service;
    uint64_t all_bundles = 0, largest_bundle = 0;
    // Every validated layer participates in this conservative bound, including
    // optional blocks not instantiated by the current decoder. This cannot
    // underestimate the space needed if another validated layer is later used.
    for (const auto& item : layers) {
      all_bundles = add(all_bundles, item.second->bundle_charge);
      largest_bundle = std::max(largest_bundle, item.second->bundle_charge);
    }
    const auto turnover = add(all_bundles, largest_bundle);
    uint64_t quota = std::min<uint64_t>(execution_options.retain_experts_per_layer,
                                      static_cast<uint64_t>(routed_count));
    if (!turnover || service > capacity) quota = 0;
    else quota = std::min(quota, (capacity - service) / turnover);
    counters.retention_effective_experts = quota;
    if (quota < execution_options.retain_experts_per_layer) ++counters.retention_fallbacks;
    if (quota) retention = std::make_unique<ProtectedPageRetention>(provider, capacity - service);
  }

  void begin_retention(const Source& source, const ggml_tensor* ids, PagedGraphStats* profile) {
    if (!execution_options.retain_experts_per_layer) return;
    PhaseTimer timer(profile ? &profile->retention_ms : nullptr);
    std::lock_guard<std::mutex> lock(mutex);
    plan_retention(ids->ne[0]);
    if (!retention) return;
    auto& layer = *source.layer;
    const auto count = multiply(static_cast<uint64_t>(ids->ne[0]), static_cast<uint64_t>(ids->ne[1]));
    if (count > SIZE_MAX) fail("routing snapshot size overflow");
    std::vector<int32_t> routing;
    routing.reserve(static_cast<size_t>(count));
    for (int64_t token = 0; token < ids->ne[1]; ++token) {
      for (int64_t slot = 0; slot < ids->ne[0]; ++slot) {
        const auto id = read_id(ids, slot, token);
        if (id < 0 || id >= source.shape[2]) fail("retention expert ID out of range for " + source.name);
        routing.push_back(id);
      }
    }
    if (!retention->group_active()) {
      if (source.projection == 2 || layer.completed) fail("retention group must start with gate or up");
      layer.slots = ids->ne[0];
      layer.tokens = ids->ne[1];
      layer.routing_ids = std::move(routing);
      layer.next_ids.clear();
      std::vector<TensorInfo> next;
      for (uint64_t slot = 0; slot < counters.retention_effective_experts; ++slot) {
        const auto id = layer.routing_ids[static_cast<size_t>((layer.tokens - 1) * layer.slots + slot)];
        if (std::find(layer.next_ids.begin(), layer.next_ids.end(), id) != layer.next_ids.end()) continue;
        layer.next_ids.push_back(id);
        for (const auto* projection : layer.sources) next.push_back(projection->pages[static_cast<size_t>(id)]);
      }
      retention->begin_group(static_cast<uint64_t>(layer.id), next);
    } else {
      if (retention->active_group_id() != static_cast<uint64_t>(layer.id))
        fail("overlapping layer retention groups");
      if (layer.slots != ids->ne[0] || layer.tokens != ids->ne[1] || layer.routing_ids != routing)
        fail("routing IDs changed between expert projections in layer " + std::to_string(layer.id));
    }
    if (layer.completed & (1u << source.projection)) fail("expert projection repeated before down completion");
    if (source.projection == 2 && layer.completed != 3) fail("down projection preceded gate/up completion");
    update_retention_stats();
  }

  void retain_loaded(const ManifestTensor& descriptor, PagedGraphStats* profile) {
    if (!retention) return;
    PhaseTimer timer(profile ? &profile->retention_ms : nullptr);
    std::lock_guard<std::mutex> lock(mutex);
    retention->observe_loaded(descriptor);
    update_retention_stats();
  }

  void finish_retention(const Source& source, PagedGraphStats* profile) {
    if (!retention) return;
    PhaseTimer timer(profile ? &profile->retention_ms : nullptr);
    std::lock_guard<std::mutex> lock(mutex);
    auto& layer = *source.layer;
    layer.completed |= 1u << source.projection;
    if (source.projection == 2) {
      retention->finish_group(static_cast<uint64_t>(layer.id));
      layer.committed_ids = std::move(layer.next_ids);
      layer.routing_ids.clear();
      layer.completed = 0;
      ++counters.retention_transitions;
    }
    update_retention_stats();
  }

  static void validate_metadata(const Source& source, const ggml_tensor* tensor) {
    if (!tensor || static_cast<uint32_t>(tensor->type) != source.type || !ggml_is_contiguous(tensor))
      fail("metadata type/strides differ for " + source.name);
    for (int i = 0; i < 4; ++i)
      if (tensor->ne[i] != source.shape[i]) fail("metadata shape differs for " + source.name);
  }

  void count_request(bool lookup) {
    std::lock_guard<std::mutex> lock(mutex);
    ++counters.page_requests;
    if (lookup) ++counters.lookup_requests; else ++counters.expert_requests;
  }

  PageLease request(const ManifestTensor& descriptor, bool lookup, PagedGraphStats* profile, bool logical = true) {
    if (logical) count_request(lookup);
    if (profile) {
      ++profile->profile_provider_requests;
      if (lookup) ++profile->profile_lookup_requests; else ++profile->profile_expert_requests;
    }
    PhaseTimer timer(!profile ? nullptr : lookup ? &profile->lookup_request_ms : &profile->expert_request_ms);
    auto page = provider.request(descriptor).get();
    if (!page || page->size() != descriptor.byte_length) fail("page size differs for " + descriptor.name);
    if (reinterpret_cast<uintptr_t>(page->data()) % 64) fail("page is not 64-byte aligned: " + descriptor.name);
    return page;
  }

  void record(const Source& source, int64_t token, int64_t slot, int64_t id,
              const float* output, size_t count) {
    if (!options.enabled) return;
    std::lock_guard<std::mutex> lock(mutex);
    if (events.size() >= options.max_events) { ++counters.trace_dropped; return; }
    const auto saved = std::min(count, options.max_output_values_per_event);
    PagedGraphEvent event;
    event.tensor_name = source.name;
    event.operation = source.lookup ? "get_rows" : "mul_mat_id";
    event.token = token;
    event.slot = slot;
    event.id = id;
    event.output.assign(output, output + saved);
    event.output_truncated = saved != count;
    events.push_back(std::move(event));
  }

  void project_split(const Source& source, ggml_tensor* dst, PagedGraphStats* profile) {
    const auto* x = dst->src[0];
    const auto* ids = dst->src[1];
    struct Job {
      const ManifestTensor* descriptor{};
      DeviceLease lease;
      std::vector<float> input, result;
      int32_t id{};
      int64_t slot{};
    };
    // At most 64 rows and 4MiB of input+result vectors per group. Existing
    // kernels remain single-matrix; this groups dispatch/join, not GPU matmuls.
    const auto vector_bytes = multiply(add(static_cast<uint64_t>(x->ne[0]),
                                          static_cast<uint64_t>(source.shape[1])), sizeof(float));
    constexpr uint64_t staging_limit = 4 * 1024 * 1024;
    if (!vector_bytes || vector_bytes > staging_limit) fail("split vector exceeds 4MiB staging limit");
    const auto group_limit = static_cast<int64_t>(std::min<uint64_t>(64, staging_limit / vector_bytes));
    for (int64_t token = 0; token < x->ne[2]; ++token) {
      for (int64_t first = 0; first < ids->ne[0]; first += group_limit) {
        PhaseTimer critical(profile ? &profile->split_critical_ms : nullptr);
        std::lock_guard<std::mutex> ordered(projection_mutex);
        std::vector<Job> jobs(static_cast<size_t>(std::min(group_limit, ids->ne[0] - first)));
        uint64_t hits = 0;
        // Complete every cache operation BEFORE launching the device worker.
        // Immutable hit leases pin the allocation through computation/readback.
        for (size_t i = 0; i < jobs.size(); ++i) {
          auto& job = jobs[i]; job.slot = first + static_cast<int64_t>(i);
          job.id = read_id(ids, job.slot, token);
          if (job.id < 0 || job.id >= source.shape[2]) fail("split expert ID out of range");
          job.descriptor = &source.pages[static_cast<size_t>(job.id)];
          {
            PhaseTimer timer(profile ? &profile->activation_copy_ms : nullptr);
            job.input.resize(static_cast<size_t>(x->ne[0]));
            const auto* activation = static_cast<const uint8_t*>(x->data) +
                (job.slot % x->ne[1]) * x->nb[1] + token * x->nb[2];
            for (int64_t col = 0; col < x->ne[0]; ++col)
              std::memcpy(&job.input[static_cast<size_t>(col)], activation + col * x->nb[0], sizeof(float));
          }
          count_request(false);
          if (profile) ++profile->profile_projection_calls;
          job.lease = execution_options.projection_executor->resident(*job.descriptor, !device_seed_finished);
          if (job.lease) ++hits;
        }
        {
          std::lock_guard<std::mutex> lock(mutex);
          ++counters.split_groups;
          counters.split_device_projections += hits;
          counters.split_cpu_projections += jobs.size() - hits;
        }
        using Interval = std::pair<ProfileClock::time_point, ProfileClock::time_point>;
        std::future<Interval> device;
        Interval cpu_interval;
        if (hits) {
          device = std::async(std::launch::async, [&] {
            const auto start = profile ? ProfileClock::now() : ProfileClock::time_point{};
            for (auto& job : jobs) if (job.lease)
              job.result = execution_options.projection_executor->project_resident(job.lease, job.input);
            return Interval{start, profile ? ProfileClock::now() : ProfileClock::time_point{}};
          });
        }
        try {
          if (hits != jobs.size()) {
            PhaseTimer cpu(profile ? &profile->split_cpu_ms : nullptr);
            if (profile) cpu_interval.first = ProfileClock::now();
            for (auto& job : jobs) if (!job.lease) {
              auto page = request(*job.descriptor, false, profile, false);
              CpuProjection graph(*job.descriptor, *page, job.input, profile);
              job.result.assign(graph.data(), graph.data() + source.shape[1]);
            }
            if (profile) cpu_interval.second = ProfileClock::now();
          }
          if (device.valid()) {
            PhaseTimer join(profile ? &profile->split_join_ms : nullptr);
            const auto elapsed = device.get();
            if (profile) {
              profile->split_device_ms += std::chrono::duration<double, std::milli>(elapsed.second - elapsed.first).count();
              const auto begin = std::max(cpu_interval.first, elapsed.first);
              const auto end = std::min(cpu_interval.second, elapsed.second);
              if (end > begin) profile->split_overlap_ms += std::chrono::duration<double, std::milli>(end - begin).count();
            }
          }
        } catch (...) {
          // Drain on CPU failure too. No job/input/lease/cache can disappear
          // while the device still consumes it, even if both branches fail.
          const auto original = std::current_exception();
          if (device.valid()) { try { device.get(); } catch (...) {} }
          std::rethrow_exception(original);
        }
        for (auto& job : jobs) {
          const auto count = static_cast<size_t>(source.shape[1]);
          if (job.result.size() != count) fail("split projection output size differs");
          auto* output = reinterpret_cast<float*>(static_cast<uint8_t*>(dst->data) + job.slot * dst->nb[1] + token * dst->nb[2]);
          {
            PhaseTimer timer(profile ? &profile->output_copy_ms : nullptr);
            std::memcpy(output, job.result.data(), count * sizeof(float));
          }
          record(source, token, job.slot, job.id, output, count);
        }
      }
    }
  }

  void project(const Source& source, ggml_tensor* dst) {
    ProfileScope profiling(*this);
    auto* profile = profiling.get();
    const auto* x = dst->src[0];
    const auto* ids = dst->src[1];
    if (!x || !ids || !x->data || !ids->data) fail("missing expert activation or routing IDs");
    begin_retention(source, ids, profile);
    if (execution_options.split_cpu_misses) {
      project_split(source, dst, profile);
      finish_retention(source, profile);
      return;
    }
    std::vector<float> input;
    {
      PhaseTimer timer(profile ? &profile->activation_copy_ms : nullptr);
      input.resize(static_cast<size_t>(x->ne[0]));
    }
    for (int64_t token = 0; token < x->ne[2]; ++token) {
      for (int64_t slot = 0; slot < ids->ne[0]; ++slot) {
        const auto id = read_id(ids, slot, token);
        if (id < 0 || id >= source.shape[2]) fail("expert ID out of range for " + source.name);
        {
          PhaseTimer timer(profile ? &profile->activation_copy_ms : nullptr);
          const auto* activation = static_cast<const uint8_t*>(x->data) +
              (slot % x->ne[1]) * x->nb[1] + token * x->nb[2];
          for (int64_t col = 0; col < x->ne[0]; ++col)
            std::memcpy(&input[static_cast<size_t>(col)], activation + col * x->nb[0], sizeof(float));
        }
        const auto& descriptor = source.pages[static_cast<size_t>(id)];
        auto* output = reinterpret_cast<float*>(static_cast<uint8_t*>(dst->data) + slot * dst->nb[1] + token * dst->nb[2]);
        const auto count = static_cast<size_t>(source.shape[1]);
        if (execution_options.projection_executor) {
          if (profile) ++profile->profile_projection_calls;
          // Request the compact selected expert through the executor first:
          // a device hit must not touch the provider or create a host lease.
          std::vector<float> projected;
          {
            std::lock_guard<std::mutex> lock(projection_mutex);
            count_request(false);
            projected = execution_options.projection_executor->project(descriptor, input);
          }
          if (projected.size() != count) fail("projection executor output size differs for " + descriptor.name);
          {
            PhaseTimer timer(profile ? &profile->output_copy_ms : nullptr);
            std::memcpy(output, projected.data(), count * sizeof(float));
          }
        } else {
          auto page = request(descriptor, false, profile);
          retain_loaded(descriptor, profile);
          if (profile) ++profile->profile_projection_calls;
          CpuProjection graph(descriptor, *page, input, profile);
          {
            PhaseTimer timer(profile ? &profile->output_copy_ms : nullptr);
            std::memcpy(output, graph.data(), count * sizeof(float));
          }
        }
        record(source, token, slot, id, output, count);
      }
    }
    finish_retention(source, profile);
  }

  void lookup(const Source& source, ggml_tensor* dst) {
    ProfileScope profiling(*this);
    auto* profile = profiling.get();
    const auto* ids = dst->src[0];
    if (!ids || !ids->data) fail("missing lookup row IDs");
    const auto type = static_cast<ggml_type>(source.type);
    const auto* traits = ggml_get_type_traits(type);
    for (int64_t index = 0; index < ids->ne[0]; ++index) {
      const auto id = read_id(ids, index);
      if (id < 0 || id >= source.shape[1]) fail("PLE row ID out of range");
      auto row = source.pages.front();
      const auto displacement = multiply(static_cast<uint64_t>(id), source.row_bytes);
      row.name += "#row:" + std::to_string(id);
      row.shape[1] = 1;
      row.absolute_offset = add(row.absolute_offset, displacement);
      row.offset = add(row.offset, displacement);
      row.byte_length = source.row_bytes;
      row.required_alignment = 1;
      if (profile) ++profile->profile_lookup_rows;
      auto page = request(row, true, profile);
      auto* output = reinterpret_cast<float*>(static_cast<uint8_t*>(dst->data) + index * dst->nb[1]);
      {
        PhaseTimer timer(profile ? &profile->lookup_decode_ms : nullptr);
        if (type == GGML_TYPE_F32) std::memcpy(output, page->data(), static_cast<size_t>(source.row_bytes));
        else traits->to_float(page->data(), output, source.shape[0]);
      }
      record(source, index, 0, id, output, static_cast<size_t>(source.shape[0]));
    }
  }

  static void compute(ggml_tensor* dst, int ith, int /*nth*/, void* userdata) noexcept {
    if (ith != 0) return; // Custom nodes explicitly request one CPU task.
    auto& source = *static_cast<Source*>(userdata);
    auto& owner = *source.owner;
    try {
      if (!dst || !dst->data || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst))
        fail("invalid custom node output");
      if (owner.failed.load(std::memory_order_acquire)) {
        std::memset(dst->data, 0, ggml_nbytes(dst));
        return;
      }
      {
        std::lock_guard<std::mutex> lock(owner.mutex);
        ++owner.counters.executed_nodes;
      }
      if (source.lookup) owner.lookup(source, dst); else owner.project(source, dst);
    } catch (...) {
      owner.latch();
      if (dst && dst->data) std::memset(dst->data, 0, ggml_nbytes(dst));
    }
  }
#endif
};

PagedQwenGraph::PagedQwenGraph(const ModelManifest& manifest, FilePageProvider& provider,
                               PagedGraphTraceOptions options, PagedGraphExecutionOptions execution_options)
    : impl_(std::make_unique<Impl>(provider, options, execution_options)) {
#ifdef PMI_WITH_LLAMA
  impl_->initialize(manifest);
#else
  (void) manifest;
  fail("execution requires PMI_WITH_LLAMA and the retained ggml CPU backend");
#endif
}

PagedQwenGraph::~PagedQwenGraph() = default;

#ifdef PMI_WITH_LLAMA
bool PagedQwenGraph::select(const ggml_tensor* metadata, void* userdata) noexcept {
  if (!userdata || !metadata) return false;
  auto& impl = *static_cast<PagedQwenGraph*>(userdata)->impl_;
  bool selected = false;
  try {
    const auto it = impl.sources.find(metadata->name);
    if (it == impl.sources.end()) return false;
    selected = true;
    Impl::validate_metadata(*it->second, metadata);
    {
      std::lock_guard<std::mutex> lock(impl.mutex);
      auto& source = *it->second;
      if (!source.selected) {
        impl.counters.deferred_weight_bytes = add(impl.counters.deferred_weight_bytes, source.logical_bytes);
        source.selected = true;
      }
    }
    return true;
  } catch (...) {
    impl.latch();
    // Preserve deferral for a known name, even when malformed. The caller must
    // check throw_if_failed() after model loading; do not load huge fallback data.
    return selected;
  }
}

ggml_tensor* PagedQwenGraph::build(ggml_context* ctx, ggml_op op, const ggml_tensor* weight,
                                  ggml_tensor* input, ggml_tensor* ids, void* userdata) noexcept {
  if (!userdata) return nullptr;
  auto& impl = *static_cast<PagedQwenGraph*>(userdata)->impl_;
  try {
    if (impl.failed.load(std::memory_order_acquire)) return nullptr;
    if (!ctx || !weight) fail("missing graph context or weight metadata");
    const auto it = impl.sources.find(weight->name);
    if (it == impl.sources.end()) fail("unrecognized deferred weight");
    auto& source = *it->second;
    Impl::validate_metadata(source, weight);
    if (source.lookup) {
      if (op != GGML_OP_GET_ROWS) fail("PLE table used by unsupported operation");
      auto* rows = input ? input : ids;
      if (!rows || (input && ids && input != ids) || rows->type != GGML_TYPE_I32 ||
          rows->ne[0] <= 0 || rows->ne[1] != 1 || rows->ne[2] != 1 || rows->ne[3] != 1)
        fail("PLE requires a one-dimensional global row-ID tensor");
      ggml_tensor* args[] = {rows};
      return ggml_custom_4d(ctx, GGML_TYPE_F32, source.shape[0], rows->ne[0], 1, 1,
                           args, 1, Impl::compute, 1, &source);
    }
    if (op != GGML_OP_MUL_MAT_ID || !input || !ids || input->type != GGML_TYPE_F32 ||
        ids->type != GGML_TYPE_I32 || input->ne[0] != source.shape[0] ||
        input->ne[1] <= 0 || input->ne[2] <= 0 || input->ne[3] != 1 ||
        ids->ne[0] <= 0 || ids->ne[1] != input->ne[2] || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        ids->ne[0] % input->ne[1] != 0)
      fail("unsupported expert operation, activation shape/type, or routing-ID broadcast");
    ggml_tensor* args[] = {input, ids};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, source.shape[1], ids->ne[0], input->ne[2], 1,
                         args, 2, Impl::compute, 1, &source);
  } catch (...) {
    impl.latch();
    return nullptr;
  }
}
#endif

bool PagedQwenGraph::abort(void* userdata) noexcept {
  return userdata && static_cast<PagedQwenGraph*>(userdata)->impl_->failed.load(std::memory_order_acquire);
}

void PagedQwenGraph::throw_if_failed() const {
  if (!impl_->failed.load(std::memory_order_acquire)) return;
  std::exception_ptr error;
  { std::lock_guard<std::mutex> lock(impl_->mutex); error = impl_->failure; }
  if (error) std::rethrow_exception(error);
  fail("callback failed without a recoverable exception description");
}

PagedGraphStats PagedQwenGraph::stats() const {
  // Match project()'s lock order so logical expert requests and executor
  // counters describe the same completed/failed call boundary.
  std::lock_guard<std::mutex> projection_lock(impl_->projection_mutex);
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto result = impl_->counters;
  if (impl_->execution_options.projection_executor) {
    result.device_projection = impl_->execution_options.projection_executor->stats();
  }
  return result;
}

std::vector<PagedGraphRetainedExperts> PagedQwenGraph::retained_experts() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  std::vector<PagedGraphRetainedExperts> result;
#ifdef PMI_WITH_LLAMA
  for (const auto& item : impl_->layers) {
    if (!item.second->committed_ids.empty()) result.push_back({item.first, item.second->committed_ids});
  }
#endif
  return result;
}

std::vector<PagedGraphEvent> PagedQwenGraph::trace() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->events;
}

void PagedQwenGraph::clear_trace() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->events.clear();
  impl_->counters.trace_dropped = 0;
}

void PagedQwenGraph::finish_device_seed() {
  std::lock_guard<std::mutex> ordered(impl_->projection_mutex);
  impl_->device_seed_finished = true;
}

} // namespace pmi
