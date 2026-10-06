#pragma once

#include "pmi/manifest.hpp"
#include "pmi/page_provider.hpp"
#include "pmi/device_projection.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef PMI_WITH_LLAMA
#include "ggml.h"
#endif

namespace pmi {

struct PagedGraphTraceOptions {
  bool enabled{false};
  size_t max_events{4096};
  size_t max_output_values_per_event{4096};
};

struct PagedGraphExecutionOptions {
  uint32_t retain_experts_per_layer{}; // 0 is baseline LRU; supported range is 0..4.
  bool profile{false};
  // Explicit opt-in. Use a dedicated executor for this graph's ordered stream;
  // its compact projection cache is independent of the host page cache.
  // Host previous-top4 retention cannot be combined with this initial mode.
  std::shared_ptr<ProjectionExecutor> projection_executor;
  bool split_cpu_misses{false}; // Frozen first-evaluation device seed; grouped hit worker + CPU misses.
};

struct PagedGraphRetainedExperts {
  int layer{-1};
  std::vector<int32_t> expert_ids; // Committed final-token top-k order after down completes.
};

struct PagedGraphEvent {
  std::string tensor_name;
  std::string operation;
  int64_t token{};
  int64_t slot{};
  int64_t id{}; // Global expert ID or global PLE row ID, never a cache index.
  std::vector<float> output;
  bool output_truncated{};
};

struct PagedGraphStats {
  uint64_t candidate_weight_bytes{}; // Validated manifest sources, including optional blocks.
  uint64_t deferred_weight_bytes{};
  uint64_t largest_page_bytes{};
  uint64_t page_requests{}; // Logical selected pages, including device-cache hits.
  uint64_t expert_requests{};
  uint64_t lookup_requests{};
  uint64_t executed_nodes{};
  uint64_t trace_dropped{};
  uint64_t retention_requested_experts{};
  uint64_t retention_effective_experts{}; // Planned on the first expert operation, when routed count is known.
  uint64_t retention_protected_bytes{};
  uint64_t retention_peak_protected_bytes{};
  uint64_t retention_service_reserve_bytes{};
  uint64_t retention_fallbacks{};
  uint64_t retention_transitions{};
  bool profile_enabled{};
  // Provider total equals expert_request_ms + lookup_request_ms; do not add the
  // category total to its subdivisions. Other named phases are disjoint.
  double provider_request_ms{}, activation_copy_ms{}, graph_setup_ms{}, kernel_ms{};
  double output_copy_ms{}, graph_teardown_ms{}, retention_ms{};
  double expert_request_ms{}, lookup_request_ms{}, lookup_decode_ms{};
  // Provider phases/counters cover direct host requests in this connector.
  // Executor host misses/uploads and projection phases are reported separately
  // in device_projection; logical expert_requests still includes device hits.
  uint64_t profile_projection_calls{}, profile_lookup_rows{}, profile_provider_requests{};
  uint64_t profile_expert_requests{}, profile_lookup_requests{};
  bool device_projection_enabled{};
  bool split_enabled{};
  uint64_t split_groups{}, split_cpu_projections{}, split_device_projections{};
  // Group wall time, independent CPU/device branch wall times and residual
  // join wait. Branches overlap; do not sum them as end-to-end wall time.
  double split_critical_ms{}, split_cpu_ms{}, split_device_ms{}, split_join_ms{}, split_overlap_ms{};
  ProjectionStats device_projection{};
};

// Qwen-specific graph boundary. Dense weights and graph state remain in llama;
// the generic FilePageProvider owns immutable shard storage, cache and leases.
// This object and its provider must outlive the model AND decoder contexts.
// Descriptors survive graph rebuilds; no graph tensor pointer is retained.
// Retention/execution supports one ordered decoder evaluation stream per
// instance. Executor calls and stats are serialized; the executor must not be
// independently used while attached. Destruction clears its unleased pages.
class PagedQwenGraph final {
 public:
  PagedQwenGraph(const ModelManifest&, FilePageProvider&, PagedGraphTraceOptions = {},
                 PagedGraphExecutionOptions = {});
  ~PagedQwenGraph();
  PagedQwenGraph(const PagedQwenGraph&) = delete;
  PagedQwenGraph& operator=(const PagedQwenGraph&) = delete;

#ifdef PMI_WITH_LLAMA
  static bool select(const ggml_tensor* metadata, void* userdata) noexcept;
  // MUL_MAT_ID takes activation=input and global expert IDs=ids.
  // GET_ROWS takes global row IDs=input and ids=nullptr (or vice versa).
  // nullptr means a latched failure, not permission to use a resident fallback.
  static ggml_tensor* build(ggml_context*, ggml_op, const ggml_tensor* weight,
                            ggml_tensor* input, ggml_tensor* ids, void* userdata) noexcept;
#endif
  static bool abort(void* userdata) noexcept;
  void throw_if_failed() const;
  PagedGraphStats stats() const;
  std::vector<PagedGraphRetainedExperts> retained_experts() const;
  std::vector<PagedGraphEvent> trace() const;
  void clear_trace();
  // Called only between decoder evaluations; freezes admission after evaluation0.
  void finish_device_seed();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace pmi
