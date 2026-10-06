#pragma once
#include "pmi/page_provider.hpp"
#include "pmi/paged_graph.hpp"
#include "pmi/telemetry.hpp"
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>
namespace pmi {
enum class TraceMode { Full, Compact };
enum class ReferenceMode { Default, Demand };
enum class CachePolicy { Lru, PreviousTop4 };
enum class ProjectionBackend { Cpu, DeviceCpu, DeviceVulkan };
enum class DenseLoadPolicy { Default, DirectRequired };
struct DenseLoadShardMetrics {
  uint64_t shard_index{}; // Canonical GGUF split order; no filesystem paths.
  bool direct_io_requested{}, direct_io_required{}, direct_io_active{};
  uint64_t payload_requested_bytes{}, payload_completed_bytes{};
  uint64_t direct_read_bytes{}, buffered_read_bytes{}, staging_peak_bytes{}, fallback_count{};
};
struct DenseLoadMetrics {
  DenseLoadPolicy requested{DenseLoadPolicy::Default};
  std::string effective;
  uint64_t payload_requested_bytes{}, payload_completed_bytes{};
  uint64_t direct_read_bytes{}, buffered_read_bytes{}, staging_peak_bytes{}, fallback_count{};
  std::vector<DenseLoadShardMetrics> shards;
};
struct ProjectionDeviceIdentity {
  std::string backend_name, name, description, type, device_id;
  std::optional<uint32_t> index;
};
struct GenerationOptions {
  int32_t max_tokens{32};
  // Benchmark replay: emitted IDs are supplied, while greedy predictions are
  // recorded separately. Empty preserves ordinary generation/EOG behavior.
  std::vector<int32_t> forced_tokens;
  int32_t gpu_layers{};
  uint32_t prefill_chunk{1};
  uint32_t context_length{};
  size_t cache_bytes{64 * 1024 * 1024};
  std::filesystem::path trace_json;
  std::filesystem::path routing_jsonl;
  CachePolicy cache_policy{CachePolicy::Lru};
  bool profile_paging{false};
  ProjectionBackend projection_backend{ProjectionBackend::Cpu};
  bool split_cpu_misses{false};
  ProjectionActivationMode projection_activation{ProjectionActivationMode::BackendNative};
  DenseLoadPolicy dense_load{DenseLoadPolicy::Default};
  bool dense_base_pages{false};
  std::filesystem::path load_trace_jsonl;
  std::filesystem::path progress_jsonl;
  std::optional<uint32_t> projection_device_index;
  bool transient_upload_host{};
  HostReadPolicy upload_host_io{HostReadPolicy::Buffered};
  uint64_t device_cache_bytes{};
  uint64_t device_graph_bytes{};
  TraceMode trace_mode{TraceMode::Full};
  ReferenceMode reference_mode{ReferenceMode::Default};
  bool cpu_repack{true};
};
struct EvaluationMetrics {
  uint64_t evaluation{}, input_tokens{};
  bool prefill{};
  double elapsed_ms{};
  uint64_t io_bytes{}, cache_hits{}, cache_misses{}, evictions{};
  std::optional<uint64_t> process_storage_read_bytes_delta;
  ProcessMemory memory;
  PagedGraphStats paging;
};
struct GenerationMetrics {
  double elapsed_ms{}, setup_ms{}, prefill_ms{}, decode_ms{};
  // Null when generation emits no token (for example, an immediate EOG).
  std::optional<double> ttft_ms, first_token_after_setup_ms;
  uint64_t prompt_tokens{};
  uint64_t prefill_evaluations{}, decode_evaluations{};
  uint64_t page_requests{}, expert_requests{}, lookup_requests{}, deferred_weight_bytes{};
  RuntimeStats cache{};
  IoStats provider_io{};
  ProcessMemory memory;
  bool paged{};
  bool teacher_forcing{};
  uint64_t prediction_matches{};
  bool cpu_repack{true};
  CachePolicy cache_policy{CachePolicy::Lru};
  ProjectionBackend projection_backend{ProjectionBackend::Cpu};
  ProjectionActivationMode projection_activation{ProjectionActivationMode::BackendNative};
  bool transient_upload_host{};
  HostReadPolicy upload_host_io{HostReadPolicy::Buffered};
  std::optional<ProjectionDeviceIdentity> projection_device;
  // Dense loader read_raw counters only: excludes metadata and PMI page reads.
  // Absent for the native reference, whose tensor payload scope is different.
  std::optional<DenseLoadMetrics> dense_load;
  bool dense_base_pages{}, dense_load_observed{};
  uint64_t dense_owned_buffers{}, dense_advised_buffers{}, dense_advised_bytes{};
  bool routing_observation{};
  PagedGraphStats paging;
  TraceMode trace_mode{TraceMode::Full};
  ReferenceMode reference_mode{ReferenceMode::Default};
  std::vector<EvaluationMetrics> evaluations;
};
struct GenerationResult {
  std::vector<int32_t> tokens;
  std::vector<int32_t> predictions;
  std::string text;
  GenerationMetrics metrics;
};
GenerationResult run_reference(const std::string& model_path, const std::string& prompt,
                               int32_t max_tokens, int32_t gpu_layers);
GenerationResult run_reference(const std::string& model_path, const std::string& prompt,
                               const GenerationOptions& options);
GenerationResult run_paged(const std::vector<std::filesystem::path>& shards,
                          const std::string& prompt, const GenerationOptions& options);
std::string generation_metrics_json(const GenerationResult& result, bool include_evaluations = true);
}
