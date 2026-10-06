#include "pmi/inference.hpp"
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>

#ifdef PMI_WITH_LLAMA
#include "pmi/manifest.hpp"
#include "pmi/owned_buffer.hpp"
#include <cstdio>
#include <exception>
#include "pmi/paged_graph.hpp"
#include "pmi/routing_trace.hpp"
#include "pmi/device_projection.hpp"
#include "pmi/progress.hpp"
#include "pmi/ggml_device.hpp"
#include "ggml-cpu.h"
#ifdef PMI_WITH_VULKAN
#include "ggml-vulkan.h"
#endif
#include "llama.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <map>
#endif

namespace pmi {
namespace {
std::string quoted(const std::string& text) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : text) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
    else out << c;
  }
  out << '"'; return out.str();
}
const char* projection_backend_json(ProjectionBackend backend) {
  switch (backend) {
    case ProjectionBackend::Cpu: return "\"cpu\"";
    case ProjectionBackend::DeviceCpu: return "\"device-cpu\"";
    case ProjectionBackend::DeviceVulkan: return "\"device-vulkan\"";
  }
  return "null";
}
const char* projection_activation_json(ProjectionActivationMode mode) {
  switch (mode) {
    case ProjectionActivationMode::BackendNative: return "\"backend-native\"";
    case ProjectionActivationMode::CpuCompatible: return "\"cpu-compatible\"";
  }
  return "null";
}
void projection_device_json(std::ostream& out, const std::optional<ProjectionDeviceIdentity>& device) {
  if (!device) { out << "null"; return; }
  out << "{\"backend_name\":" << quoted(device->backend_name) << ",\"name\":" << quoted(device->name)
      << ",\"description\":" << quoted(device->description) << ",\"type\":" << quoted(device->type)
      << ",\"device_id\":";
  if (device->device_id.empty()) out << "null"; else out << quoted(device->device_id);
  out << ",\"index\":";
  if (device->index) out << *device->index; else out << "null";
  out << '}';
}
void dense_load_json(std::ostream& out, const std::optional<DenseLoadMetrics>& value) {
  if (!value) { out << "null"; return; }
  const auto& m = *value;
  out << "{\"requested\":\"" << (m.requested == DenseLoadPolicy::DirectRequired ? "direct-required" : "default")
      << "\",\"effective\":" << quoted(m.effective)
      << ",\"scope\":\"llama_dense_tensor_payload_reads_excludes_gguf_metadata_and_pmi_provider\""
      << ",\"actual_read_scope\":\"read_or_fread_returned_bytes_including_direct_alignment_overlap\""
      << ",\"staging_peak_scope\":\"max_single_aligned_read_allocation\"";
  auto counters = [&](const auto& s) {
#define PMI_LOAD_FIELD(name) out << ",\"" #name "\":" << s.name
    PMI_LOAD_FIELD(payload_requested_bytes); PMI_LOAD_FIELD(payload_completed_bytes);
    PMI_LOAD_FIELD(direct_read_bytes); PMI_LOAD_FIELD(buffered_read_bytes);
    PMI_LOAD_FIELD(staging_peak_bytes); PMI_LOAD_FIELD(fallback_count);
#undef PMI_LOAD_FIELD
  };
  counters(m);
  out << ",\"shards\":[";
  for (size_t i = 0; i < m.shards.size(); ++i) {
    const auto& s = m.shards[i];
    if (i) out << ',';
    out << "{\"shard_index\":" << s.shard_index
        << ",\"direct_io_requested\":" << (s.direct_io_requested ? "true" : "false")
        << ",\"direct_io_required\":" << (s.direct_io_required ? "true" : "false")
        << ",\"direct_io_active\":" << (s.direct_io_active ? "true" : "false");
    counters(s);
    out << '}';
  }
  out << "]}";
}
void device_projection_fields(std::ostream& out, bool enabled, const ProjectionStats& s) {
  out << ",\"device_projection\":{\"enabled\":" << (enabled ? "true" : "false")
      << ",\"managed_bytes_scope\":\"ggml_buffer_reported_allocations_excluding_driver_staging_scratch\""
      << ",\"activation_preparation_peak_scope\":\"host_packed_and_expanded_activation_vectors_excludes_allocator_overhead\""
      << ",\"profile_enabled\":" << (s.profile_enabled ? "true" : "false");
#define PMI_DEVICE_FIELD(name) out << ",\"" #name "\":" << s.name
  PMI_DEVICE_FIELD(calls); PMI_DEVICE_FIELD(completed); PMI_DEVICE_FIELD(failures);
  PMI_DEVICE_FIELD(graph_current_bytes); PMI_DEVICE_FIELD(graph_peak_bytes); PMI_DEVICE_FIELD(graph_capacity_bytes);
  PMI_DEVICE_FIELD(activation_upload_bytes); PMI_DEVICE_FIELD(output_readback_bytes);
  PMI_DEVICE_FIELD(activation_preparation_calls); PMI_DEVICE_FIELD(activation_preparation_bytes);
  PMI_DEVICE_FIELD(activation_preparation_peak_bytes); PMI_DEVICE_FIELD(activation_preparation_ms);
  PMI_DEVICE_FIELD(cache_request_ms); PMI_DEVICE_FIELD(graph_setup_ms); PMI_DEVICE_FIELD(activation_upload_ms);
  PMI_DEVICE_FIELD(kernel_ms); PMI_DEVICE_FIELD(output_readback_ms); PMI_DEVICE_FIELD(graph_teardown_ms);
#undef PMI_DEVICE_FIELD
  out << ",\"cache\":{\"requests\":" << s.cache.requests;
  out << ",\"profile_enabled\":" << (s.cache.profile_enabled ? "true" : "false");
#define PMI_DEVICE_CACHE_FIELD(name) out << ",\"" #name "\":" << s.cache.name
  PMI_DEVICE_CACHE_FIELD(provider_read_ms); PMI_DEVICE_CACHE_FIELD(allocation_ms);
  PMI_DEVICE_CACHE_FIELD(upload_ms); PMI_DEVICE_CACHE_FIELD(eviction_ms);
  PMI_DEVICE_CACHE_FIELD(requested_bytes); PMI_DEVICE_CACHE_FIELD(hits); PMI_DEVICE_CACHE_FIELD(misses);
  PMI_DEVICE_CACHE_FIELD(failures); PMI_DEVICE_CACHE_FIELD(provider_requests);
  PMI_DEVICE_CACHE_FIELD(upload_bytes); PMI_DEVICE_CACHE_FIELD(uploads); PMI_DEVICE_CACHE_FIELD(evictions);
  PMI_DEVICE_CACHE_FIELD(current_bytes); PMI_DEVICE_CACHE_FIELD(peak_bytes); PMI_DEVICE_CACHE_FIELD(capacity_bytes);
  PMI_DEVICE_CACHE_FIELD(entries); PMI_DEVICE_CACHE_FIELD(active_leases);
#undef PMI_DEVICE_CACHE_FIELD
  out << "}}";
}
void paging_fields(std::ostream& out, const PagedGraphStats& s) {
#define PMI_FIELD(name) out << ",\"" #name "\":" << s.name
  out << ",\"split_enabled\":" << (s.split_enabled ? "true" : "false");
  PMI_FIELD(split_groups); PMI_FIELD(split_cpu_projections); PMI_FIELD(split_device_projections);
  PMI_FIELD(split_critical_ms); PMI_FIELD(split_cpu_ms); PMI_FIELD(split_device_ms); PMI_FIELD(split_join_ms); PMI_FIELD(split_overlap_ms);
  PMI_FIELD(retention_requested_experts); PMI_FIELD(retention_effective_experts);
  PMI_FIELD(retention_protected_bytes); PMI_FIELD(retention_peak_protected_bytes);
  PMI_FIELD(retention_service_reserve_bytes); PMI_FIELD(retention_fallbacks); PMI_FIELD(retention_transitions);
  out << ",\"profile_enabled\":" << (s.profile_enabled ? "true" : "false");
  PMI_FIELD(provider_request_ms); PMI_FIELD(expert_request_ms); PMI_FIELD(lookup_request_ms);
  PMI_FIELD(activation_copy_ms); PMI_FIELD(graph_setup_ms); PMI_FIELD(kernel_ms);
  PMI_FIELD(output_copy_ms); PMI_FIELD(lookup_decode_ms); PMI_FIELD(graph_teardown_ms); PMI_FIELD(retention_ms);
  PMI_FIELD(profile_projection_calls); PMI_FIELD(profile_lookup_rows); PMI_FIELD(profile_provider_requests);
  PMI_FIELD(profile_expert_requests); PMI_FIELD(profile_lookup_requests);
#undef PMI_FIELD
  device_projection_fields(out, s.device_projection_enabled, s.device_projection);
}
void evaluation_json(std::ostream& out, const EvaluationMetrics& e) {
  auto optional = [&](const char* name, const auto& value) {
    out << ",\"" << name << "\":";
    if (value) out << *value; else out << "null";
  };
  out << "{\"evaluation\":" << e.evaluation << ",\"phase\":\"" << (e.prefill ? "prefill" : "decode")
      << "\",\"input_tokens\":" << e.input_tokens << ",\"elapsed_ms\":" << e.elapsed_ms
      << ",\"io_bytes\":" << e.io_bytes << ",\"cache_hits\":" << e.cache_hits
      << ",\"cache_misses\":" << e.cache_misses << ",\"evictions\":" << e.evictions
      << ",\"hit_rate\":" << (e.cache_hits + e.cache_misses ? double(e.cache_hits) / (e.cache_hits + e.cache_misses) : 0);
  optional("process_storage_read_bytes_delta", e.process_storage_read_bytes_delta);
  optional("rss_bytes", e.memory.rss_bytes);
  optional("peak_rss_bytes", e.memory.peak_rss_bytes);
  paging_fields(out, e.paging);
  out << '}';
}
#ifdef PMI_WITH_LLAMA
PagedGraphStats paging_delta(PagedGraphStats after, const PagedGraphStats& before) {
#define PMI_DELTA(name) after.name -= before.name
  PMI_DELTA(page_requests); PMI_DELTA(expert_requests); PMI_DELTA(lookup_requests);
  PMI_DELTA(executed_nodes); PMI_DELTA(trace_dropped);
  PMI_DELTA(retention_fallbacks); PMI_DELTA(retention_transitions);
  PMI_DELTA(provider_request_ms); PMI_DELTA(expert_request_ms); PMI_DELTA(lookup_request_ms);
  PMI_DELTA(activation_copy_ms); PMI_DELTA(graph_setup_ms); PMI_DELTA(kernel_ms);
  PMI_DELTA(output_copy_ms); PMI_DELTA(lookup_decode_ms); PMI_DELTA(graph_teardown_ms); PMI_DELTA(retention_ms);
  PMI_DELTA(profile_projection_calls); PMI_DELTA(profile_lookup_rows); PMI_DELTA(profile_provider_requests);
  PMI_DELTA(profile_expert_requests); PMI_DELTA(profile_lookup_requests);
  PMI_DELTA(split_groups); PMI_DELTA(split_cpu_projections); PMI_DELTA(split_device_projections);
  PMI_DELTA(split_critical_ms); PMI_DELTA(split_cpu_ms); PMI_DELTA(split_device_ms); PMI_DELTA(split_join_ms); PMI_DELTA(split_overlap_ms);
#undef PMI_DELTA
  auto& device = after.device_projection;
  const auto& previous = before.device_projection;
#define PMI_DEVICE_DELTA(name) device.name -= previous.name
  PMI_DEVICE_DELTA(calls); PMI_DEVICE_DELTA(completed); PMI_DEVICE_DELTA(failures);
  PMI_DEVICE_DELTA(activation_upload_bytes); PMI_DEVICE_DELTA(output_readback_bytes);
  PMI_DEVICE_DELTA(activation_preparation_calls); PMI_DEVICE_DELTA(activation_preparation_bytes);
  PMI_DEVICE_DELTA(activation_preparation_ms);
  PMI_DEVICE_DELTA(cache_request_ms); PMI_DEVICE_DELTA(graph_setup_ms); PMI_DEVICE_DELTA(activation_upload_ms);
  PMI_DEVICE_DELTA(kernel_ms); PMI_DEVICE_DELTA(output_readback_ms); PMI_DEVICE_DELTA(graph_teardown_ms);
  PMI_DEVICE_DELTA(cache.requests); PMI_DEVICE_DELTA(cache.requested_bytes);
  PMI_DEVICE_DELTA(cache.hits); PMI_DEVICE_DELTA(cache.misses); PMI_DEVICE_DELTA(cache.failures);
  PMI_DEVICE_DELTA(cache.provider_requests); PMI_DEVICE_DELTA(cache.upload_bytes);
  PMI_DEVICE_DELTA(cache.uploads); PMI_DEVICE_DELTA(cache.evictions);
  PMI_DEVICE_DELTA(cache.provider_read_ms); PMI_DEVICE_DELTA(cache.allocation_ms);
  PMI_DEVICE_DELTA(cache.upload_ms); PMI_DEVICE_DELTA(cache.eviction_ms);
#undef PMI_DEVICE_DELTA
  return after; // Quota, reserved bytes and lifetime peak remain gauges.
}
#endif
}
const char* host_io_name(HostReadPolicy policy) {
  switch(policy) {
    case HostReadPolicy::Buffered:return "buffered";
    case HostReadPolicy::DirectPreferred:return "direct-preferred";
    case HostReadPolicy::DirectRequired:return "direct-required";
  }
  return "unknown";
}
void provider_io_json(std::ostream& out,const IoStats& s) {
  out << "{\"scope\":\"provider_read_api_bytes_excludes_dense_metadata_kernel_internal_fallback_unknown\""
      << ",\"direct_requests\":" << s.direct_requests << ",\"direct_completed\":" << s.direct_completed
      << ",\"fallback_count\":" << s.fallback_count << ",\"direct_read_bytes\":" << s.direct_read_bytes
      << ",\"buffered_read_bytes\":" << s.buffered_read_bytes << ",\"direct_logical_bytes\":" << s.direct_logical_bytes
      << ",\"staging_peak_bytes\":" << s.staging_peak_bytes << ",\"alignment_statx_requests\":" << s.alignment_statx_requests
      << ",\"alignment_known_fs_requests\":" << s.alignment_known_fs_requests << "}";
}
std::string generation_metrics_json(const GenerationResult& result, bool include_evaluations) {
  const auto& m = result.metrics;
  std::ostringstream out;
  out.imbue(std::locale::classic()); out << std::setprecision(12);
  auto optional = [&](const char* name, const auto& value) {
    out << ",\"" << name << "\":";
    if (value) out << *value; else out << "null";
  };
  out << "{\"path\":\"" << (!m.paged ? "llama_reference" :
          m.projection_backend == ProjectionBackend::DeviceVulkan ? "pmi_paged_qwen_vulkan_projections" : "pmi_paged_qwen_cpu")
      << "\",\"elapsed_ms\":" << m.elapsed_ms << ",\"setup_ms\":" << m.setup_ms
      << ",\"prefill_ms\":" << m.prefill_ms << ",\"decode_ms\":" << m.decode_ms;
  optional("ttft_ms", m.ttft_ms);
  optional("first_token_after_setup_ms", m.first_token_after_setup_ms);
  out << ",\"prompt_tokens\":" << m.prompt_tokens
      << ",\"trace_mode\":\"" << (m.trace_mode == TraceMode::Compact ? "compact" : "full") << '"'
      << ",\"cpu_repack\":" << (m.cpu_repack ? "true" : "false")
      << ",\"teacher_forcing\":" << (m.teacher_forcing ? "true" : "false")
      << ",\"prediction_matches\":" << (m.teacher_forcing ? std::to_string(m.prediction_matches) : "null")
      << ",\"cache_policy\":\"" << (m.cache_policy == CachePolicy::PreviousTop4 ? "previous-top4" : "lru") << '"'
      << ",\"projection_backend\":" << (m.paged ? projection_backend_json(m.projection_backend) : "null")
      << ",\"device_upload_host_policy\":\"" << (m.transient_upload_host ? "transient" : "retained") << "\""
      << ",\"projection_activation\":" << (m.paged ? projection_activation_json(m.projection_activation) : "null")
      << ",\"projection_device\":";
  projection_device_json(out, m.projection_device);
  out << ",\"device_upload_host_io\":\"" << host_io_name(m.upload_host_io) << "\",\"provider_io\":";
  if(m.paged)provider_io_json(out,m.provider_io);else out << "null";
  out << ",\"dense_load\":";
  dense_load_json(out, m.dense_load);
  out << ",\"dense_pages\":{\"base_requested\":" << (m.dense_base_pages ? "true" : "false")
      << ",\"observed\":" << (m.dense_load_observed ? "true" : "false")
      << ",\"owned_buffers\":" << m.dense_owned_buffers << ",\"advised_buffers\":" << m.dense_advised_buffers
      << ",\"advised_bytes\":" << m.dense_advised_bytes << "}";
  out << ",\"routing_observation\":" << (m.routing_observation ? "true" : "false")
      << ",\"reference_mode\":";
  if (m.paged) out << "null";
  else out << '"' << (m.reference_mode == ReferenceMode::Demand ? "demand" : "default") << '"';
  out
      << ",\"generated_tokens\":" << result.tokens.size()
      << ",\"prefill_evaluations\":" << m.prefill_evaluations
      << ",\"decode_evaluations\":" << m.decode_evaluations
      << ",\"decode_evaluations_per_second\":" << (m.decode_ms > 0 ? m.decode_evaluations * 1000.0 / m.decode_ms : 0)
      << ",\"page_requests\":" << m.page_requests << ",\"expert_requests\":" << m.expert_requests
      << ",\"lookup_requests\":" << m.lookup_requests << ",\"deferred_weight_bytes\":" << m.deferred_weight_bytes
      << ",\"io_bytes\":" << m.cache.io_bytes << ",\"cache_hits\":" << m.cache.hits
      << ",\"cache_misses\":" << m.cache.misses << ",\"evictions\":" << m.cache.evictions
      << ",\"cache_reserved_bytes\":" << m.cache.current_bytes
      << ",\"cache_peak_reserved_bytes\":" << m.cache.peak_bytes
      << ",\"cache_capacity_bytes\":" << m.cache.capacity_bytes
      << ",\"hit_rate\":" << (m.cache.hits + m.cache.misses ? double(m.cache.hits) / (m.cache.hits + m.cache.misses) : 0);
  optional("rss_bytes", m.memory.rss_bytes);
  optional("peak_rss_bytes", m.memory.peak_rss_bytes);
  optional("process_storage_read_bytes", m.memory.storage_read_bytes);
  if (m.paged) {
    if (m.projection_backend == ProjectionBackend::DeviceVulkan)
      out << ",\"vram_bytes\":null,\"vram_bytes_scope\":\"physical_device_usage_not_sampled\"";
    else out << ",\"vram_bytes\":0,\"vram_bytes_scope\":\"pmi_gpu_allocations_none\"";
  }
  paging_fields(out, m.paging);
  if (include_evaluations) {
    out << ",\"evaluations\":[";
    for (size_t i = 0; i < m.evaluations.size(); ++i) {
      if (i) out << ',';
      evaluation_json(out, m.evaluations[i]);
    }
    out << ']';
  }
  out << '}'; return out.str();
}

#ifdef PMI_WITH_LLAMA
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
class DenseLoadObserver final {
 public:
  DenseLoadObserver(const GenerationOptions& options, GenerationMetrics& metrics)
      : base_(options.dense_base_pages), metrics_(metrics) {
    metrics_.dense_base_pages = base_;
    metrics_.dense_load_observed = true;
    if (!options.load_trace_jsonl.empty()) {
      file_.reset(std::fopen(options.load_trace_jsonl.c_str(), "wx"));
      if (!file_) throw std::system_error(errno, std::generic_category(), "create exclusive loader trace");
      write(std::string("{\"schema\":1,\"scope\":\"dense_cpu_load\",\"base_pages\":") + (base_ ? "true" : "false") + "}");
    }
  }
  llama_load_observer binding() { return {callback, this}; }
  void check() const { if (failure_) std::rethrow_exception(failure_); }
 private:
  bool base_;
  GenerationMetrics& metrics_;
  std::unique_ptr<FILE, decltype(&std::fclose)> file_{nullptr, std::fclose};
  std::exception_ptr failure_;
  uint64_t records_{}, bytes_{};
  void write(const std::string& line) {
    if (++records_ > 65536 || line.size() + 1 > 16 * 1024 * 1024 - bytes_)
      throw std::runtime_error("loader trace exceeded bounded record/byte quota");
    bytes_ += line.size() + 1;
    if (std::fwrite(line.data(), 1, line.size(), file_.get()) != line.size() ||
        std::fputc('\n', file_.get()) == EOF || std::fflush(file_.get()))
      throw std::runtime_error("loader trace write failed");
  }
  static bool callback(const llama_load_event* event, void* data) noexcept {
    auto& self = *static_cast<DenseLoadObserver*>(data);
    try { self.event(*event); return true; }
    catch (...) { if (!self.failure_) self.failure_ = std::current_exception(); return false; }
  }
  void event(const llama_load_event& e) {
    OwnedPageRange advised;
    if (e.kind == LLAMA_LOAD_BUFFER_READY) {
      if (!e.owned_cpu_buffer && base_)
        throw std::runtime_error("base pages require a single owned CPU weight buffer");
      if (e.owned_cpu_buffer) {
        ++metrics_.dense_owned_buffers;
#ifdef __linux__
        if (base_) advised = advise_owned_base_pages(e.owned_cpu_buffer, e.owned_cpu_bytes);
#endif
        if (advised.bytes) ++metrics_.dense_advised_buffers;
        metrics_.dense_advised_bytes += advised.bytes;
      }
    }
    if (!file_) return;
    const char* phases[] = {"buffer_begin", "buffer_ready", "tensor_begin", "tensor_end", "read_begin", "read_end", "copy_end"};
    if (e.kind < 0 || static_cast<size_t>(e.kind) >= sizeof(phases)/sizeof(phases[0]))
      throw std::runtime_error("unknown loader event");
    std::ostringstream line;
    line << "{\"phase\":\"" << phases[e.kind] << "\",\"monotonic_ns\":" << e.monotonic_ns
         << ",\"duration_ns\":" << e.duration_ns << ",\"shard_index\":" << e.shard_index
         << ",\"tensor\":" << (e.tensor_name ? quoted(e.tensor_name) : "null")
         << ",\"offset\":" << e.offset << ",\"requested_bytes\":" << e.requested_bytes
         << ",\"completed_bytes\":" << e.completed_bytes << ",\"owned_begin\":" << reinterpret_cast<uintptr_t>(e.owned_cpu_buffer)
         << ",\"owned_bytes\":" << e.owned_cpu_bytes << ",\"advised_begin\":" << advised.begin
         << ",\"advised_bytes\":" << advised.bytes << "}";
    write(line.str());
  }
};

DenseLoadMetrics dense_load_metrics(const llama_model* model, DenseLoadPolicy policy, size_t shard_count) {
  DenseLoadMetrics result;
  result.requested = policy;
  const auto count = llama_model_load_io_count(model);
  if (count != shard_count || !count) throw std::runtime_error("dense loader shard telemetry is incomplete");
  size_t active = 0;
  auto accumulate = [](uint64_t& total, uint64_t value) {
    if (value > UINT64_MAX - total) throw std::overflow_error("dense loader telemetry overflow");
    total += value;
  };
  for (size_t i = 0; i < count; ++i) {
    llama_model_load_io_info raw{};
    if (!llama_model_get_load_io_info(model, i, &raw))
      throw std::runtime_error("dense loader shard telemetry is unavailable");
    if (raw.payload_requested_bytes != raw.payload_completed_bytes)
      throw std::runtime_error("dense loader payload was not completely read");
    if (policy == DenseLoadPolicy::DirectRequired &&
        (!raw.direct_io_requested || !raw.direct_io_required || !raw.direct_io_active ||
         raw.buffered_read_bytes || raw.fallback_count))
      throw std::runtime_error("required direct I/O was not active on every dense loader shard");
    DenseLoadShardMetrics s;
    s.shard_index = i;
    s.direct_io_requested = raw.direct_io_requested;
    s.direct_io_required = raw.direct_io_required;
    s.direct_io_active = raw.direct_io_active;
    active += s.direct_io_active;
#define PMI_LOAD_COPY(name) s.name = raw.name; accumulate(result.name, s.name)
    PMI_LOAD_COPY(payload_requested_bytes); PMI_LOAD_COPY(payload_completed_bytes);
    PMI_LOAD_COPY(direct_read_bytes); PMI_LOAD_COPY(buffered_read_bytes); PMI_LOAD_COPY(fallback_count);
#undef PMI_LOAD_COPY
    s.staging_peak_bytes = raw.staging_peak_bytes;
    result.staging_peak_bytes = std::max(result.staging_peak_bytes, s.staging_peak_bytes);
    result.shards.push_back(s);
  }
  result.effective = policy == DenseLoadPolicy::DirectRequired ? "direct-required" :
                     active == count ? "direct" : active ? "mixed" : "buffered";
  return result;
}
ProjectionDeviceIdentity selected_device(ggml_backend_t backend, std::optional<uint32_t> index) {
  const auto device = ggml_backend_get_device(backend);
  if (!device) throw std::runtime_error("selected projection backend has no device identity");
  ProjectionDeviceIdentity identity;
  identity.backend_name = ggml_backend_name(backend);
  identity.name = ggml_backend_dev_name(device);
  identity.description = ggml_backend_dev_description(device);
  ggml_backend_dev_props properties{};
  ggml_backend_dev_get_props(device, &properties);
  if (properties.device_id) identity.device_id = properties.device_id;
  identity.index = index;
  switch (ggml_backend_dev_type(device)) {
    case GGML_BACKEND_DEVICE_TYPE_CPU: identity.type = "cpu"; break;
    case GGML_BACKEND_DEVICE_TYPE_GPU: identity.type = "gpu"; break;
    case GGML_BACKEND_DEVICE_TYPE_IGPU: identity.type = "igpu"; break;
    case GGML_BACKEND_DEVICE_TYPE_ACCEL: identity.type = "accelerator"; break;
    case GGML_BACKEND_DEVICE_TYPE_META: identity.type = "meta"; break;
    default: identity.type = "unknown"; break;
  }
  return identity;
}
struct Snapshot {
  uint64_t evaluation{};
  bool prefill{};
  std::string name;
  std::array<int64_t, 4> shape;
  std::vector<float> floats;
  std::vector<int32_t> integers;
};
struct Execution {
  PagedQwenGraph* graph{};
  std::unique_ptr<RoutingTraceWriter> routing;
  std::map<int, Snapshot> pending_routes;
  uint64_t routing_pending_bytes{}, token_start{}, input_tokens{};
  int layer_count{};
  bool tracing{};
  TraceMode trace_mode{TraceMode::Full};
  uint64_t evaluation{}, trace_bytes{};
  bool prefill{};
  std::atomic<bool> failed{false};
  std::mutex mutex;
  std::string error;
  std::vector<Snapshot> snapshots;

  void fail(const char* message) noexcept {
    try { std::lock_guard<std::mutex> lock(mutex); if (!failed.load()) error = message; }
    catch (...) {}
    failed.store(true);
  }
  void check() {
    if (graph) graph->throw_if_failed();
    if (failed.load()) throw std::runtime_error(error.empty() ? "inference callback failed" : error);
  }
  static bool abort(void* data) noexcept {
    auto& self = *static_cast<Execution*>(data);
    return self.failed.load() || (self.graph && PagedQwenGraph::abort(self.graph));
  }
  static bool named(const char* name, const char* label) noexcept {
    const size_t length = std::strlen(label);
    if (std::strncmp(name, label, length)) return false;
    const char* suffix = name + length;
    if (!*suffix) return true;
    if (*suffix++ != '-' || !*suffix) return false;
    for (; *suffix; ++suffix) if (*suffix < '0' || *suffix > '9') return false;
    return true;
  }
  bool routing_selected(const ggml_tensor* tensor) const noexcept {
    return routing && routing->accepting() &&
        ((tensor->type == GGML_TYPE_I32 && named(ggml_get_name(tensor), "ffn_moe_topk")) ||
         (tensor->type == GGML_TYPE_F32 && named(ggml_get_name(tensor), "ffn_moe_weights_norm")));
  }
  void emit_route(int layer, const Snapshot& ids, const Snapshot* weights) {
    const auto slots = static_cast<size_t>(ids.shape[0]);
    const auto rows = static_cast<size_t>(ids.shape[1]);
    if (weights && weights->shape != ids.shape) throw std::runtime_error("routing weight shape mismatch");
    for (size_t column = 0; column < rows && routing->accepting(); ++column) {
      // This API submits one sequence and requests logits only for its final
      // input. Qwen prunes the last layer to that input before its routed FFN.
      const bool final_only = rows == 1 && input_tokens > 1 && layer == layer_count - 1;
      const uint64_t index = final_only ? input_tokens - 1 : column;
      const bool position_known = final_only || rows == input_tokens;
      std::ostringstream out;
      out.imbue(std::locale::classic()); out << std::setprecision(std::numeric_limits<float>::max_digits10);
      out << "\"sequence_id\":\"0\",\"eval_id\":" << evaluation
          << ",\"phase\":\"" << (prefill ? "prefill" : "decode") << "\",\"layer\":" << layer
          << ",\"token_index\":" << index << ",\"token_position\":";
      if (position_known) out << token_start + index; else out << "null";
      out << ",\"expert_ids\":[";
      for (size_t slot = 0; slot < slots; ++slot) { if (slot) out << ','; out << ids.integers[column * slots + slot]; }
      out << "],\"routing_weights\":";
      if (!weights) out << "null";
      else {
        out << '[';
        for (size_t slot = 0; slot < slots; ++slot) { if (slot) out << ','; out << weights->floats[column * slots + slot]; }
        out << ']';
      }
      out << ",\"complete\":true,\"truncated\":false";
      routing->append("routing", out.str());
    }
  }
  void capture_routing(const ggml_tensor* tensor) {
    const std::string name = ggml_get_name(tensor);
    const auto dash = name.rfind('-');
    if (dash == std::string::npos) throw std::runtime_error("routing tensor lacks layer identity");
    const auto layer_number = std::stoull(name.substr(dash + 1));
    if (layer_number >= static_cast<uint64_t>(layer_count)) throw std::runtime_error("routing layer is out of range");
    const int layer = static_cast<int>(layer_number);
    constexpr uint64_t limit = 1024 * 1024;
    const uint64_t storage = ggml_nbytes(tensor);
    const uint64_t elements = ggml_nelements(tensor);
    if (tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] != 1 || tensor->ne[3] != 1)
      throw std::runtime_error("unsupported routing observation layout");
    if (storage > limit || elements > limit / 4 || routing_pending_bytes > limit - elements * 4) {
      routing->stop_capture(); pending_routes.clear(); routing_pending_bytes = 0; return;
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(storage));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    Snapshot snapshot;
    std::copy_n(tensor->ne, 4, snapshot.shape.begin());
    if (tensor->type == GGML_TYPE_I32) snapshot.integers.reserve(elements);
    else snapshot.floats.reserve(elements);
    for (int64_t row = 0; row < tensor->ne[1]; ++row)
      for (int64_t slot = 0; slot < tensor->ne[0]; ++slot) {
        const size_t offset = row * tensor->nb[1] + slot * tensor->nb[0];
        if (offset > bytes.size() || 4 > bytes.size() - offset) throw std::runtime_error("routing stride exceeds storage");
        if (tensor->type == GGML_TYPE_I32) {
          int32_t value; std::memcpy(&value, bytes.data() + offset, 4);
          if (value < 0) throw std::runtime_error("negative routed expert ID");
          snapshot.integers.push_back(value);
        } else {
          float value; std::memcpy(&value, bytes.data() + offset, 4);
          if (!std::isfinite(value)) throw std::runtime_error("nonfinite routing weight");
          snapshot.floats.push_back(value);
        }
      }
    if (tensor->type == GGML_TYPE_I32) {
      if (pending_routes.count(layer)) throw std::runtime_error("duplicate unpaired routing IDs");
      routing_pending_bytes += elements * 4;
      pending_routes.emplace(layer, std::move(snapshot));
    } else {
      const auto found = pending_routes.find(layer);
      if (found == pending_routes.end()) throw std::runtime_error("routing weights lack matching IDs");
      emit_route(layer, found->second, &snapshot);
      routing_pending_bytes -= found->second.integers.size() * 4;
      pending_routes.erase(found);
    }
  }
  void finish_routes() {
    if (routing && routing->accepting())
      for (const auto& item : pending_routes) emit_route(item.first, item.second, nullptr);
    pending_routes.clear(); routing_pending_bytes = 0;
  }
  bool selected(const ggml_tensor* tensor) const noexcept {
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_I32) return false;
    const char* name = ggml_get_name(tensor);
    if (trace_mode == TraceMode::Compact) {
      for (const char* label : {"result_output", "ffn_moe_logits", "ffn_moe_topk",
          "ffn_moe_weights_norm", "ffn_moe_weights_scaled", "ffn_moe_out", "ffn_out",
          "ple_embd", "ple_gated_value"}) {
        if (named(name, label)) return true;
      }
      return false;
    }
    for (const char* prefix : {"ffn_moe_", "ffn_shexp", "shared_expert_gate", "ffn_out", "ple_", "result_output", "indexer_k_raw", "indexer_pool_k_new", "indexer_q", "indexer_top_k"}) {
      if (std::strncmp(name, prefix, std::strlen(prefix)) == 0) return true;
    }
    return false;
  }
  void capture(const ggml_tensor* tensor, const std::string& name) {
    const uint64_t elements = static_cast<uint64_t>(ggml_nelements(tensor));
    const uint64_t storage = ggml_nbytes(tensor);
    constexpr uint64_t limit = 64 * 1024 * 1024;
    if (elements > limit / 4 || storage > limit || trace_bytes > limit - elements * 4)
      throw std::runtime_error("trace payload exceeds 64 MiB; reduce prompt/generation or use compact tracing");
    trace_bytes += elements * 4;
    std::vector<uint8_t> bytes(static_cast<size_t>(storage));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    Snapshot item;
    item.evaluation = evaluation; item.prefill = prefill; item.name = name;
    std::copy_n(tensor->ne, 4, item.shape.begin());
    if (tensor->type == GGML_TYPE_F32) item.floats.reserve(static_cast<size_t>(elements));
    else item.integers.reserve(static_cast<size_t>(elements));
    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3)
      for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2)
        for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1)
          for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
            const size_t offset = i0 * tensor->nb[0] + i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3];
            if (offset > bytes.size() || 4 > bytes.size() - offset) throw std::runtime_error("trace tensor stride exceeds storage");
            if (tensor->type == GGML_TYPE_F32) {
              float value; std::memcpy(&value, bytes.data() + offset, 4);
              if (!std::isfinite(value)) throw std::runtime_error("non-finite inference trace: " + item.name);
              item.floats.push_back(value);
            } else {
              int32_t value; std::memcpy(&value, bytes.data() + offset, 4); item.integers.push_back(value);
            }
          }
    snapshots.push_back(std::move(item));
  }
  static bool evaluate(ggml_tensor* tensor, bool ask, void* data) noexcept {
    auto& self = *static_cast<Execution*>(data);
    if (ask) return (self.tracing && self.selected(tensor)) || self.routing_selected(tensor);
    try {
      if (self.routing_selected(tensor)) self.capture_routing(tensor);
      if (!self.tracing || !self.selected(tensor)) return true;
      const char* name = ggml_get_name(tensor);
      // The router consumes the exact post-HC FFN input. Snapshot that source
      // while this node is synchronized, before later nodes can recycle buffers.
      if (named(name, "ffn_moe_logits")) {
        const auto* weight = tensor->src[0];
        const auto* input = tensor->src[1];
        if (tensor->op != GGML_OP_MUL_MAT || !weight || !input || input->type != GGML_TYPE_F32 ||
            tensor->type != GGML_TYPE_F32 || input->ne[0] <= 0 || input->ne[1] <= 0 ||
            input->ne[2] != 1 || input->ne[3] != 1 || weight->ne[0] != input->ne[0] ||
            weight->ne[2] != 1 || weight->ne[3] != 1 || tensor->ne[0] != weight->ne[1] ||
            tensor->ne[1] != input->ne[1] || tensor->ne[2] != 1 || tensor->ne[3] != 1)
          throw std::runtime_error("unsupported router graph for exact FFN input capture: " + std::string(name));
        self.capture(input, "ffn_input" + std::string(name + std::strlen("ffn_moe_logits")));
      }
      self.capture(tensor, name);
    } catch (const std::exception& e) { self.fail(e.what()); }
    catch (...) { self.fail("unknown trace callback failure"); }
    // Scheduler callback false is not a reliable failure status. The CPU abort
    // callback and post-decode check propagate the latched error instead.
    return true;
  }
  void write(const std::filesystem::path& path, const GenerationResult& result) {
    if (!tracing) return;
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot create trace: " + path.string());
    out.imbue(std::locale::classic()); out << std::setprecision(std::numeric_limits<float>::max_digits10);
    out << "{\"trace_mode\":\"" << (trace_mode == TraceMode::Compact ? "compact" : "full") << "\",\"tokens\":[";
    for (size_t i = 0; i < result.tokens.size(); ++i) { if (i) out << ','; out << result.tokens[i]; }
    out << ']';
    if (!result.predictions.empty()) {
      out << ",\"predictions\":[";
      for (size_t i = 0; i < result.predictions.size(); ++i) { if (i) out << ','; out << result.predictions[i]; }
      out << ']';
    }
    out << ",\"records\":[";
    for (size_t i = 0; i < snapshots.size(); ++i) {
      if (i) out << ',';
      const auto& s = snapshots[i];
      out << "{\"evaluation\":" << s.evaluation << ",\"phase\":\"" << (s.prefill ? "prefill" : "decode")
          << "\",\"name\":" << quoted(s.name) << ",\"type\":\"" << (s.floats.empty() ? "i32" : "f32") << "\",\"shape\":[";
      for (size_t k = 0; k < s.shape.size(); ++k) { if (k) out << ','; out << s.shape[k]; }
      out << "],\"values\":[";
      if (!s.floats.empty()) for (size_t k = 0; k < s.floats.size(); ++k) { if (k) out << ','; out << s.floats[k]; }
      else for (size_t k = 0; k < s.integers.size(); ++k) { if (k) out << ','; out << s.integers[k]; }
      out << "]}";
    }
    out << "],\"metrics\":" << generation_metrics_json(result) << "}\n";
    if (!out) throw std::runtime_error("failed writing trace: " + path.string());
  }
};

GenerationResult generate(std::vector<std::filesystem::path> paths, const std::string& prompt,
                          const GenerationOptions& options, bool paged) {
  if (!options.forced_tokens.empty() &&
      (options.forced_tokens.size() > 65536 || options.max_tokens <= 0 ||
       options.forced_tokens.size() != static_cast<size_t>(options.max_tokens) ||
       std::any_of(options.forced_tokens.begin(), options.forced_tokens.end(), [](int32_t id) { return id < 0; })))
    throw std::invalid_argument("teacher forcing requires exactly max-tokens nonnegative IDs (maximum 65536)");
  if (paths.empty() || options.max_tokens <= 0 || !options.prefill_chunk || options.prefill_chunk > INT32_MAX)
    throw std::invalid_argument("model paths, positive max_tokens and prefill_chunk are required");
  if (prompt.size() > INT32_MAX) throw std::invalid_argument("prompt is too long");
  if (options.projection_activation != ProjectionActivationMode::BackendNative &&
      options.projection_activation != ProjectionActivationMode::CpuCompatible)
    throw std::invalid_argument("projection-activation must be backend-native or cpu-compatible");
  if (options.projection_activation == ProjectionActivationMode::CpuCompatible &&
      (!paged || options.projection_backend != ProjectionBackend::DeviceVulkan))
    throw std::invalid_argument("cpu-compatible projection-activation requires run with device-vulkan");
  if (options.upload_host_io != HostReadPolicy::Buffered &&
      (!paged || options.projection_backend == ProjectionBackend::Cpu))
    throw std::invalid_argument("device-upload host I/O requires paged device projection");
  if (options.transient_upload_host && (!paged || options.projection_backend == ProjectionBackend::Cpu))
    throw std::invalid_argument("transient upload host policy requires paged device projection");
  if (options.dense_load != DenseLoadPolicy::Default && options.dense_load != DenseLoadPolicy::DirectRequired)
    throw std::invalid_argument("dense-load must be default or direct-required");
  if ((options.dense_base_pages || !options.load_trace_jsonl.empty()) &&
      (!paged || options.dense_load != DenseLoadPolicy::DirectRequired || options.gpu_layers != 0 || options.cpu_repack))
    throw std::invalid_argument("dense page policy/loader trace requires run, direct-required, gpu-layers=0 and cpu-repack=off");
#ifndef __linux__
  if (options.dense_base_pages) throw std::invalid_argument("dense base pages require Linux");
#endif
  if (!paged && options.dense_load != DenseLoadPolicy::Default)
    throw std::invalid_argument("dense-load is only valid for the run command");
#ifndef __linux__
  if (options.dense_load == DenseLoadPolicy::DirectRequired)
    throw std::invalid_argument("dense-load direct-required requires Linux");
#endif
  if (paged && (options.gpu_layers != 0 || !options.cache_bytes))
    throw std::invalid_argument("PMI paged generation requires CPU gpu_layers=0 and positive cache capacity");
  if (paged && options.reference_mode != ReferenceMode::Default)
    throw std::invalid_argument("reference-mode is only valid for the reference command");
  if (!paged && (options.cache_policy != CachePolicy::Lru || options.profile_paging))
    throw std::invalid_argument("cache policy and paging profiling require the paged run command");
  if (options.split_cpu_misses && (!paged || options.projection_backend == ProjectionBackend::Cpu ||
      options.projection_activation != ProjectionActivationMode::BackendNative))
    throw std::invalid_argument("split dispatch requires a device backend and backend-native activation");
  if (options.projection_backend == ProjectionBackend::DeviceCpu ||
      options.projection_backend == ProjectionBackend::DeviceVulkan) {
    if (!paged || !options.device_cache_bytes || !options.device_graph_bytes ||
        options.device_cache_bytes > SIZE_MAX || options.device_graph_bytes > SIZE_MAX)
      throw std::invalid_argument("device projection requires paged run and explicit positive device cache/graph budgets");
    if (options.cache_policy != CachePolicy::Lru)
      throw std::invalid_argument("device projection currently requires lru; previous-top4 protects only host pages");
  } else if (options.projection_backend != ProjectionBackend::Cpu || options.device_cache_bytes || options.device_graph_bytes) {
    throw std::invalid_argument("device budgets require projection-backend device-cpu or device-vulkan");
  }
  if (options.projection_device_index && options.projection_backend != ProjectionBackend::DeviceVulkan)
    throw std::invalid_argument("projection-device-index is only valid for device-vulkan");
#ifndef PMI_WITH_VULKAN
  if (options.projection_backend == ProjectionBackend::DeviceVulkan)
    throw std::runtime_error("device-vulkan is unavailable in this build; rebuild with PMI_WITH_LLAMA=ON and GGML_VULKAN=ON");
#endif
  const bool demand = options.reference_mode == ReferenceMode::Demand;
  if (demand && options.gpu_layers != 0)
    throw std::invalid_argument("demand reference mode requires CPU gpu_layers=0");
  if (!options.routing_jsonl.empty() && !options.trace_json.empty() &&
      std::filesystem::weakly_canonical(options.routing_jsonl) == std::filesystem::weakly_canonical(options.trace_json))
    throw std::invalid_argument("routing-jsonl and trace-json require distinct destinations");
  const auto started = Clock::now();
  GenerationResult result;
  result.metrics.paged = paged;
  result.metrics.teacher_forcing = !options.forced_tokens.empty();
  result.metrics.reference_mode = options.reference_mode;
  result.metrics.trace_mode = options.trace_mode;
  result.metrics.cpu_repack = options.cpu_repack && !demand;
  result.metrics.cache_policy = options.cache_policy;
  result.metrics.projection_backend = options.projection_backend;
  result.metrics.projection_activation = options.projection_activation;
  result.metrics.transient_upload_host = options.transient_upload_host;
  result.metrics.upload_host_io = options.upload_host_io;
  result.metrics.routing_observation = !options.routing_jsonl.empty();
  // CMake disables Vulkan's implicit registry entry. The CPU/default path must
  // never call load_all(): that discovers optional GPUs even at gpu_layers=0.
  ggml_backend_register(ggml_backend_cpu_reg());
#ifdef PMI_WITH_VULKAN
  if (!paged && options.gpu_layers != 0) ggml_backend_register(ggml_backend_vk_reg());
#endif
  // Declared before backend/model/graph: borrowed observers never outlive it.
  std::unique_ptr<ProgressWriter> progress;
  FilePageProvider* progress_pages = nullptr;
  uint64_t progress_eval = 0;
  bool progress_prefill = false;
  if (!options.progress_jsonl.empty()) progress = std::make_unique<ProgressWriter>(options.progress_jsonl);
  auto mark = [&](const char* phase, const std::string& fields = std::string{}) {
    if (progress) progress->append(phase, fields);
  };
  DeviceCacheObserver cache_observer;
  if (progress) cache_observer = [&](const DeviceCacheEvent& e) {
    std::ostringstream fields;
    fields.imbue(std::locale::classic());
    fields << "\"evaluation\":" << progress_eval << ",\"prefill\":" << (progress_prefill ? "true" : "false")
           << ",\"request\":" << e.request;
    if (std::strcmp(e.phase, "cache_request_begin") == 0 || std::strcmp(e.phase, "cache_error") == 0)
      fields << ",\"shard\":" << e.shard << ",\"offset\":" << e.offset << ",\"bytes\":" << e.bytes;
    if (std::strcmp(e.phase, "allocation_end") == 0 || std::strcmp(e.phase, "cache_request_end") == 0)
      fields << ",\"device_current_bytes\":" << e.current_bytes << ",\"device_capacity_bytes\":" << e.capacity_bytes;
    if (std::strcmp(e.phase, "cache_request_end") == 0)
      fields << ",\"uploads\":" << e.uploads << ",\"upload_bytes\":" << e.upload_bytes;
    if (progress_pages && (std::strcmp(e.phase, "provider_end") == 0 || std::strcmp(e.phase, "cache_request_end") == 0)) {
      const auto host = progress_pages->stats();
      fields << ",\"host_current_bytes\":" << host.current_bytes << ",\"host_active_leases\":" << host.active_leases;
      if (std::strcmp(e.phase,"provider_end")==0) {
        const auto io=progress_pages->io_stats();
        fields << ",\"host_direct_read_bytes\":" << io.direct_read_bytes << ",\"host_buffered_read_bytes\":" << io.buffered_read_bytes
               << ",\"host_direct_fallbacks\":" << io.fallback_count;
      }
    }
    progress->append(e.phase, fields.str(), true);
  };
  if (progress) mark("run_config", "\"host_cache_capacity_bytes\":" + std::to_string(options.cache_bytes) +
       ",\"device_cache_capacity_bytes\":" + std::to_string(options.device_cache_bytes));
  mark("backend_begin");
  std::shared_ptr<GgmlDeviceBackend> projection_adapter;
  if (options.projection_backend != ProjectionBackend::Cpu) {
    GgmlBackendOwner backend;
    if (options.projection_backend == ProjectionBackend::DeviceCpu) {
      backend = GgmlBackendOwner(ggml_backend_cpu_init(), ggml_backend_free);
      if (!backend) throw std::runtime_error("cannot initialize explicit CPU projection backend");
      ggml_backend_cpu_set_n_threads(backend.get(), 1);
    } else {
#ifdef PMI_WITH_VULKAN
      const uint32_t index = options.projection_device_index.value_or(0);
      const int count = ggml_backend_vk_get_device_count();
      if (count <= 0 || index >= static_cast<uint32_t>(count))
        throw std::runtime_error("requested Vulkan projection device is unavailable: index " + std::to_string(index));
      backend = GgmlBackendOwner(ggml_backend_vk_init(index), ggml_backend_free);
      if (!backend || !ggml_backend_is_vk(backend.get()))
        throw std::runtime_error("cannot initialize selected Vulkan projection backend; no CPU fallback");
#else
      throw std::runtime_error("device-vulkan is unavailable in this build");
#endif
    }
    result.metrics.projection_device = selected_device(backend.get(),
        options.projection_backend == ProjectionBackend::DeviceVulkan ?
            std::optional<uint32_t>(options.projection_device_index.value_or(0)) : std::nullopt);
    projection_adapter = std::make_shared<GgmlDeviceBackend>(std::move(backend));
  }
  mark("backend_complete");
  // Declared before model/context so deferred descriptors and storage outlive all graphs.
  std::unique_ptr<FilePageProvider> pages;
  std::unique_ptr<PagedQwenGraph> graph;
  if (paged) {
    std::vector<GgufFile> files;
    for (const auto& path : paths) files.push_back(GgufReader(path).inspect());
    auto manifest = build_manifest(files);
    paths.clear();
    for (const auto& shard : manifest.shards) paths.push_back(shard.source_path);
    pages = std::make_unique<FilePageProvider>(paths, options.cache_bytes);
    progress_pages = pages.get();
    PagedGraphExecutionOptions execution_options;
    execution_options.retain_experts_per_layer = options.cache_policy == CachePolicy::PreviousTop4 ? 4 : 0;
    execution_options.profile = options.profile_paging;
    execution_options.split_cpu_misses = options.split_cpu_misses;
    if (projection_adapter) {
      execution_options.projection_executor = std::make_shared<GgmlDeviceProjection>(
          *pages, projection_adapter, options.device_cache_bytes, options.device_graph_bytes,
          options.profile_paging, options.projection_activation, cache_observer, options.transient_upload_host, options.upload_host_io);
    }
    graph = std::make_unique<PagedQwenGraph>(manifest, *pages, PagedGraphTraceOptions{}, execution_options);
  }
  Execution execution;
  execution.graph = graph.get(); execution.tracing = !options.trace_json.empty();
  execution.trace_mode = options.trace_mode;
  auto mp = llama_model_default_params();
  mp.n_gpu_layers = options.gpu_layers; mp.load_mtp = false;
  mp.use_extra_bufts = result.metrics.cpu_repack;
  if (options.dense_load == DenseLoadPolicy::DirectRequired)
    mp.load_mode = LLAMA_LOAD_MODE_DIRECT_IO_REQUIRED;
  if (demand) {
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.prefetch_mmap = false;
  }
  ggml_backend_dev_t cpu_only_devices[] = {nullptr};
  if (options.gpu_layers == 0) mp.devices = cpu_only_devices;
  if (paged) {
    mp.external_tensor = PagedQwenGraph::select;
    mp.external_tensor_op = PagedQwenGraph::build;
    mp.external_tensor_user_data = graph.get();
  }
  std::vector<std::string> model_paths;
  std::vector<const char*> model_path_pointers;
  for (const auto& path : paths) model_paths.push_back(path.string());
  for (const auto& path : model_paths) model_path_pointers.push_back(path.c_str());
  std::unique_ptr<DenseLoadObserver> load_observer;
  llama_load_observer binding{};
  if (options.dense_base_pages || !options.load_trace_jsonl.empty()) {
    load_observer = std::make_unique<DenseLoadObserver>(options, result.metrics);
    binding = load_observer->binding();
  }
  mark("model_load_begin");
  std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
      load_observer ? llama_model_load_from_splits_observed(model_path_pointers.data(), model_path_pointers.size(), mp, &binding) :
      paged ? llama_model_load_from_splits(model_path_pointers.data(), model_path_pointers.size(), mp)
            : llama_model_load_from_file(model_paths.front().c_str(), mp), llama_model_free);
  execution.check();
  if (load_observer) load_observer->check();
  if (!model) throw std::runtime_error("llama.cpp could not load model");
  if (paged) result.metrics.dense_load = dense_load_metrics(model.get(), options.dense_load, model_paths.size());
  if (progress) mark("model_load_complete", "\"metrics\":" + generation_metrics_json(result, false));
  const auto* vocab = llama_model_get_vocab(model.get());
  for (auto id : options.forced_tokens)
    if (id >= llama_vocab_n_tokens(vocab))
      throw std::invalid_argument("teacher-forcing token ID is outside model vocabulary");
  const int32_t needed = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
  if (needed >= 0 || needed == INT32_MIN) throw std::runtime_error("prompt tokenization failed");
  const int32_t count = -needed;
  std::vector<llama_token> prompt_tokens(static_cast<size_t>(count));
  if (llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), prompt_tokens.data(), count, true, true) != count)
    throw std::runtime_error("prompt tokenization failed");
  result.metrics.prompt_tokens = prompt_tokens.size();
  const uint64_t minimum_context = static_cast<uint64_t>(count) + static_cast<uint32_t>(options.max_tokens);
  const uint64_t context = options.context_length ? options.context_length : minimum_context;
  if (context < minimum_context || context > static_cast<uint64_t>(llama_model_n_ctx_train(model.get())) || context > UINT32_MAX)
    throw std::invalid_argument("context must contain prompt and generation and not exceed model training context");
  if (!options.routing_jsonl.empty()) {
    char architecture[128]{};
    if (llama_model_meta_val_str(model.get(), "general.architecture", architecture, sizeof(architecture)) < 0 ||
        std::string(architecture) != "qwen4exp")
      throw std::invalid_argument("routing observation currently requires the Qwen4Exp connector");
    execution.layer_count = llama_model_n_layer(model.get());
    static std::atomic<uint64_t> serial{0};
    const auto run_id = "pmi-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) +
                        "-" + std::to_string(serial.fetch_add(1));
    std::ostringstream header;
    header.imbue(std::locale::classic());
    // A local GGUF does not establish an upstream identity/revision or a named
    // quantization recipe. The offline importer can attach verified provenance.
    header << "\"sequence_id\":\"0\",\"split\":\"unassigned\",\"model\":{\"identity\":null,"
           << "\"revision\":null,\"quantization\":null,\"architecture\":\"qwen4exp\",\"layer_count\":"
           << execution.layer_count << "},\"config\":{\"max_tokens\":" << options.max_tokens
           << ",\"gpu_layers\":" << options.gpu_layers << ",\"prefill_chunk\":" << options.prefill_chunk
           << ",\"context_length\":" << context << ",\"cache_bytes\":" << (paged ? options.cache_bytes : 0)
           << ",\"cpu_repack\":" << (result.metrics.cpu_repack ? "true" : "false")
           << ",\"projection_backend\":" << (paged ? projection_backend_json(options.projection_backend) : "null")
           << ",\"device_upload_host_policy\":\"" << (options.transient_upload_host ? "transient" : "retained") << "\""
      << ",\"projection_activation\":" << (paged ? projection_activation_json(options.projection_activation) : "null")
           << ",\"projection_dispatch\":\"" << (options.split_cpu_misses ? "split" : "all") << "\""
           << ",\"projection_device_index\":";
    if (options.projection_backend == ProjectionBackend::DeviceVulkan) header << options.projection_device_index.value_or(0);
    else header << "null";
    header << ",\"projection_device\":";
    projection_device_json(header, result.metrics.projection_device);
    header << ",\"dense_load\":";
    dense_load_json(header, result.metrics.dense_load);
    header << ",\"device_cache_bytes\":" << options.device_cache_bytes
           << ",\"device_graph_bytes\":" << options.device_graph_bytes
           << ",\"cache_policy\":\"" << (options.cache_policy == CachePolicy::PreviousTop4 ? "previous-top4" : "lru")
           << "\",\"trace_mode\":\"" << (options.trace_mode == TraceMode::Compact ? "compact" : "full")
           << "\",\"sampling\":\"" << (options.forced_tokens.empty() ? "greedy" : "teacher-forced")
           << "\",\"mtp_enabled\":false,\"text_only\":true,"
           << "\"routing_max_rows\":65536,\"routing_max_bytes\":8388608,\"backend_commit\":" << quoted(ggml_commit())
           << ",\"backend_patch_sha256\":" << quoted(PMI_BACKEND_PATCH_SHA256)
           << ",\"implementation_commit\":" << (std::strlen(PMI_IMPLEMENTATION_COMMIT) ? quoted(PMI_IMPLEMENTATION_COMMIT) : "null")
           << ",\"implementation_dirty\":" << (PMI_IMPLEMENTATION_DIRTY ? "true" : "false") << '}'
           << ",\"profile\":{\"enabled\":" << (options.profile_paging ? "true" : "false") << '}';
    execution.routing = std::make_unique<RoutingTraceWriter>(options.routing_jsonl, run_id, header.str());
  }
  auto cp = llama_context_default_params();
  cp.n_ctx = static_cast<uint32_t>(context);
  cp.n_batch = cp.n_ubatch = options.prefill_chunk;
  cp.no_perf = false;
  cp.op_offload = false; cp.offload_kqv = false;
  cp.abort_callback = Execution::abort; cp.abort_callback_data = &execution;
  if (execution.tracing || execution.routing) { cp.cb_eval = Execution::evaluate; cp.cb_eval_user_data = &execution; }
  mark("context_begin");
  std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
  execution.check();
  if (!ctx) throw std::runtime_error("llama.cpp could not create context");
  mark("context_complete");
  std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(llama_sampler_init_greedy(), llama_sampler_free);
  if (!sampler) throw std::runtime_error("could not create greedy sampler");
  const auto setup_done = Clock::now();
  result.metrics.setup_ms = std::chrono::duration<double, std::milli>(setup_done - started).count();
  auto decode = [&](llama_token* tokens, int32_t size, bool prefill) {
    execution.prefill = prefill;
    execution.input_tokens = static_cast<uint64_t>(size);
    progress_eval = execution.evaluation; progress_prefill = prefill;
    if (progress) mark("evaluation_begin", "\"evaluation\":" + std::to_string(progress_eval) +
         ",\"prefill\":" + (prefill ? "true" : "false") + ",\"input_tokens\":" + std::to_string(size));
    const auto cache_before = pages ? pages->stats() : RuntimeStats{};
    const auto paging_before = graph ? graph->stats() : PagedGraphStats{};
    const auto memory_before = process_memory();
    const auto start = Clock::now();
    const int status = llama_decode(ctx.get(), llama_batch_get_one(tokens, size));
    execution.check();
    if (status != 0) throw std::runtime_error("llama_decode failed: " + std::to_string(status));
    if (graph && execution.evaluation == 0) graph->finish_device_seed();
    execution.finish_routes();
    EvaluationMetrics evaluation;
    evaluation.evaluation = execution.evaluation;
    evaluation.prefill = prefill;
    evaluation.input_tokens = static_cast<uint64_t>(size);
    evaluation.elapsed_ms = milliseconds(start);
    const auto cache_after = pages ? pages->stats() : RuntimeStats{};
    evaluation.io_bytes = cache_after.io_bytes - cache_before.io_bytes;
    evaluation.cache_hits = cache_after.hits - cache_before.hits;
    evaluation.cache_misses = cache_after.misses - cache_before.misses;
    evaluation.evictions = cache_after.evictions - cache_before.evictions;
    if (graph) evaluation.paging = paging_delta(graph->stats(), paging_before);
    evaluation.memory = process_memory();
    if (memory_before.storage_read_bytes && evaluation.memory.storage_read_bytes &&
        *evaluation.memory.storage_read_bytes >= *memory_before.storage_read_bytes)
      evaluation.process_storage_read_bytes_delta =
          *evaluation.memory.storage_read_bytes - *memory_before.storage_read_bytes;
    if (prefill) { result.metrics.prefill_ms += evaluation.elapsed_ms; ++result.metrics.prefill_evaluations; }
    else { result.metrics.decode_ms += evaluation.elapsed_ms; ++result.metrics.decode_evaluations; }
    if (execution.routing && execution.routing->accepting()) {
      std::ostringstream observation;
      observation.imbue(std::locale::classic()); observation << std::setprecision(12);
      observation << "\"eval_id\":" << evaluation.evaluation << ",\"sequence_id\":\"0\",\"phase\":\""
                  << (prefill ? "prefill" : "decode") << "\",\"metrics\":";
      evaluation_json(observation, evaluation);
      observation << ",\"per_layer_cache\":null";
      execution.routing->append("metrics", observation.str());
    }
    if (progress) {
      std::ostringstream fields;
      fields.imbue(std::locale::classic()); fields << std::setprecision(12) << "\"metrics\":";
      evaluation_json(fields, evaluation);
      fields << ",\"host_current_bytes\":" << cache_after.current_bytes << ",\"host_active_leases\":" << cache_after.active_leases;
      if(pages) {fields << ",\"cumulative_provider_io\":";provider_io_json(fields,pages->io_stats());}
      progress->append("evaluation_complete", fields.str());
    }
    result.metrics.evaluations.push_back(std::move(evaluation));
    execution.token_start += static_cast<uint64_t>(size);
    ++execution.evaluation;
  };
  for (size_t offset = 0; offset < prompt_tokens.size();) {
    const auto size = std::min<size_t>(options.prefill_chunk, prompt_tokens.size() - offset);
    decode(prompt_tokens.data() + offset, static_cast<int32_t>(size), true);
    offset += size;
  }
  if (progress) mark("prefill_complete", "\"evaluations\":" + std::to_string(result.metrics.prefill_evaluations));
  llama_token current = 0;
  for (int32_t step = 0; step < options.max_tokens; ++step) {
    if (step) decode(&current, 1, false);
    const auto predicted = llama_sampler_sample(sampler.get(), ctx.get(), -1);
    if (options.forced_tokens.empty()) {
      current = predicted;
      if (llama_vocab_is_eog(vocab, current)) break;
    } else {
      current = options.forced_tokens[static_cast<size_t>(step)];
      result.predictions.push_back(predicted);
      if (predicted == current) ++result.metrics.prediction_matches;
    }
    llama_sampler_accept(sampler.get(), current);
    result.tokens.push_back(current);
    std::vector<char> piece(256);
    int n = llama_token_to_piece(vocab, current, piece.data(), static_cast<int32_t>(piece.size()), 0, true);
    if (n < 0 && n != INT32_MIN) {
      piece.resize(static_cast<size_t>(-n));
      n = llama_token_to_piece(vocab, current, piece.data(), static_cast<int32_t>(piece.size()), 0, true);
    }
    if (n < 0) throw std::runtime_error("token-to-piece conversion failed");
    result.text.append(piece.data(), static_cast<size_t>(n));
    if (result.tokens.size() == 1) {
      const auto first_token = Clock::now();
      result.metrics.ttft_ms = std::chrono::duration<double, std::milli>(first_token - started).count();
      result.metrics.first_token_after_setup_ms = std::chrono::duration<double, std::milli>(first_token - setup_done).count();
      if (progress) mark("first_output_complete", "\"ttft_ms\":" + std::to_string(*result.metrics.ttft_ms));
    }
  }
  result.metrics.elapsed_ms = milliseconds(started);
  result.metrics.memory = process_memory();
  if (pages) {result.metrics.cache = pages->stats();result.metrics.provider_io = pages->io_stats();}
  if (graph) {
    const auto stats = graph->stats();
    result.metrics.paging = stats;
    result.metrics.page_requests = stats.page_requests;
    result.metrics.expert_requests = stats.expert_requests;
    result.metrics.lookup_requests = stats.lookup_requests;
    result.metrics.deferred_weight_bytes = stats.deferred_weight_bytes;
  }
  execution.write(options.trace_json, result);
  if (execution.routing) {
    if (execution.routing->accepting())
      execution.routing->append("metrics", "\"eval_id\":null,\"metrics\":" + generation_metrics_json(result, false) +
                               ",\"per_layer_cache\":null");
    execution.routing->finish(true);
  }
  if (progress) {
    progress->append("generation_complete", "\"metrics\":" + generation_metrics_json(result, false));
    progress->finish();
  }
  return result;
}
} // namespace
GenerationResult run_reference(const std::string& path, const std::string& prompt, const GenerationOptions& options) {
  return generate({path}, prompt, options, false);
}
GenerationResult run_paged(const std::vector<std::filesystem::path>& paths, const std::string& prompt, const GenerationOptions& options) {
  return generate(paths, prompt, options, true);
}
#else
GenerationResult run_reference(const std::string&, const std::string&, const GenerationOptions&) {
  throw std::runtime_error("PMI was built without llama.cpp; rebuild with CMake");
}
GenerationResult run_paged(const std::vector<std::filesystem::path>&, const std::string&, const GenerationOptions&) {
  throw std::runtime_error("PMI was built without llama.cpp; rebuild with CMake");
}
#endif
GenerationResult run_reference(const std::string& path, const std::string& prompt, int32_t max_tokens, int32_t gpu_layers) {
  GenerationOptions options; options.max_tokens = max_tokens; options.gpu_layers = gpu_layers;
  return run_reference(path, prompt, options);
}
} // namespace pmi
