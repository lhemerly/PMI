#pragma once

#include "pmi/device_cache.hpp"
#include <memory>
#include <stdexcept>
#include <vector>

namespace pmi {

enum class ProjectionActivationMode { BackendNative, CpuCompatible };

struct ProjectionStats {
  DeviceCacheStats cache;
  uint64_t calls{}, completed{}, failures{};
  uint64_t graph_current_bytes{}, graph_peak_bytes{}, graph_capacity_bytes{};
  uint64_t activation_upload_bytes{}, output_readback_bytes{};
  uint64_t activation_preparation_calls{}, activation_preparation_bytes{};
  // Peak vector storage for packed CPU activations plus their F32 expansion;
  // excludes allocator overhead, graph storage, input and output vectors.
  uint64_t activation_preparation_peak_bytes{};
  bool profile_enabled{};
  double cache_request_ms{}, graph_setup_ms{}, activation_upload_ms{};
  double kernel_ms{}, output_readback_ms{}, graph_teardown_ms{};
  double activation_preparation_ms{};
};

// Model-independent single-matrix projection. The descriptor is a compact
// [input.size(), output.size()] serialized matrix; activation/output are F32.
// One externally serialized stream, with a dedicated executor per consumer.
class ProjectionExecutor {
 public:
  virtual ~ProjectionExecutor() noexcept = default;
  virtual std::vector<float> project(const TensorInfo&, const std::vector<float>& input) = 0;
  virtual ProjectionStats stats() const = 0;
  virtual bool supports_resident_split() const noexcept { return false; }
  // Serialized planning: seed only during the first evaluation, then lookup.
  // Returned leases are consumed by one serialized device worker after all
  // planning has finished. No cache operation/stats/clear overlaps that worker.
  virtual DeviceLease resident(const TensorInfo&, bool) {
    throw std::runtime_error("projection executor does not support resident split");
  }
  virtual std::vector<float> project_resident(const DeviceLease&, const std::vector<float>&) {
    throw std::runtime_error("projection executor does not support resident split");
  }
  // Evict unleased cached weights; cumulative counters remain unchanged.
  virtual void clear() = 0;
};

#ifdef PMI_WITH_LLAMA
class GgmlDeviceBackend;

// Uses only the supplied, already-created ggml backend. No discovery, fallback,
// backend initialization or model-specific routing takes place here. Both byte
// limits are mandatory and independent: cached weights, and per-call input plus
// output graph buffers. Quotes and support checks precede host weight reads.
//
// Graph accounting reports aligned ggml buffer sizes, excluding CPU vectors,
// metadata, backend/driver heaps, hidden staging and kernel scratch. The combined
// limits are NOT a bound on total process/device memory; native telemetry and
// external guards remain necessary for GPU execution. Times are zero when off.
// CpuCompatible is an experimental Vulkan-only activation transform for IQ3_S,
// IQ4_XS, IQ4_NL and Q8_0. It uses the native CPU dot-product RHS quantizer, then
// uploads its F32 expansion with further Vulkan RHS quantization disabled.
// Matrix multiplication remains on the supplied Vulkan backend; its reduction
// order still differs from the CPU kernel. BackendNative preserves the default.
class GgmlDeviceProjection final : public ProjectionExecutor {
 public:
  GgmlDeviceProjection(PageProvider&, std::shared_ptr<GgmlDeviceBackend>,
                       uint64_t device_cache_bytes, uint64_t graph_buffer_bytes, bool profile = false,
                       ProjectionActivationMode activation_mode = ProjectionActivationMode::BackendNative,
                       DeviceCacheObserver observer = {}, bool transient_upload_host = false, HostReadPolicy io = HostReadPolicy::Buffered);
  ~GgmlDeviceProjection() noexcept override;
  GgmlDeviceProjection(const GgmlDeviceProjection&) = delete;
  GgmlDeviceProjection& operator=(const GgmlDeviceProjection&) = delete;
  std::vector<float> project(const TensorInfo&, const std::vector<float>& input) override;
  ProjectionStats stats() const noexcept override;
  bool supports_resident_split() const noexcept override { return true; }
  DeviceLease resident(const TensorInfo&, bool seed) override;
  std::vector<float> project_resident(const DeviceLease&, const std::vector<float>&) override;
  // No-op during a reentrant callback from project(); normal callers serialize
  // clear with project, so admitted/in-flight storage cannot be invalidated.
  void clear() noexcept override;
 private:
  std::vector<float> compute(const TensorInfo&, const std::vector<float>&, DeviceLease);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
#endif

} // namespace pmi
