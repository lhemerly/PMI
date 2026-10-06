#pragma once

#include "pmi/device_cache.hpp"
#include "ggml-backend.h"

#include <memory>

namespace pmi {

// A caller-created, shared backend handle. The caller installs its deleter
// (normally ggml_backend_free); this adapter never discovers or opens a device.
using GgmlBackendOwner = std::shared_ptr<ggml_backend>;

// Synchronous adapter for compact serialized ggml rows/matrices. It preserves
// the supplied shape and quantized bytes; it never expands an expert dimension
// or repacks the payload. Backend kernel support remains a consumer check.
//
// The charge is the aligned ggml buffer allocation, verified against the
// buffer's reported size. It excludes metadata, backend-global allocations,
// driver heaps, internal staging and kernel workspaces. It is not a VRAM meter.
// All calls and consumers of this backend must be externally serialized.
// Upload and destruction synchronize the supplied stream. If ggml throws from
// synchronization, this adapter terminates: returning could free in-use memory,
// and the public ggml API provides no recoverable completion result.
class GgmlDeviceBackend final : public DeviceBackend {
 public:
  static constexpr const char* format = "ggml";
  static constexpr const char* layout = "contiguous";

  explicit GgmlDeviceBackend(GgmlBackendOwner);
  ~GgmlDeviceBackend() noexcept override;
  GgmlDeviceBackend(const GgmlDeviceBackend&) = delete;
  GgmlDeviceBackend& operator=(const GgmlDeviceBackend&) = delete;

  uint64_t allocation_bytes(const DevicePageRequest&) override;
  std::unique_ptr<DeviceAllocation> allocate(const DevicePageRequest&, uint64_t charged_bytes) override;
  void upload_sync(DeviceAllocation&, const Page&, const DevicePageRequest&) override;

  // Checks that the allocation belongs to this adapter. The mutable pointer is
  // required by ggml graph constructors; consumers must treat the weight's
  // metadata and payload as immutable and retain the DeviceLease until their
  // kernel and readback finish. This accessor does not submit any work.
  ggml_tensor* tensor(const DevicePage&) const;
  ggml_backend_t backend() const noexcept;

 private:
  struct State;
  std::shared_ptr<State> state_;
};

} // namespace pmi
