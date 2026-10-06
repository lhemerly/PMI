#pragma once

#include "pmi/page_provider.hpp"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace pmi {

// Format/layout tags are opaque, case-sensitive backend identifiers. The backend
// validates them during allocation_bytes(). Type and shape also participate in
// identity; source name and relative offset do not change the requested bytes.
struct DevicePageRequest {
  TensorInfo source;
  std::string format;
  std::string layout;
};

class DeviceAllocation {
 public:
  virtual ~DeviceAllocation() noexcept = default;
  // Immutable charge including backend allocation padding, not logical bytes.
  virtual uint64_t charged_bytes() const noexcept = 0;
};

class DeviceBackend {
 public:
  virtual ~DeviceBackend() noexcept = default;
  // Validate the descriptor and quote the exact, nonzero allocation charge.
  // This must not allocate device storage or start work/retain any host pointer.
  virtual uint64_t allocation_bytes(const DevicePageRequest&) = 0;
  // The returned allocation must report exactly the quoted charge. Failed calls
  // clean up their resources before throwing. Destruction must never throw.
  virtual std::unique_ptr<DeviceAllocation> allocate(const DevicePageRequest&, uint64_t charged_bytes) = 0;
  // Must complete all upload work before returning OR throwing; never retain a
  // host pointer after this call. Backend adapters are responsible for waiting
  // on their native API. The cache keeps the immutable Page lease throughout.
  virtual void upload_sync(DeviceAllocation&, const Page&, const DevicePageRequest&) = 0;
};

class DevicePage final {
 public:
  ~DevicePage() noexcept;
  DevicePage(const DevicePage&) = delete;
  DevicePage& operator=(const DevicePage&) = delete;
  const DeviceAllocation& allocation() const noexcept;
  const DevicePageRequest& descriptor() const noexcept;
  uint64_t charged_bytes() const noexcept;
 private:
  friend class DeviceCache;
  struct Impl;
  explicit DevicePage(std::unique_ptr<Impl>) noexcept;
  std::unique_ptr<Impl> impl_;
};

using DeviceLease = std::shared_ptr<const DevicePage>;

struct DeviceCacheStats {
  bool profile_enabled{};
  // Synchronous service wall time, including failure paths; no clock reads
  // when disabled. Read includes provider cache/IO, upload includes fence wait.
  double provider_read_ms{}, allocation_ms{}, upload_ms{}, eviction_ms{};
  uint64_t requests{};
  uint64_t provider_requests{}; // Actual host request invocations, including throws/invalid futures; saturating.
  uint64_t requested_bytes{}; // Sum of request byte_length arguments, including hits/failures; saturating.
  uint64_t hits{};
  uint64_t misses{}; // Valid source ranges absent from cache, including subsequent preflight failures.
  uint64_t failures{};
  uint64_t upload_bytes{}; // Logical host bytes in successfully completed uploads only.
  uint64_t uploads{};
  uint64_t evictions{}; // Includes unleased entries removed by clear().
  uint64_t current_bytes{};
  uint64_t peak_bytes{}; // Includes temporary device allocations on failed uploads.
  uint64_t capacity_bytes{};
  uint64_t entries{};
  uint64_t active_leases{}; // Outstanding shared_ptr copies beyond the cache's own references.
};

// Synchronous, backend-independent exact-range device cache. Cache operations
// and acquiring/copying/promoting leases must be externally serialized. Releases
// use shared ownership and atomic byte accounting; concurrent weak promotion
// during requests is unsupported. Mutating methods reject callback reentry.
// No asynchronous workers or GPU discovery are started.
// Provider is borrowed and must outlive the cache. Backend ownership is shared
// by the cache and every device page, so leases survive both cache and provider.
// Consumers retain their lease until their device use is synchronously complete;
// this interface does not model fences or asynchronous in-flight consumers.
//
// Bounds backend-reported charged allocations only, excluding host pages,
// backend-global state, cache metadata, workspaces and unrelated allocations.
// Each cache has one immutable backend identity and immutable source shards.
// Borrowed synchronous event; observer may throw to fail an explicitly requested
// capture. No reentry into the cache. Counters exclude hidden driver allocations.
struct DeviceCacheEvent {
  const char* phase;
  uint32_t shard;
  uint64_t offset, bytes, request, current_bytes, capacity_bytes, uploads, upload_bytes;
};
using DeviceCacheObserver = std::function<void(const DeviceCacheEvent&)>;

class DeviceCache final {
 public:
  // Opt-in transient host misses stay charged/alive through upload, then expire;
  // existing host cache hits and pins are preserved. Default retains misses.
  DeviceCache(PageProvider&, std::shared_ptr<DeviceBackend>, uint64_t capacity_bytes, bool profile = false, DeviceCacheObserver observer = {}, bool transient_upload_host = false, HostReadPolicy io = HostReadPolicy::Buffered);
  ~DeviceCache() noexcept;
  DeviceCache(const DeviceCache&) = delete;
  DeviceCache& operator=(const DeviceCache&) = delete;

  // Capacity/preflight failures occur before host reads or device allocation.
  // Host-read failure preserves existing entries. Allocation/upload failures
  // publish no entry, but planned unleased victims may already have been freed
  // to keep the strict charge bound. Leased pages are never evicted.
  DeviceLease request(const DevicePageRequest&);
  // Count an exact-range selection without fetching an absent page.
  DeviceLease lookup(const DevicePageRequest&);
  // Admit only into currently free capacity. Never evict; a capacity miss
  // returns empty before host reads/allocation. Other validation/IO errors throw.
  DeviceLease seed(const DevicePageRequest&);
  // Remove only unleased entries; live leases remain cached and fully charged.
  // Returns the number removed, without resetting cumulative counters.
  uint64_t clear();
  DeviceCacheStats stats() const noexcept;
 private:
  DeviceLease access(const DevicePageRequest&, bool admit, bool evict);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace pmi
