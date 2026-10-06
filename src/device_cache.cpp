#include "pmi/device_cache.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace pmi {
namespace {
uint64_t add_counter(uint64_t value, uint64_t amount = 1) noexcept {
  const auto maximum = std::numeric_limits<uint64_t>::max();
  return amount > maximum - value ? maximum : value + amount;
}

struct DeviceLedger {
  std::atomic<uint64_t> current{0};
  std::atomic<uint64_t> peak{0};

  void add(uint64_t bytes) noexcept {
    const auto now = current.fetch_add(bytes) + bytes;
    auto previous = peak.load();
    while (previous < now && !peak.compare_exchange_weak(previous, now)) {}
  }
};

struct RequestScope {
  bool& active;
  explicit RequestScope(bool& flag) noexcept : active(flag) { active = true; }
  ~RequestScope() noexcept { active = false; }
};

struct Timer {
  double* destination;
  std::chrono::steady_clock::time_point start{};
  explicit Timer(double* value) noexcept : destination(value) {
    if (destination) start = std::chrono::steady_clock::now();
  }
  ~Timer() noexcept {
    if (destination) *destination += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }
};

struct Key {
  uint32_t shard;
  uint64_t offset;
  uint64_t length;
  uint32_t type;
  std::vector<uint64_t> shape;
  std::string format;
  std::string layout;

  explicit Key(const DevicePageRequest& request)
      : shard(request.source.shard_id), offset(request.source.absolute_offset),
        length(request.source.byte_length), type(request.source.ggml_type),
        shape(request.source.shape), format(request.format), layout(request.layout) {}

  bool operator<(const Key& other) const noexcept {
    return std::tie(shard, offset, length, type, shape, format, layout) <
           std::tie(other.shard, other.offset, other.length, other.type, other.shape, other.format, other.layout);
  }
};

void validate_range(const TensorInfo& source) {
  if (!source.byte_length || source.byte_length > std::numeric_limits<size_t>::max() ||
      source.absolute_offset > UINT64_MAX - source.byte_length)
    throw std::invalid_argument("PMI device cache: empty or overflowing source range");
}
} // namespace

struct DevicePage::Impl {
  DevicePageRequest request;
  // Backend ownership precedes allocation ownership: allocation teardown always
  // happens first, even when this is the final page after cache destruction.
  std::shared_ptr<DeviceBackend> backend;
  std::shared_ptr<DeviceLedger> ledger;
  std::unique_ptr<DeviceAllocation> allocation;
  uint64_t charge{};

  Impl(const DevicePageRequest& descriptor, std::shared_ptr<DeviceBackend> device,
       std::shared_ptr<DeviceLedger> accounting)
      : request(descriptor), backend(std::move(device)), ledger(std::move(accounting)) {}

  ~Impl() noexcept {
    allocation.reset();
    if (charge) ledger->current.fetch_sub(charge);
  }
};

DevicePage::DevicePage(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
DevicePage::~DevicePage() noexcept = default;
const DeviceAllocation& DevicePage::allocation() const noexcept { return *impl_->allocation; }
const DevicePageRequest& DevicePage::descriptor() const noexcept { return impl_->request; }
uint64_t DevicePage::charged_bytes() const noexcept { return impl_->charge; }

struct DeviceCache::Impl {
  struct Entry { DeviceLease page; uint64_t tick; };
  using Entries = std::map<Key, Entry>;

  PageProvider& provider;
  std::shared_ptr<DeviceBackend> backend;
  std::shared_ptr<DeviceLedger> ledger = std::make_shared<DeviceLedger>();
  Entries entries;
  uint64_t capacity;
  uint64_t tick{};
  bool in_request{};
  DeviceCacheStats counters;
  DeviceCacheObserver observer;
  bool transient_upload_host;
  HostReadPolicy host_io;
  void emit(const char* phase, const DevicePageRequest& r) {
    if(observer) observer({phase,r.source.shard_id,r.source.absolute_offset,r.source.byte_length,
      counters.requests,ledger->current.load(),capacity,counters.uploads,counters.upload_bytes});
  }

  Impl(PageProvider& pages, std::shared_ptr<DeviceBackend> device, uint64_t bytes, bool profile, DeviceCacheObserver o, bool transient, HostReadPolicy io)
      : provider(pages), backend(std::move(device)), capacity(bytes), observer(std::move(o)), transient_upload_host(transient), host_io(io) { counters.profile_enabled = profile; }
  double* time(double& value) noexcept { return counters.profile_enabled ? &value : nullptr; }
};

DeviceCache::DeviceCache(PageProvider& provider, std::shared_ptr<DeviceBackend> backend, uint64_t capacity, bool profile, DeviceCacheObserver observer, bool transient_upload_host, HostReadPolicy io) {
  if (io != HostReadPolicy::Buffered && io != HostReadPolicy::DirectPreferred && io != HostReadPolicy::DirectRequired)
    throw std::invalid_argument("unknown device-upload host I/O policy");
  if (!backend || !capacity)
    throw std::invalid_argument("PMI device cache: backend and nonzero capacity required");
  impl_ = std::make_unique<Impl>(provider, std::move(backend), capacity, profile, std::move(observer), transient_upload_host, io);
}
DeviceCache::~DeviceCache() noexcept = default;

DeviceLease DeviceCache::request(const DevicePageRequest& request) {
  return access(request, true, true);
}
DeviceLease DeviceCache::lookup(const DevicePageRequest& request) {
  return access(request, false, false);
}
DeviceLease DeviceCache::seed(const DevicePageRequest& request) {
  return access(request, true, false);
}
DeviceLease DeviceCache::access(const DevicePageRequest& request, bool admit, bool evict) {
  auto& state = *impl_;
  state.counters.requests = add_counter(state.counters.requests);
  state.counters.requested_bytes = add_counter(state.counters.requested_bytes, request.source.byte_length);
  if (state.in_request) {
    state.counters.failures = add_counter(state.counters.failures);
    throw std::logic_error("PMI device cache: request callback reentry is not supported");
  }
  RequestScope scope(state.in_request);
  try {
    state.emit("cache_request_begin",request);
    validate_range(request.source);
    Key key(request);
    state.tick = add_counter(state.tick);
    if (auto hit = state.entries.find(key); hit != state.entries.end()) {
      hit->second.tick = state.tick;
      state.counters.hits = add_counter(state.counters.hits);
      state.emit("cache_hit",request);
      return hit->second.page;
    }
    state.counters.misses = add_counter(state.counters.misses);
    if (!admit) {state.emit("cache_not_resident",request);return {};}

    // Descriptor validation and the exact allocation quote are backend-specific;
    // neither may start an allocation/upload before capacity has been admitted.
    const auto charge = state.backend->allocation_bytes(request);
    if (!charge) throw std::runtime_error("PMI device cache: backend quoted a zero allocation charge");
    if (charge > state.capacity && !evict) {state.emit("cache_capacity_miss",request);return {};}
    if (charge > state.capacity)
      throw std::runtime_error("PMI device cache: allocation charge exceeds capacity");
    const auto current = state.ledger->current.load();
    if (current > state.capacity) throw std::runtime_error("PMI device cache: accounting exceeds capacity");
    const auto available = state.capacity - current;
    const auto needed = charge > available ? charge - available : 0;
    if (needed && !evict) {state.emit("cache_capacity_miss",request);return {};}
    std::vector<Impl::Entries::iterator> victims;
    uint64_t reclaim = 0;
    size_t victim_count = 0;
    if (needed) {
      for (auto it = state.entries.begin(); it != state.entries.end(); ++it)
        if (it->second.page.use_count() == 1) victims.push_back(it);
      std::sort(victims.begin(), victims.end(), [](const auto& a, const auto& b) {
        return std::tie(a->second.tick, a->first) < std::tie(b->second.tick, b->first);
      });
      while (reclaim < needed && victim_count < victims.size()) {
        reclaim += victims[victim_count]->second.page->charged_bytes();
        ++victim_count;
      }
      if (reclaim < needed)
        throw std::runtime_error("PMI device cache: capacity held by active device leases");
    }

    // Prepare all entry/page/control-block bookkeeping before any device side
    // effects. The map node can be published after upload without allocating.
    auto storage = std::make_unique<DevicePage::Impl>(request, state.backend, state.ledger);
    auto* storage_pointer = storage.get();
    DeviceLease page(new DevicePage(std::move(storage)));
    Impl::Entries prepared;
    prepared.emplace(std::move(key), Impl::Entry{page, state.tick});
    auto node = prepared.extract(prepared.begin());

    // Keep this lease alive until upload_sync has completed, including its
    // exception path. A failed host read leaves all prior device pages intact.
    state.counters.provider_requests = add_counter(state.counters.provider_requests);
    state.emit("provider_begin",request);
    const auto host = [&] {
      Timer timer(state.time(state.counters.provider_read_ms));
      if (state.host_io != HostReadPolicy::Buffered)
        return state.provider.request_direct(request.source, state.transient_upload_host, state.host_io == HostReadPolicy::DirectRequired).get();
      return (state.transient_upload_host ? state.provider.request_transient(request.source) :
          state.provider.request(request.source)).get();
    }();
    state.emit("provider_end",request);
    if (!host || host->size() != request.source.byte_length)
      throw std::runtime_error("PMI device cache: host page has unexpected size");

    // An abstract host callback can promote a weak lease without requesting a
    // new page. Preserve those entries and reject the stale eviction plan.
    for (size_t i = 0; i < victim_count; ++i)
      if (victims[i]->second.page.use_count() != 1)
        throw std::runtime_error("PMI device cache: a planned victim acquired a device lease");

    // Freed victims cannot be restored on later device failure without keeping
    // duplicate allocations outside the bound. No partial new entry is exposed.
    state.emit("eviction_begin",request);
    {
      Timer timer(state.time(state.counters.eviction_ms));
      for (size_t i = 0; i < victim_count; ++i) {
        if (victims[i]->second.page.use_count() != 1)
          throw std::runtime_error("PMI device cache: a planned victim acquired a device lease");
        state.entries.erase(victims[i]);
        state.counters.evictions = add_counter(state.counters.evictions);
      }
    }
    state.emit("eviction_end",request);
    const auto current_after_eviction = state.ledger->current.load();
    if (current_after_eviction > state.capacity || charge > state.capacity - current_after_eviction)
      throw std::runtime_error("PMI device cache: live allocation charges prevent admission after eviction");
    state.emit("allocation_begin",request);
    auto allocation = [&] {
      Timer timer(state.time(state.counters.allocation_ms));
      return state.backend->allocate(request, charge);
    }();
    if (!allocation) throw std::runtime_error("PMI device cache: backend returned no allocation");
    if (allocation->charged_bytes() != charge)
      throw std::runtime_error("PMI device cache: allocation charge differs from preflight quote");
    storage_pointer->allocation = std::move(allocation);
    storage_pointer->charge = charge;
    state.ledger->add(charge);
    state.emit("allocation_end",request);
    state.emit("upload_begin",request);
    {
      Timer timer(state.time(state.counters.upload_ms));
      state.backend->upload_sync(*storage_pointer->allocation, *host, request);
    }

    state.emit("upload_end",request);
    const auto installed = state.entries.insert(std::move(node));
    if (!installed.inserted)
      throw std::runtime_error("PMI device cache: duplicate cache insertion");
    state.counters.uploads = add_counter(state.counters.uploads);
    state.counters.upload_bytes = add_counter(state.counters.upload_bytes, request.source.byte_length);
    state.emit("cache_request_end",request);
    return page;
  } catch (...) {
    state.counters.failures = add_counter(state.counters.failures);
    try {state.emit("cache_error",request);} catch (...) {}
    throw;
  }
}

uint64_t DeviceCache::clear() {
  auto& state = *impl_;
  if (state.in_request) throw std::logic_error("PMI device cache: clear callback reentry is not supported");
  RequestScope scope(state.in_request);
  Timer timer(state.time(state.counters.eviction_ms));
  uint64_t removed = 0;
  for (auto it = state.entries.begin(); it != state.entries.end();) {
    if (it->second.page.use_count() != 1) { ++it; continue; }
    it = state.entries.erase(it);
    state.counters.evictions = add_counter(state.counters.evictions);
    ++removed;
  }
  return removed;
}

DeviceCacheStats DeviceCache::stats() const noexcept {
  const auto& state = *impl_;
  auto result = state.counters;
  result.current_bytes = state.ledger->current.load();
  result.peak_bytes = state.ledger->peak.load();
  result.capacity_bytes = state.capacity;
  result.entries = state.entries.size();
  for (const auto& entry : state.entries) {
    const auto copies = entry.second.page.use_count();
    if (copies > 1) result.active_leases = add_counter(result.active_leases, static_cast<uint64_t>(copies - 1));
  }
  return result;
}

} // namespace pmi
