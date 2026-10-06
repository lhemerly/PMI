#include "pmi/device_cache.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F&& action, const char* message) {
  bool failed = false;
  try { action(); } catch (const std::exception&) { failed = true; }
  require(failed, message);
}

struct Directory {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("pmi-device-cache-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::vector<std::filesystem::path> shards;
  Directory() {
    std::filesystem::create_directories(path);
    for (int shard = 0; shard < 2; ++shard) {
      shards.push_back(path / ("shard-" + std::to_string(shard) + ".bin"));
      std::ofstream output(shards.back(), std::ios::binary);
      for (int i = 0; i < 1024; ++i) output.put(static_cast<char>((i + shard * 29) % 256));
      require(static_cast<bool>(output), "cannot write device-cache fixture");
    }
  }
  ~Directory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};

pmi::TensorInfo tensor(uint64_t offset = 0, uint64_t elements = 4, uint32_t shard = 0) {
  pmi::TensorInfo result;
  result.name = "fixture";
  result.ggml_type = 0; // F32; the mock device preserves the source bytes.
  result.shape = {elements};
  result.absolute_offset = offset;
  result.byte_length = elements * sizeof(float);
  result.shard_id = shard;
  return result;
}

// The real Rust provider supplies host leases. Only its request boundary is
// wrapped so the tests can prove preflight ordering and inject invalid results.
class Provider final : public pmi::PageProvider {
 public:
  explicit Provider(const Directory& fixture, size_t capacity = 1024)
      : file(fixture.shards, capacity) {}
  pmi::PageFuture request(const pmi::TensorInfo& value) override {
    ++requests;
    if (before_request) before_request();
    if (throw_request) throw std::runtime_error("injected host request failure");
    if (invalid_future) return {};
    if (null_page) {
      std::promise<pmi::PageLease> result;
      result.set_value({});
      return result.get_future().share();
    }
    auto page = file.request(substitute ? replacement : value);
    if (page.valid()) last_page = page.get();
    return page;
  }
  pmi::PageFuture prefetch(const pmi::TensorInfo& value) override {
    ++prefetches;
    return file.prefetch(value);
  }
  bool pin(const pmi::TensorInfo& value) override { ++pins; return file.pin(value); }
  bool release(const pmi::TensorInfo& value) override { ++releases; return file.release(value); }

  pmi::FilePageProvider file;
  size_t requests{}, prefetches{}, pins{}, releases{};
  bool throw_request{}, invalid_future{}, null_page{}, substitute{};
  pmi::TensorInfo replacement;
  std::weak_ptr<const pmi::Page> last_page;
  std::function<void()> before_request;
};

struct DeviceState {
  uint64_t quote_calls{}, allocation_calls{}, allocations{}, frees{}, upload_calls{};
  uint64_t live_bytes{}, peak_bytes{}, backend_destructions{};
  bool backend_alive{true}, freed_after_backend{};
  bool throw_quote{}, zero_quote{}, oversized_quote{}, throw_allocate{}, null_allocate{};
  bool wrong_charge{}, throw_upload{};
  std::function<void()> before_quote;
  std::function<void(const pmi::Page&)> during_upload;
};

class Allocation final : public pmi::DeviceAllocation {
 private:
  std::shared_ptr<DeviceState> state_;
  uint64_t charge_;
 public:
  Allocation(std::shared_ptr<DeviceState> state, uint64_t bytes)
      : state_(std::move(state)), charge_(bytes), id(++state_->allocations) {
    state_->live_bytes += charge_;
    state_->peak_bytes = std::max(state_->peak_bytes, state_->live_bytes);
  }
  ~Allocation() noexcept override {
    if (!state_->backend_alive) state_->freed_after_backend = true;
    state_->live_bytes -= charge_;
    ++state_->frees;
  }
  uint64_t charged_bytes() const noexcept override { return charge_; }
  std::vector<uint8_t> bytes;
  uint64_t id;
};

// This backend has no hardware dependency. F32 and I32 both use four-byte
// elements; accepting both permits an isolated check of the type cache key.
class Backend final : public pmi::DeviceBackend {
 public:
  explicit Backend(std::shared_ptr<DeviceState> state) : state_(std::move(state)) {}
  ~Backend() noexcept override { state_->backend_alive = false; ++state_->backend_destructions; }
  uint64_t allocation_bytes(const pmi::DevicePageRequest& request) override {
    ++state_->quote_calls;
    if (state_->before_quote) state_->before_quote();
    if (state_->throw_quote) throw std::runtime_error("injected quote failure");
    if (state_->zero_quote) return 0;
    if (state_->oversized_quote) return std::numeric_limits<uint64_t>::max();
    if (request.format != "raw" && request.format != "raw-v2")
      throw std::invalid_argument("unsupported mock format");
    if (request.layout != "row-major" && request.layout != "column-major")
      throw std::invalid_argument("unsupported mock layout");
    if (request.source.ggml_type != 0 && request.source.ggml_type != 26)
      throw std::invalid_argument("unsupported mock type");
    if (request.source.shape.empty() || request.source.shape.size() > 4)
      throw std::invalid_argument("unsupported mock rank");
    uint64_t bytes = sizeof(float);
    for (const auto dimension : request.source.shape) {
      if (!dimension || dimension > std::numeric_limits<uint64_t>::max() / bytes)
        throw std::invalid_argument("invalid mock dimensions");
      bytes *= dimension;
    }
    if (bytes != request.source.byte_length || bytes > std::numeric_limits<uint64_t>::max() - 63)
      throw std::invalid_argument("invalid mock byte length");
    return (bytes + 63) & ~uint64_t{63};
  }
  std::unique_ptr<pmi::DeviceAllocation> allocate(const pmi::DevicePageRequest&, uint64_t charge) override {
    ++state_->allocation_calls;
    if (state_->throw_allocate) throw std::runtime_error("injected device allocation failure");
    if (state_->null_allocate) return {};
    return std::make_unique<Allocation>(state_, charge + (state_->wrong_charge ? 1 : 0));
  }
  void upload_sync(pmi::DeviceAllocation& allocation, const pmi::Page& page,
                   const pmi::DevicePageRequest& request) override {
    ++state_->upload_calls;
    require(page.size() == request.source.byte_length, "upload received incorrect host byte length");
    if (state_->during_upload) state_->during_upload(page);
    auto& target = dynamic_cast<Allocation&>(allocation);
    target.bytes.assign(page.data(), page.data() + page.size());
    if (state_->throw_upload) throw std::runtime_error("injected partially written upload failure");
  }
 private:
  std::shared_ptr<DeviceState> state_;
};

pmi::DevicePageRequest request(uint64_t offset = 0, uint64_t elements = 4, uint32_t shard = 0) {
  return {tensor(offset, elements, shard), "raw", "row-major"};
}
const Allocation& allocation(const pmi::DeviceLease& page) {
  return dynamic_cast<const Allocation&>(page->allocation());
}
void expect_bytes(const pmi::DeviceLease& page) {
  const auto& source = page->descriptor().source;
  const auto& bytes = allocation(page).bytes;
  require(bytes.size() == source.byte_length, "device payload has incorrect size");
  for (size_t i = 0; i < bytes.size(); ++i)
    require(bytes[i] == static_cast<uint8_t>((source.absolute_offset + i + source.shard_id * 29) % 256),
            "device payload differs from its source shard/range");
}

void exact_identity(const Directory& fixture) {
  Provider provider(fixture);
  auto state = std::make_shared<DeviceState>();
  pmi::DeviceCache cache(provider, std::make_shared<Backend>(state), 1024);
  const auto original = request();
  auto first = cache.request(original);
  auto alias = original;
  alias.source.name = "name does not change byte identity";
  alias.source.offset = 999;
  auto hit = cache.request(alias);
  require(&first->allocation() == &hit->allocation(), "logical alias caused another upload");
  require(provider.requests == 1 && state->upload_calls == 1, "device hit read host storage");
  require(cache.stats().active_leases == 2, "device hit aliases missing from lease accounting");
  std::vector<pmi::DevicePageRequest> variants;
  variants.push_back(request(0, 4, 1));
  variants.push_back(request(32));
  variants.push_back(request(0, 2));
  auto typed = original; typed.source.ggml_type = 26; variants.push_back(typed);
  auto shaped = original; shaped.source.shape = {2, 2}; variants.push_back(shaped);
  auto formatted = original; formatted.format = "raw-v2"; variants.push_back(formatted);
  auto layout = original; layout.layout = "column-major"; variants.push_back(layout);
  std::vector<pmi::DeviceLease> pages{first};
  for (const auto& variant : variants) {
    auto page = cache.request(variant);
    for (const auto& previous : pages)
      require(&previous->allocation() != &page->allocation(), "device key aliased a distinct descriptor");
    expect_bytes(page);
    pages.push_back(std::move(page));
  }
  expect_bytes(first);
  const auto stats = cache.stats();
  require(stats.requests == 9 && stats.hits == 1 && stats.misses == 8 && stats.uploads == 8 &&
          stats.failures == 0 && stats.evictions == 0 && stats.entries == 8,
          "exact-identity counters do not reconcile");
  require(stats.requested_bytes == 136 && stats.upload_bytes == 120 && stats.current_bytes == 512 &&
          stats.peak_bytes == 512 && stats.capacity_bytes == 1024,
          "logical bytes and aligned device charge were conflated");
  require(stats.provider_requests == 8 && stats.provider_requests == provider.requests &&
          !provider.prefetches && !provider.pins && !provider.releases,
          "device cache performed unexpected host requests or pins");
}

void eviction_leases_and_reload(const Directory& fixture) {
  Provider provider(fixture, 64); // Only one host range survives: reload must perform actual I/O.
  auto state = std::make_shared<DeviceState>();
  pmi::DeviceCache cache(provider, std::make_shared<Backend>(state), 128);
  auto a = cache.request(request());
  const auto original_id = allocation(a).id;
  { auto hit = cache.request(request()); require(allocation(hit).id == original_id, "cache hit reallocated"); }
  auto b = cache.request(request(32));
  const auto host_requests = provider.requests;
  rejects([&] { cache.request(request(64)); }, "active consumer leases were evicted");
  require(provider.requests == host_requests && state->allocation_calls == 2,
          "over-budget admission touched host or device allocation");
  require(cache.clear() == 0 && cache.stats().current_bytes == 128, "clear hid leased device allocations");
  expect_bytes(a); expect_bytes(b);
  a.reset();
  auto c = cache.request(request(64));
  rejects([&] { cache.request(request()); }, "second active lease pressure was ignored");
  expect_bytes(b); expect_bytes(c);
  b.reset(); c.reset();
  a = cache.request(request());
  require(allocation(a).id != original_id, "evicted device allocation was incorrectly reused");
  expect_bytes(a);
  auto stats = cache.stats();
  require(stats.requests == 7 && stats.hits == 1 && stats.misses == 6 && stats.failures == 2 &&
          stats.uploads == 4 && stats.evictions == 2 && stats.entries == 2 && stats.active_leases == 1,
          "eviction/reload counters do not reconcile");
  require(stats.upload_bytes == 64 && stats.current_bytes == 128 && stats.peak_bytes == 128 &&
          state->peak_bytes == 128 && provider.file.stats().io_bytes == 64,
          "reload I/O or strict aligned allocation budget mismatch");
  require(cache.clear() == 1 && cache.stats().current_bytes == 64, "clear failed to remove unleased entry");
  a.reset();
  require(cache.clear() == 1 && !cache.stats().current_bytes && !state->live_bytes &&
          cache.stats().evictions == 4, "clear leaked a retired device allocation");
}

void independent_lifetimes(const Directory& fixture) {
  auto state = std::make_shared<DeviceState>();
  auto backend = std::make_shared<Backend>(state);
  std::weak_ptr<Backend> backend_lifetime = backend;
  std::weak_ptr<const pmi::Page> host_lifetime;
  pmi::DeviceLease survivor;
  {
    Provider provider(fixture, 64);
    state->during_upload = [&](const pmi::Page& page) {
      const auto active = provider.last_page.lock();
      require(active && active.get() == &page, "host page died or was copied before upload completion");
      host_lifetime = active;
    };
    {
      pmi::DeviceCache cache(provider, backend, 64);
      survivor = cache.request(request());
      require(host_lifetime.expired(), "device cache retained a host lease after synchronous upload");
      backend.reset();
      require(!backend_lifetime.expired(), "cache released its backend while allocations exist");
    }
    state->during_upload = {};
  }
  require(!backend_lifetime.expired() && state->live_bytes == 64,
          "device lease did not survive cache/provider destruction");
  expect_bytes(survivor);
  survivor.reset();
  require(backend_lifetime.expired() && !state->live_bytes && state->backend_destructions == 1 &&
          !state->freed_after_backend, "allocation/backend destruction order is unsafe");
}

void lru_and_variable_charges(const Directory& fixture) {
  Provider provider(fixture);
  auto state = std::make_shared<DeviceState>();
  pmi::DeviceCache cache(provider, std::make_shared<Backend>(state), 192);
  std::weak_ptr<const pmi::DevicePage> a, b, c;
  { auto page = cache.request(request()); a = page; }
  { auto page = cache.request(request(32)); b = page; }
  { auto page = cache.request(request(64)); c = page; }
  cache.request(request()); // A becomes newer than both B and C.
  cache.request(request(96));
  require(!a.expired() && b.expired() && !c.expired(), "cache hit did not update LRU recency");
  const auto host_requests = provider.requests;
  cache.request(request()); cache.request(request(64)); cache.request(request(96));
  require(provider.requests == host_requests, "LRU retained the wrong three pages");

  // Keep A leased. The 80-byte request needs a 128-byte allocation, so both
  // other unleased 64-byte allocations must retire before allocation starts.
  auto active = cache.request(request());
  auto large = cache.request(request(128, 20));
  require(!a.expired() && c.expired() && large->charged_bytes() == 128,
          "variable charge admission did not preserve the active small allocation");
  expect_bytes(active); expect_bytes(large);
  const auto stats = cache.stats();
  require(stats.entries == 2 && stats.evictions == 3 && stats.current_bytes == 192 &&
          stats.peak_bytes == 192 && state->peak_bytes == 192 && state->frees == 3,
          "multi-victim admission exceeded the budget or retired the wrong charge");
  rejects([&] { cache.request(request(32)); }, "variable-size leased allocations escaped capacity enforcement");
}

void preflight_errors(const Directory& fixture) {
  Provider provider(fixture);
  auto state = std::make_shared<DeviceState>();
  auto backend = std::make_shared<Backend>(state);
  rejects([&] { pmi::DeviceCache invalid(provider, backend, 0); }, "zero device capacity accepted");
  rejects([&] { pmi::DeviceCache invalid(provider, {}, 64); }, "null backend accepted");
  pmi::DeviceCache cache(provider, backend, 127);
  // Logical bytes fit, but the backend's 128-byte allocation charge does not.
  rejects([&] { cache.request(request(0, 17)); }, "backend padding escaped the device budget");
  std::vector<pmi::DevicePageRequest> invalid;
  auto value = request(); value.source.byte_length = 0; invalid.push_back(value);
  value = request(); value.source.absolute_offset = std::numeric_limits<uint64_t>::max() - 8; invalid.push_back(value);
  value = request(); value.source.shape.clear(); invalid.push_back(value);
  value = request(); value.source.shape = {0}; invalid.push_back(value);
  value = request(); value.source.shape = {std::numeric_limits<uint64_t>::max(), 2}; invalid.push_back(value);
  value = request(); value.source.byte_length = std::numeric_limits<uint64_t>::max() - 3;
  value.source.shape = {value.source.byte_length / 4}; invalid.push_back(value); // Allocation alignment overflow.
  value = request(); value.source.ggml_type = std::numeric_limits<uint32_t>::max(); invalid.push_back(value);
  value = request(); value.format = "unsupported"; invalid.push_back(value);
  value = request(); value.layout = "unsupported"; invalid.push_back(value);
  value = request(); value.source.byte_length = 8; invalid.push_back(value); // Shape/length disagreement.
  for (const auto& descriptor : invalid)
    rejects([&] { cache.request(descriptor); }, "invalid device descriptor accepted");
  state->throw_quote = true;
  rejects([&] { cache.request(request()); }, "quote exception ignored");
  state->throw_quote = false; state->zero_quote = true;
  rejects([&] { cache.request(request()); }, "zero backend allocation charge accepted");
  state->zero_quote = false; state->oversized_quote = true;
  rejects([&] { cache.request(request()); }, "oversized backend allocation charge accepted");
  state->oversized_quote = false;
  require(!provider.requests && !cache.stats().provider_requests && !state->allocation_calls &&
          !cache.stats().current_bytes,
          "descriptor/quote/capacity failure performed a host read or allocation");
  require(cache.stats().requested_bytes == std::numeric_limits<uint64_t>::max(),
          "requested-byte accounting wrapped instead of saturating");
  auto recovered = cache.request(request());
  expect_bytes(recovered);
  require(cache.stats().current_bytes == 64 && cache.stats().uploads == 1,
          "preflight failure poisoned subsequent admission");
}

void host_failures_preserve_victims(const Directory& fixture) {
  for (int mode = 0; mode < 5; ++mode) {
    Directory short_read_fixture;
    Provider provider(mode == 4 ? short_read_fixture : fixture);
    auto state = std::make_shared<DeviceState>();
    pmi::DeviceCache cache(provider, std::make_shared<Backend>(state), 64);
    uint64_t original_id;
    { auto original = cache.request(request()); original_id = allocation(original).id; }
    provider.throw_request = mode == 0;
    provider.invalid_future = mode == 1;
    provider.null_page = mode == 2;
    provider.substitute = mode == 3;
    provider.replacement = tensor(32, 2);
    if (mode == 4) std::filesystem::resize_file(short_read_fixture.shards[0], 34);
    rejects([&] { cache.request(request(32)); }, "invalid/failed host read was admitted");
    const auto stats = cache.stats();
    require(stats.entries == 1 && stats.current_bytes == 64 && !stats.evictions && stats.uploads == 1 &&
            stats.failures == 1 && state->allocation_calls == 1 && !state->frees,
            "host failure evicted a planned victim or allocated device storage");
    require(provider.last_page.expired(), "failed host validation retained a Page lease");
    provider.throw_request = provider.invalid_future = provider.null_page = provider.substitute = false;
    auto original = cache.request(request());
    require(allocation(original).id == original_id, "host failure removed the existing device page");
    expect_bytes(original);
  }
}

void allocation_and_upload_failures(const Directory& fixture) {
  for (int mode = 0; mode < 4; ++mode) {
    Provider provider(fixture);
    auto state = std::make_shared<DeviceState>();
    pmi::DeviceCache cache(provider, std::make_shared<Backend>(state), 64);
    state->throw_allocate = mode == 0;
    state->null_allocate = mode == 1;
    state->wrong_charge = mode == 2;
    state->throw_upload = mode == 3;
    state->during_upload = [&](const pmi::Page& page) {
      auto active = provider.last_page.lock();
      require(active && active.get() == &page, "upload error path lost its host lease");
    };
    rejects([&] { cache.request(request()); }, "failed device allocation/upload was admitted");
    const auto stats = cache.stats();
    require(!stats.entries && !stats.current_bytes && !stats.uploads && !stats.upload_bytes &&
            stats.failures == 1 && !state->live_bytes && state->allocations == state->frees,
            "allocation/upload failure leaked bytes or published a partial page");
    require(provider.last_page.expired(), "allocation/upload failure retained its host lease");
    if (mode == 3) require(stats.peak_bytes == 64, "failed upload allocation omitted from peak bytes");
    state->throw_allocate = state->null_allocate = state->wrong_charge = state->throw_upload = false;
    auto recovered = cache.request(request());
    expect_bytes(recovered);
    require(cache.stats().uploads == 1 && cache.stats().upload_bytes == 16,
            "failed page contaminated the retry or successful-upload counters");
  }

  Provider provider(fixture);
  auto state = std::make_shared<DeviceState>();
  pmi::DeviceCache cache(provider, std::make_shared<Backend>(state), 64);
  cache.request(request()); // Unleased victim must be retired before new allocation.
  state->throw_upload = true;
  rejects([&] { cache.request(request(32)); }, "replacement upload failure ignored");
  require(cache.stats().evictions == 1 && !cache.stats().entries && !state->live_bytes &&
          cache.stats().peak_bytes == 64 && state->peak_bytes == 64,
          "replacement failure exceeded capacity or leaked the victim/new allocation");
}

void callbacks_cannot_invalidate_admission(const Directory& fixture) {
  Provider provider(fixture);
  auto state = std::make_shared<DeviceState>();
  pmi::DeviceCache cache(provider, std::make_shared<Backend>(state), 64);
  std::weak_ptr<const pmi::DevicePage> weak;
  { auto original = cache.request(request()); weak = original; }
  pmi::DeviceLease promoted;
  provider.before_request = [&] { promoted = weak.lock(); require(static_cast<bool>(promoted), "victim died before host read"); };
  rejects([&] { cache.request(request(32)); }, "host callback promoted victim was evicted");
  require(promoted && cache.stats().entries == 1 && !cache.stats().evictions &&
          state->allocation_calls == 1 && state->live_bytes == 64,
          "victim lease promotion caused over-admission or eviction");
  expect_bytes(promoted);
  provider.before_request = {};
  promoted.reset();
  cache.clear();
  provider.before_request = [&] { cache.request(request(64)); };
  rejects([&] { cache.request(request(32)); }, "host callback request reentry was accepted");
  provider.before_request = {};
  state->before_quote = [&] { cache.request(request(64)); };
  rejects([&] { cache.request(request(32)); }, "backend callback request reentry was accepted");
  state->before_quote = {};
  provider.before_request = [&] { cache.clear(); };
  rejects([&] { cache.request(request(32)); }, "host callback clear reentry was accepted");
  provider.before_request = {};
  auto recovered = cache.request(request(32));
  expect_bytes(recovered);
  require(cache.stats().current_bytes == 64 && state->peak_bytes == 64,
          "callback failure poisoned cache state or exceeded the device budget");
}

void service_profile(const Directory& fixture) {
  for (bool enabled : {false, true}) {
    Provider provider(fixture);
    auto state = std::make_shared<DeviceState>();
    pmi::DeviceCache cache(provider, std::make_shared<Backend>(state), 64, enabled);
    cache.request(request());
    const auto first = cache.stats();
    require(first.profile_enabled == enabled, "device cache profiling flag lost");
    require(enabled ? first.provider_read_ms > 0 && first.allocation_ms > 0 && first.upload_ms > 0
                    : first.provider_read_ms == 0 && first.allocation_ms == 0 && first.upload_ms == 0 && first.eviction_ms == 0,
            "device cache service timers not gated by profiling");
    cache.request(request());
    auto after = cache.stats();
    require(after.provider_read_ms == first.provider_read_ms && after.allocation_ms == first.allocation_ms &&
            after.upload_ms == first.upload_ms && after.eviction_ms == first.eviction_ms,
            "cache hit timed nonexistent read/allocation/upload/eviction");
    provider.throw_request = true;
    rejects([&] { cache.request(request(32)); }, "profiled provider failure accepted");
    after = cache.stats();
    require(enabled ? after.provider_read_ms > first.provider_read_ms : after.provider_read_ms == 0,
            "failed provider service omitted from timing");
    require(after.allocation_ms == first.allocation_ms && after.upload_ms == first.upload_ms &&
            after.eviction_ms == first.eviction_ms && after.entries == 1,
            "failed provider read started device service or evicted cached page");
    provider.throw_request = false;
    state->throw_upload = true;
    rejects([&] { cache.request(request(32)); }, "profiled upload failure accepted");
    after = cache.stats();
    require(enabled ? after.upload_ms > first.upload_ms && after.allocation_ms > first.allocation_ms && after.eviction_ms > first.eviction_ms
                    : after.upload_ms == 0 && after.allocation_ms == 0 && after.eviction_ms == 0,
            "failed upload service timing incorrect");
    require(after.failures == 2 && !after.entries && !after.current_bytes,
            "service profiling changed failure cleanup or charges");
  }
}

void seed_and_lookup(const Directory& fixture) {
  Provider provider(fixture);
  auto state = std::make_shared<DeviceState>();
  pmi::DeviceCache cache(provider, std::make_shared<Backend>(state), 64);
  require(!cache.lookup(request()), "absent lookup fetched weights");
  require(!provider.requests && !state->allocation_calls, "lookup started IO/allocation");
  auto seed = cache.seed(request());
  expect_bytes(seed);
  const auto original = allocation(seed).id;
  require(!cache.seed(request(32)), "seed evicted live resident weights");
  require(provider.requests == 1 && state->allocation_calls == 1 && !cache.stats().evictions,
          "refused seed started IO or changed residency");
  auto hit = cache.lookup(request());
  require(hit && allocation(hit).id == original, "resident lookup changed payload");
  seed.reset(); hit.reset();
  require(!cache.seed(request(32)), "seed evicted an unleased resident page");
  require(!cache.lookup(request(32)) && provider.requests == 1 && !cache.stats().failures,
          "frozen miss should be empty, without failure or IO");
  hit = cache.lookup(request()); cache.clear();
  require(cache.stats().current_bytes == 64 && cache.stats().active_leases == 1,
          "clear destroyed a pinned hit lease");
  expect_bytes(hit); hit.reset(); cache.clear();
  require(!cache.stats().current_bytes && !state->live_bytes, "seed/lookup leaked charges");
}
void progress_observer(const Directory& fixture) {
  Provider provider(fixture);auto state=std::make_shared<DeviceState>();
  auto backend=std::make_shared<Backend>(state);std::vector<std::string> phases;
  pmi::DeviceCache cache(provider,backend,64,false,[&](const pmi::DeviceCacheEvent& e){
    require(e.bytes==16 && e.capacity_bytes==64,"observer byte coordinates");phases.push_back(e.phase);
  });
  auto page=cache.request(request());
  require(phases==std::vector<std::string>{"cache_request_begin","provider_begin","provider_end",
    "eviction_begin","eviction_end","allocation_begin","allocation_end","upload_begin","upload_end","cache_request_end"},"observer stage order");
  phases.clear();auto hit=cache.request(request());
  require(phases==std::vector<std::string>{"cache_request_begin","cache_hit"},"observed hit performed IO");
  page.reset();hit.reset();cache.clear();
  pmi::DeviceCache failing(provider,backend,64,false,[](const pmi::DeviceCacheEvent& e){
    if(std::string(e.phase)=="upload_begin")throw std::runtime_error("injected telemetry failure");
  });
  rejects([&]{failing.request(request());},"observer failure not propagated");
  require(failing.stats().failures==1 && !failing.stats().current_bytes && !state->live_bytes,"observer failure leaked allocation");
}
void transient_upload(const Directory& fixture) {
  pmi::FilePageProvider provider(fixture.shards,128);
  auto state=std::make_shared<DeviceState>();auto backend=std::make_shared<Backend>(state);
  state->during_upload=[&](const pmi::Page& page) {
    require(page.size()==16 && provider.stats().current_bytes==64 && provider.stats().active_leases==1,
            "transient payload not charged/alive during synchronized upload");
  };
  pmi::DeviceCache cache(provider,backend,64,false,{},true);
  auto page=cache.request(request());
  require(provider.stats().current_bytes==0 && provider.stats().entries==0,"upload payload retained");
  auto hit=cache.request(request());require(provider.stats().misses==1,"device hit read host");
  page.reset();hit.reset();cache.clear();state->throw_upload=true;
  rejects([&]{cache.request(request());},"failed transient upload accepted");
  require(provider.stats().current_bytes==0 && provider.stats().active_leases==0 && state->live_bytes==0,
          "failed transient upload leaked host/device bytes");
}
void direct_upload(const Directory& fixture) {
  pmi::FilePageProvider provider(fixture.shards,65536);
  auto state=std::make_shared<DeviceState>();auto backend=std::make_shared<Backend>(state);
  state->during_upload=[&](const pmi::Page& page) {
    require(page.size()==16 && provider.stats().current_bytes==64 && provider.stats().active_leases==1,
            "direct staging survived upload or payload lease was lost");
  };
  pmi::DeviceCache cache(provider,backend,64,false,{},true,pmi::HostReadPolicy::DirectPreferred);
  auto page=cache.request(request(1));expect_bytes(page);
  require(provider.stats().current_bytes==0 && provider.stats().entries==0,"direct payload retained");
  const auto before=provider.io_stats();
  auto hit=cache.request(request(1));
  require(provider.io_stats().direct_requests==before.direct_requests && provider.stats().misses==1,
          "device hit performed a direct source read");
  require(before.direct_completed+before.fallback_count==1,"direct upload policy was bypassed");
  page.reset();hit.reset();cache.clear();state->throw_upload=true;
  rejects([&]{cache.request(request(1));},"failed direct upload accepted");
  require(provider.stats().current_bytes==0 && provider.stats().active_leases==0 &&
          provider.stats().entries==0 && state->live_bytes==0 && !cache.stats().entries,
          "failed direct upload leaked host/device bytes or cached a partial upload");
  state->throw_upload=false;page=cache.request(request(1));expect_bytes(page);
  std::cout<<"direct upload lease/failure/retry passed; native="<<(before.direct_completed>0)<<'\n';
}
void retained_direct_reuse(const Directory& fixture) {
  pmi::FilePageProvider provider(fixture.shards,65536);
  auto state=std::make_shared<DeviceState>();auto backend=std::make_shared<Backend>(state);
  state->during_upload=[&](const pmi::Page&) {
    require(provider.stats().active_leases==1 && provider.stats().current_bytes<=192,
            "retained upload double-counted payload or kept direct staging alive");
  };
  pmi::DeviceCache cache(provider,backend,64,false,{},false,pmi::HostReadPolicy::DirectPreferred);
  const auto first=request(1),second=request(65),third=request(129);
  {auto page=cache.request(first);expect_bytes(page);}
  const auto staging=provider.io_stats().staging_peak_bytes;
  require(provider.stats().entries==1 && provider.stats().current_bytes==64 && !provider.stats().active_leases,
          "retained source did not survive upload lease release");
  {auto page=cache.request(second);expect_bytes(page);}
  const auto io=provider.io_stats().direct_requests;
  {auto page=cache.request(first);expect_bytes(page);}
  require(provider.io_stats().direct_requests==io && provider.stats().hits==1 && cache.stats().uploads==3,
          "device-evicted page was not reuploaded from retained host storage");
  auto retained=provider.request(first.source).get();
  auto transient=provider.request_direct(first.source,true,true).get();
  require(retained->data()==transient->data() && provider.stats().current_bytes==128,
          "transient hit duplicated retained payload instead of sharing it");
  retained.reset();transient.reset();
  cache.clear();state->throw_upload=true;
  rejects([&]{cache.request(third);},"failed retained upload accepted");
  require(provider.stats().entries==3 && provider.stats().current_bytes==192 && !provider.stats().active_leases &&
          !cache.stats().entries && !state->live_bytes,"retained failure lost valid host bytes or leaked upload lease/device storage");
  const auto before_retry=provider.io_stats().direct_requests;state->throw_upload=false;
  {auto page=cache.request(third);expect_bytes(page);}
  require(provider.io_stats().direct_requests==before_retry,"retry reread a valid retained source");

  // This minimum quota admits one payload plus staging, forcing host eviction
  // on the next miss. On an unsupported filesystem staging is zero.
  pmi::FilePageProvider bounded(fixture.shards,staging+64);
  auto bounded_state=std::make_shared<DeviceState>();
  pmi::DeviceCache limited(bounded,std::make_shared<Backend>(bounded_state),64,false,{},false,pmi::HostReadPolicy::DirectPreferred);
  for(const auto& source:{first,second,first}){auto page=limited.request(source);expect_bytes(page);}
  require(bounded.stats().misses==3 && bounded.stats().hits==0 && bounded.stats().evictions==2 &&
          bounded.stats().current_bytes==64 && bounded.stats().peak_bytes<=staging+64,
          "small reusable host quota failed eviction/reload or exceeded payload-plus-staging bound");
  std::cout<<"retained direct reuse/shared transient hit/upload-failure retry/host eviction passed; native="
           <<(provider.io_stats().direct_completed>0)<<'\n';
}
} // namespace

int main() {
  try {
    Directory fixture;
    exact_identity(fixture);
    eviction_leases_and_reload(fixture);
    lru_and_variable_charges(fixture);
    independent_lifetimes(fixture);
    preflight_errors(fixture);
    host_failures_preserve_victims(fixture);
    allocation_and_upload_failures(fixture);
    callbacks_cannot_invalidate_admission(fixture);
    service_profile(fixture);
    seed_and_lookup(fixture);
    progress_observer(fixture);
    transient_upload(fixture);
    direct_upload(fixture);
    retained_direct_reuse(fixture);
    std::cout << "device cache tests passed (mock device, exact keys, budgets, leases, reload, failures, callbacks)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
