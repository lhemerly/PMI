#include "pmi/retention.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F&& action) {
  bool failed = false;
  try { action(); } catch (const std::exception&) { failed = true; }
  require(failed, "expected retention failure");
}

struct Directory {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("pmi-retention-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::vector<std::filesystem::path> shards;
  Directory() {
    std::filesystem::create_directories(path);
    for (int shard = 0; shard < 2; ++shard) {
      shards.push_back(path / ("shard-" + std::to_string(shard) + ".bin"));
      std::ofstream out(shards.back(), std::ios::binary);
      for (int i = 0; i < 1024; ++i) out.put(static_cast<char>((i + shard * 17) % 256));
      require(static_cast<bool>(out), "cannot write retention fixture");
    }
  }
  ~Directory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};

pmi::TensorInfo range(uint64_t offset, uint64_t length = 4, uint32_t shard = 0) {
  pmi::TensorInfo tensor;
  tensor.absolute_offset = offset;
  tensor.byte_length = length;
  tensor.shard_id = shard;
  return tensor;
}

// Delegate actual storage/pins to Rust; only inject failures before a side effect.
// Counters prove the helper performs no additional requests or counted pins.
class Provider final : public pmi::PageProvider {
 public:
  using Key = std::tuple<uint32_t, uint64_t, uint64_t>;
  explicit Provider(const Directory& fixture, size_t capacity = 512)
      : file(fixture.shards, capacity) {}
  pmi::PageFuture request(const pmi::TensorInfo& tensor) override {
    ++requests;
    return file.request(tensor);
  }
  pmi::PageFuture prefetch(const pmi::TensorInfo& tensor) override {
    ++prefetches;
    return file.prefetch(tensor);
  }
  bool pin(const pmi::TensorInfo& tensor) override {
    ++pin_attempts;
    if (pin_attempts == throw_pin_at) throw std::runtime_error("injected pin exception");
    if (pin_attempts == fail_pin_at) return false;
    if (!file.pin(tensor)) return false;
    ++counts[key(tensor)];
    return true;
  }
  bool release(const pmi::TensorInfo& tensor) override {
    ++release_attempts;
    if (release_attempts == fail_release_at) return false;
    if (!file.release(tensor)) return false;
    const auto it = counts.find(key(tensor));
    require(it != counts.end() && it->second > 0, "released unowned counted pin");
    if (--it->second == 0) counts.erase(it);
    return true;
  }
  size_t count(const pmi::TensorInfo& tensor) const {
    const auto it = counts.find(key(tensor));
    return it == counts.end() ? 0 : it->second;
  }
  static Key key(const pmi::TensorInfo& tensor) {
    return {tensor.shard_id, tensor.absolute_offset, tensor.byte_length};
  }
  pmi::FilePageProvider file;
  size_t requests{}, prefetches{}, pin_attempts{}, release_attempts{};
  size_t fail_pin_at{}, throw_pin_at{}, fail_release_at{};
  std::map<Key, size_t> counts;
};

void load(Provider& provider, pmi::ProtectedPageRetention& retention, const pmi::TensorInfo& tensor) {
  auto lease = provider.request(tensor).get();
  const auto requests = provider.requests;
  const auto bytes = provider.file.stats().io_bytes;
  retention.observe_loaded(tensor);
  require(provider.requests == requests && !provider.prefetches && provider.file.stats().io_bytes == bytes,
          "retention loaded or prefetched data");
  require(lease->size() == tensor.byte_length, "retention invalidated active lease");
}

void rotation_and_overlap(const Directory& fixture) {
  Provider provider(fixture, 256);
  pmi::ProtectedPageRetention retention(provider, 192);
  const auto a = range(0), b = range(16), c = range(32), service = range(48);
  retention.begin_group(7, {a, b});
  require(retention.group_active() && retention.active_group_id() == 7, "active group identity");
  load(provider, retention, a);
  load(provider, retention, b);
  load(provider, retention, b);
  retention.finish_group(7);
  require(provider.count(a) == 1 && provider.count(b) == 1 && provider.pin_attempts == 2,
          "duplicate observation acquired duplicate pins");
  retention.begin_group(7, {b, c});
  load(provider, retention, b);
  load(provider, retention, c);
  require(provider.count(a) == 1 && retention.protected_bytes() == 192,
          "prior selection retired before complete group");
  retention.finish_group(7);
  require(provider.count(a) == 0 && provider.count(b) == 1 && provider.count(c) == 1,
          "routing rotation did not retain overlap and replace old selection");
  require(retention.protected_bytes() == 128 && retention.peak_protected_bytes() == 192,
          "turnover byte accounting");
  provider.request(service).get();
  provider.request(range(64)).get(); // Force an unprotected page eviction.
  const auto read_before = provider.file.stats().io_bytes;
  provider.request(b).get();
  provider.request(c).get();
  require(provider.file.stats().io_bytes == read_before, "protected pages were reread after service pressure");
  retention.clear();
  require(provider.counts.empty() && !retention.protected_bytes(), "clear left rotation pins");
}

void groups_and_exact_identity(const Directory& fixture) {
  Provider provider(fixture);
  pmi::ProtectedPageRetention retention(provider, 256);
  auto a = range(0, 1);
  auto alias = a; alias.name = "different model label"; alias.shape = {999}; alias.offset = 123;
  const auto different_length = range(0, 2), other_shard = range(0, 1, 1);
  retention.begin_group(1, {a, alias});
  load(provider, retention, a);
  retention.finish_group(1);
  retention.begin_group(2, {alias, different_length, other_shard});
  load(provider, retention, alias);
  load(provider, retention, different_length);
  load(provider, retention, other_shard);
  retention.finish_group(2);
  require(retention.protected_bytes() == 192 && provider.pin_attempts == 3,
          "exact range identity or cross-group deduplication failed");
  retention.begin_group(1, {});
  retention.finish_group(1);
  require(provider.count(a) == 1 && retention.protected_bytes() == 192,
          "removing one group released another group's shared pin");
  retention.begin_group(2, {other_shard});
  load(provider, retention, other_shard);
  retention.finish_group(2);
  require(provider.count(a) == 0 && provider.count(different_length) == 0 && provider.count(other_shard) == 1,
          "group replacement leaked obsolete exact ranges");
}

void budget_and_protocol_failures(const Directory& fixture) {
  Provider provider(fixture);
  pmi::ProtectedPageRetention retention(provider, 128);
  const auto a = range(0, 1), b = range(64, 65);
  retention.begin_group(4, {a});
  load(provider, retention, a);
  retention.finish_group(4);
  const auto requests = provider.requests, pins = provider.pin_attempts, releases = provider.release_attempts;
  rejects([&] { retention.begin_group(4, {b}); }); // 64 old + 128 next > 128.
  require(!retention.group_active() && retention.protected_bytes() == 64 && provider.count(a) == 1,
          "over-budget begin changed committed state");
  require(provider.requests == requests && provider.pin_attempts == pins && provider.release_attempts == releases,
          "over-budget begin touched provider");
  rejects([&] { retention.begin_group(4, {range(0, 0)}); });
  rejects([&] { retention.begin_group(4, {range(0, std::numeric_limits<uint64_t>::max())}); });
  rejects([&] { retention.begin_group(4, {range(std::numeric_limits<uint64_t>::max(), 1)}); });
  retention.begin_group(4, {range(16)});
  rejects([&] { retention.begin_group(5, {}); });
  rejects([&] { retention.finish_group(5); });
  rejects([&] { retention.finish_group(4); }); // Desired page has not been observed.
  require(retention.group_active() && provider.count(a) == 1, "protocol failure altered prior pins");
  load(provider, retention, range(32)); // Nondesignated prefill/service page is ignored.
  require(provider.pin_attempts == pins, "nondesignated page acquired a pin");
  retention.cancel_group();
  require(provider.count(a) == 1 && !retention.group_active(), "cancel discarded committed pins");
  rejects([&] { retention.active_group_id(); });
  retention.clear();
  pmi::ProtectedPageRetention disabled(provider, 0);
  disabled.begin_group(0, {}); disabled.finish_group(0);
  rejects([&] { disabled.begin_group(0, {a}); });
}

void pin_failure_rollback(const Directory& fixture, bool throws) {
  Provider provider(fixture);
  pmi::ProtectedPageRetention retention(provider, 256);
  const auto a = range(0), b = range(16), c = range(32);
  retention.begin_group(9, {a}); load(provider, retention, a); retention.finish_group(9);
  retention.begin_group(9, {b, c});
  load(provider, retention, b);
  if (throws) provider.throw_pin_at = provider.pin_attempts + 1;
  else provider.fail_pin_at = provider.pin_attempts + 1;
  auto lease = provider.request(c).get();
  rejects([&] { retention.observe_loaded(c); });
  require(!retention.group_active() && retention.protected_bytes() == 64 && provider.count(a) == 1 &&
          !provider.count(b) && !provider.count(c), "failed pin did not roll back only transaction pins");
  retention.begin_group(9, {a}); load(provider, retention, a); retention.finish_group(9);
}

void release_failure_tracking(const Directory& fixture) {
  Provider provider(fixture);
  pmi::ProtectedPageRetention retention(provider, 256);
  const auto a = range(0), b = range(16), c = range(32);
  retention.begin_group(3, {a, b});
  load(provider, retention, a); load(provider, retention, b); retention.finish_group(3);
  require(provider.pin(a), "external pin acquisition");
  retention.begin_group(3, {c}); load(provider, retention, c);
  provider.fail_release_at = provider.release_attempts + 2; // a succeeds, b fails.
  rejects([&] { retention.finish_group(3); });
  require(!retention.group_active() && retention.protected_bytes() == 128 && provider.count(a) == 1 &&
          provider.count(b) == 1 && provider.count(c) == 1, "partial release lost owned/external pin accounting");
  rejects([&] { retention.begin_group(3, {}); });
  rejects([&] { retention.cancel_group(); });
  provider.fail_release_at = provider.release_attempts + 1; // clear fails b, still releases c.
  rejects([&] { retention.clear(); });
  require(retention.protected_bytes() == 64 && provider.count(a) == 1 && provider.count(b) == 1 &&
          !provider.count(c), "failed clear did not retain only outstanding owned pins");
  retention.clear();
  require(!retention.protected_bytes() && provider.count(a) == 1 && !provider.count(b),
          "retry double-released an external pin or leaked failed pin");
  require(provider.release(a) && provider.counts.empty(), "external pin was not preserved");
}

void pin_and_rollback_failure(const Directory& fixture) {
  Provider provider(fixture);
  pmi::ProtectedPageRetention retention(provider, 192);
  const auto a = range(0), b = range(16), c = range(32);
  retention.begin_group(8, {a}); load(provider, retention, a); retention.finish_group(8);
  retention.begin_group(8, {b, c}); load(provider, retention, b);
  provider.fail_pin_at = provider.pin_attempts + 1;
  provider.fail_release_at = provider.release_attempts + 1;
  auto lease = provider.request(c).get();
  rejects([&] { retention.observe_loaded(c); });
  require(!retention.group_active() && retention.protected_bytes() == 128 &&
          provider.count(a) == 1 && provider.count(b) == 1 && !provider.count(c),
          "combined pin/rollback failure lost outstanding owned pins");
  rejects([&] { retention.begin_group(8, {}); });
  retention.clear();
  require(!retention.protected_bytes() && provider.counts.empty(), "combined failure could not be cleared");
}

void early_cleanup_and_failed_cancel(const Directory& fixture) {
  Provider provider(fixture);
  const auto a = range(0), b = range(16), c = range(32);
  {
    pmi::ProtectedPageRetention retention(provider, 256);
    retention.begin_group(1, {a}); load(provider, retention, a); retention.finish_group(1);
    retention.begin_group(1, {b, c}); load(provider, retention, b);
    provider.fail_release_at = provider.release_attempts + 1;
    rejects([&] { retention.cancel_group(); });
    require(retention.protected_bytes() == 128 && provider.count(a) == 1 && provider.count(b) == 1,
            "failed cancel lost tracked committed or new pins");
    rejects([&] { retention.begin_group(2, {}); });
    rejects([&] { retention.cancel_group(); });
  } // Destructor releases both committed and outstanding failed-cleanup pins.
  require(provider.counts.empty() && provider.file.stats().pinned_entries == 0,
          "destructor leaked pins after incomplete group/cancellation");
  {
    pmi::ProtectedPageRetention retention(provider, 128);
    retention.begin_group(1, {a, b}); load(provider, retention, a);
  }
  require(provider.counts.empty(), "early destruction retained partially loaded selection");
}
} // namespace

int main() {
  try {
    Directory fixture;
    rotation_and_overlap(fixture);
    groups_and_exact_identity(fixture);
    budget_and_protocol_failures(fixture);
    pin_failure_rollback(fixture, false);
    pin_failure_rollback(fixture, true);
    release_failure_tracking(fixture);
    pin_and_rollback_failure(fixture);
    early_cleanup_and_failed_cancel(fixture);
    std::cout << "retention tests passed (rotation, exact overlap, budgets, pins, cleanup, no extra reads)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
