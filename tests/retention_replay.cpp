// Offline metadata-only policy replay. This uses the production retention helper
// with a simulated byte-LRU provider; it neither opens weights nor runs kernels.
#include "pmi/retention.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
using Key = std::tuple<uint32_t, uint64_t, uint64_t>;
Key key(const pmi::TensorInfo& t) { return {t.shard_id, t.absolute_offset, t.byte_length}; }
uint64_t charge(uint64_t bytes) {
  require(bytes && bytes <= UINT64_MAX - 63, "invalid replay page size");
  return (bytes + 63) & ~uint64_t{63};
}
struct Counters {
  uint64_t hits{}, misses{}, evictions{}, io{}, current{}, peak{};
  bool operator==(const Counters& b) const {
    return std::tie(hits, misses, evictions, io, current, peak) ==
           std::tie(b.hits, b.misses, b.evictions, b.io, b.current, b.peak);
  }
};
void counters_json(const Counters& c) {
  std::cout << "{\"hits\":" << c.hits << ",\"misses\":" << c.misses << ",\"evictions\":" << c.evictions
            << ",\"io_bytes\":" << c.io << ",\"current_bytes\":" << c.current << ",\"peak_bytes\":" << c.peak << '}';
}

class MetadataCache final : public pmi::PageProvider {
  struct Entry { uint64_t charge, tick, pins{}, leases{}; };
  uint64_t capacity_, tick_{};
  std::map<Key, Entry> entries_;
 public:
  Counters counters;
  explicit MetadataCache(uint64_t capacity) : capacity_(capacity) {}
  // A metadata lease keeps a simulated entry ineligible during observation.
  // No Page payload is manufactured or required by ProtectedPageRetention.
  class Lease {
    MetadataCache& cache_;
    Key key_;
   public:
    Lease(MetadataCache& cache, Key range) : cache_(cache), key_(range) {}
    Lease(const Lease&) = delete;
    ~Lease() { --cache_.entries_.at(key_).leases; }
  };
  Lease load(const pmi::TensorInfo& tensor) {
    const auto range = key(tensor);
    auto found = entries_.find(range);
    if (found != entries_.end()) {
      ++counters.hits;
      found->second.tick = ++tick_;
      ++found->second.leases;
      return Lease(*this, range);
    }
    const auto bytes = charge(tensor.byte_length);
    require(bytes <= capacity_, "replay page exceeds capacity");
    while (bytes > capacity_ - counters.current) {
      auto victim = entries_.end();
      for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->second.pins || it->second.leases) continue;
        if (victim == entries_.end() || std::tie(it->second.tick, it->first) <
            std::tie(victim->second.tick, victim->first)) victim = it;
      }
      require(victim != entries_.end(), "replay capacity blocked by live pins/leases");
      counters.current -= victim->second.charge;
      entries_.erase(victim);
      ++counters.evictions;
    }
    entries_.emplace(range, Entry{bytes, ++tick_, 0, 1});
    ++counters.misses;
    require(tensor.byte_length <= UINT64_MAX - counters.io, "replay I/O sum overflow");
    counters.io += tensor.byte_length;
    counters.current += bytes;
    counters.peak = std::max(counters.peak, counters.current);
    return Lease(*this, range);
  }
  pmi::PageFuture request(const pmi::TensorInfo&) override { throw std::logic_error("payload reads forbidden in metadata replay"); }
  pmi::PageFuture prefetch(const pmi::TensorInfo&) override { throw std::logic_error("payload prefetch forbidden in metadata replay"); }
  bool pin(const pmi::TensorInfo& tensor) override {
    auto it = entries_.find(key(tensor));
    if (it == entries_.end()) return false;
    require(it->second.pins == 0, "helper acquired a duplicate global pin");
    ++it->second.pins;
    return true;
  }
  bool release(const pmi::TensorInfo& tensor) override {
    auto it = entries_.find(key(tensor));
    if (it == entries_.end() || !it->second.pins) return false;
    --it->second.pins;
    return true;
  }
  uint64_t protected_bytes() const {
    uint64_t bytes = 0;
    for (const auto& entry : entries_) if (entry.second.pins) bytes += entry.second.charge;
    return bytes;
  }
  std::set<Key> protected_keys() const {
    std::set<Key> keys;
    for (const auto& entry : entries_) if (entry.second.pins) keys.insert(entry.first);
    return keys;
  }
};

struct Event {
  std::string kind;
  uint64_t group{};
  pmi::TensorInfo tensor;
  std::vector<pmi::TensorInfo> next;
};
std::string line() {
  std::string text;
  require(static_cast<bool>(std::getline(std::cin, text)) && text.size() <= 256, "invalid or missing replay record");
  return text;
}
void end(std::istringstream& row) {
  require(!row.fail(), "invalid replay fields");
  std::string extra;
  require(!(row >> extra), "unexpected replay fields");
}
pmi::TensorInfo range(std::istringstream& row) {
  pmi::TensorInfo tensor;
  row >> tensor.shard_id >> tensor.absolute_offset >> tensor.byte_length;
  end(row);
  charge(tensor.byte_length);
  require(tensor.absolute_offset <= UINT64_MAX - tensor.byte_length, "replay range overflow");
  return tensor;
}
} // namespace

int main() {
  try {
    require(line() == "PMI_RETENTION_REPLAY_V1", "unsupported replay stream");
    uint64_t capacity{}, reserve{}, expected_policy_hits{}, expected_policy_io{};
    std::string label;
    std::istringstream config(line()); config >> label >> capacity >> reserve; end(config);
    require(label == "config" && reserve && reserve < capacity, "invalid replay capacity/reserve");
    Counters expected;
    std::istringstream expectation(line());
    expectation >> label >> expected.hits >> expected.misses >> expected.evictions >> expected.io >> expected.current >>
        expected.peak >> expected_policy_hits >> expected_policy_io;
    end(expectation); require(label == "expect", "missing baseline expectation");
    std::vector<Event> events;
    for (;;) {
      std::istringstream row(line()); Event event; row >> event.kind;
      if (event.kind == "end") { end(row); break; }
      require(events.size() < 10000, "too many replay events");
      if (event.kind == "load" || event.kind == "service") event.tensor = range(row);
      else if (event.kind == "begin") {
        size_t count{}; row >> event.group >> count; end(row);
        require(count <= 64, "too many protected ranges in replay group");
        for (size_t i = 0; i < count; ++i) {
          std::istringstream wanted(line()); wanted >> label;
          require(label == "want", "missing desired replay range");
          event.next.push_back(range(wanted));
        }
      } else if (event.kind == "finish") { row >> event.group; end(row); }
      else throw std::runtime_error("unknown replay event");
      events.push_back(std::move(event));
    }
    require(std::cin.peek() == std::char_traits<char>::eof(), "trailing replay data");
    MetadataCache baseline(capacity);
    for (const auto& event : events)
      if (event.kind == "load" || event.kind == "service") { auto lease = baseline.load(event.tensor); }
    require(baseline.counters == expected, "baseline metadata LRU does not match all six observed counters");

    // Policy execution starts only after the counter-exact baseline succeeds.
    MetadataCache cache(capacity);
    pmi::ProtectedPageRetention retention(cache, capacity - reserve);
    uint64_t groups = 0;
    std::map<uint64_t, std::set<Key>> expected_groups;
    for (const auto& event : events) {
      if (event.kind == "begin") {
        retention.begin_group(event.group, event.next);
        auto& expected_keys = expected_groups[event.group];
        expected_keys.clear();
        for (const auto& tensor : event.next) expected_keys.insert(key(tensor));
      }
      else if (event.kind == "finish") { retention.finish_group(event.group); ++groups; }
      else {
        auto lease = cache.load(event.tensor);
        if (event.kind == "load") retention.observe_loaded(event.tensor);
      }
      require(cache.protected_bytes() == retention.protected_bytes(), "helper and simulated pin accounting disagree");
      require(retention.protected_bytes() <= capacity - reserve, "protected turnover consumed service reserve");
      require(cache.counters.current <= capacity, "simulated total cache exceeded original capacity");
    }
    require(!retention.group_active(), "replay ended with unfinished group");
    require(cache.counters.hits == expected_policy_hits && cache.counters.io == expected_policy_io,
            "actual retention helper differs from independently predicted top-four savings");
    std::set<Key> final_keys;
    for (const auto& group : expected_groups) final_keys.insert(group.second.begin(), group.second.end());
    require(cache.protected_keys() == final_keys, "final protected keys differ from latest group selections");
    const auto retained = retention.protected_bytes(), peak = retention.peak_protected_bytes();
    retention.clear();
    require(!retention.protected_bytes() && !cache.protected_bytes(), "replay cleanup left owned pins");
    std::cout << "{\"baseline_matches_observed\":true,\"baseline\":"; counters_json(baseline.counters);
    std::cout << ",\"retention\":"; counters_json(cache.counters);
    std::cout << ",\"capacity_bytes\":" << capacity << ",\"service_reserve_bytes\":" << reserve
              << ",\"group_commits\":" << groups << ",\"final_protected_pages\":" << final_keys.size()
              << ",\"protected_bytes_before_clear\":" << retained
              << ",\"peak_protected_bytes\":" << peak << ",\"all_pins_cleared\":true}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
