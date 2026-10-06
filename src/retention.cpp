#include "pmi/retention.hpp"

#include <algorithm>
#include <exception>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace pmi {
namespace {
struct Range {
  uint32_t shard;
  uint64_t offset;
  uint64_t length;
  bool operator<(const Range& other) const noexcept {
    return std::tie(shard, offset, length) < std::tie(other.shard, other.offset, other.length);
  }
};

Range range_of(const TensorInfo& tensor) noexcept {
  return {tensor.shard_id, tensor.absolute_offset, tensor.byte_length};
}

TensorInfo descriptor(const Range& range) {
  TensorInfo tensor;
  tensor.shard_id = range.shard;
  tensor.absolute_offset = range.offset;
  tensor.byte_length = range.length;
  return tensor;
}

uint64_t charge(const Range& range) {
  const auto maximum = std::numeric_limits<uint64_t>::max();
  if (!range.length || range.offset > maximum - range.length || range.length > maximum - 63)
    throw std::invalid_argument("PMI retention: invalid page range or aligned size overflow");
  return (range.length + 63) & ~uint64_t{63};
}
} // namespace

struct ProtectedPageRetention::Impl {
  using Ranges = std::set<Range>;
  struct PendingPage { uint64_t bytes; bool observed{false}; };
  struct Pending {
    uint64_t id;
    std::map<Range, PendingPage> next;
  };

  PageProvider& provider;
  uint64_t limit;
  uint64_t bytes{};
  uint64_t peak{};
  bool cleanup_failed{};
  std::map<uint64_t, Ranges> groups;
  std::map<Range, uint64_t> owned;
  std::optional<Pending> pending;

  Impl(PageProvider& p, uint64_t capacity) : provider(p), limit(capacity) {}

  bool committed(const Range& range) const {
    for (const auto& group : groups)
      if (group.second.count(range)) return true;
    return false;
  }

  // Bookkeeping is erased only after a successful release. Continue through
  // failures so clear/destruction still attempts every independent owned pin.
  bool release_pins(bool all) noexcept {
    bool success = true;
    for (auto it = owned.begin(); it != owned.end();) {
      if (!all && committed(it->first)) { ++it; continue; }
      bool released = false;
      try { released = provider.release(descriptor(it->first)); } catch (...) {}
      if (released) {
        bytes -= it->second;
        it = owned.erase(it);
      } else {
        success = false;
        ++it;
      }
    }
    return success;
  }

  void ensure_ready() const {
    if (cleanup_failed)
      throw std::runtime_error("PMI retention: prior cleanup failed; clear owned pins before continuing");
  }

  void cancel() {
    pending.reset();
    cleanup_failed = !release_pins(false);
    if (cleanup_failed)
      throw std::runtime_error("PMI retention: cancellation could not release every new pin; call clear");
  }
};

ProtectedPageRetention::ProtectedPageRetention(PageProvider& provider, uint64_t limit)
    : impl_(std::make_unique<Impl>(provider, limit)) {}

ProtectedPageRetention::~ProtectedPageRetention() noexcept {
  impl_->release_pins(true);
}

void ProtectedPageRetention::begin_group(uint64_t id, const std::vector<TensorInfo>& next) {
  auto& state = *impl_;
  state.ensure_ready();
  if (state.pending) throw std::logic_error("PMI retention: a group is already active");
  Impl::Pending prepared{id, {}};
  uint64_t union_bytes = state.bytes;
  for (const auto& tensor : next) {
    const auto range = range_of(tensor);
    const auto size = charge(range);
    if (!prepared.next.emplace(range, Impl::PendingPage{size, false}).second) continue;
    if (!state.owned.count(range)) {
      if (union_bytes > state.limit || size > state.limit - union_bytes)
        throw std::runtime_error("PMI retention: old/new protected union exceeds configured limit");
      union_bytes += size;
    }
  }
  // All potentially allocating preparation and budget checks precede any pins.
  state.pending.emplace(std::move(prepared));
}

void ProtectedPageRetention::observe_loaded(const TensorInfo& tensor) {
  auto& state = *impl_;
  state.ensure_ready();
  if (!state.pending) throw std::logic_error("PMI retention: no active group for loaded page");
  const auto range = range_of(tensor);
  const auto desired = state.pending->next.find(range);
  if (desired == state.pending->next.end()) return;
  try {
    if (!state.owned.count(range)) {
      // Allocate the ownership record before pinning; success then requires no
      // allocation. A failed pin cannot leave an untracked counted pin.
      const auto inserted = state.owned.emplace(range, desired->second.bytes).first;
      try {
        if (!state.provider.pin(descriptor(range)))
          throw std::runtime_error("PMI retention: provider could not pin loaded page");
      } catch (...) {
        state.owned.erase(inserted);
        throw;
      }
      state.bytes += desired->second.bytes;
      state.peak = std::max(state.peak, state.bytes);
    }
    desired->second.observed = true;
  } catch (...) {
    const auto failure = std::current_exception();
    state.cancel();
    std::rethrow_exception(failure);
  }
}

void ProtectedPageRetention::finish_group(uint64_t id) {
  auto& state = *impl_;
  state.ensure_ready();
  if (!state.pending || state.pending->id != id)
    throw std::logic_error("PMI retention: finishing group does not match active group");
  Impl::Ranges replacement;
  for (const auto& item : state.pending->next) {
    if (!item.second.observed)
      throw std::logic_error("PMI retention: desired page was not observed before group completion");
    replacement.insert(item.first);
  }
  // Allocate the new committed group before releasing anything. Once swapped,
  // old pins may be retired; a failed release cannot roll back earlier releases.
  auto group = state.groups.try_emplace(id).first;
  group->second.swap(replacement);
  if (group->second.empty()) state.groups.erase(group);
  state.pending.reset();
  state.cleanup_failed = !state.release_pins(false);
  if (state.cleanup_failed)
    throw std::runtime_error("PMI retention: group committed but obsolete pin cleanup failed; call clear");
}

void ProtectedPageRetention::cancel_group() {
  impl_->ensure_ready();
  impl_->cancel();
}

void ProtectedPageRetention::clear() {
  auto& state = *impl_;
  state.pending.reset();
  state.groups.clear();
  state.cleanup_failed = !state.release_pins(true);
  if (state.cleanup_failed)
    throw std::runtime_error("PMI retention: could not release every owned pin; clear may be retried");
}

bool ProtectedPageRetention::group_active() const noexcept { return impl_->pending.has_value(); }
uint64_t ProtectedPageRetention::active_group_id() const {
  if (!impl_->pending) throw std::logic_error("PMI retention: no active group");
  return impl_->pending->id;
}
uint64_t ProtectedPageRetention::protected_bytes() const noexcept { return impl_->bytes; }
uint64_t ProtectedPageRetention::peak_protected_bytes() const noexcept { return impl_->peak; }
uint64_t ProtectedPageRetention::protected_limit_bytes() const noexcept { return impl_->limit; }

} // namespace pmi
