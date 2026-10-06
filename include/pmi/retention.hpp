#pragma once

#include "pmi/page_provider.hpp"
#include <cstdint>
#include <memory>
#include <vector>

namespace pmi {

// Retains exact byte ranges across caller-defined groups using counted pins.
// This helper never requests or prefetches data and has no model/layout logic.
// The provider must outlive it. Calls to one helper must be externally serialized;
// provider pin/release failures (false or exceptions) must leave the pin unchanged.
//
// The limit bounds 64-byte-rounded owned pins, including old/new turnover and any
// failed-cleanup pins. The caller must separately plan provider capacity for these
// pins, a service reserve, and unrelated leases/pins. It is not an RSS bound.
class ProtectedPageRetention final {
 public:
  ProtectedPageRetention(PageProvider&, uint64_t protected_limit_bytes);
  // Best-effort cleanup only if the provider fails: use clear() to observe errors.
  ~ProtectedPageRetention() noexcept;
  ProtectedPageRetention(const ProtectedPageRetention&) = delete;
  ProtectedPageRetention& operator=(const ProtectedPageRetention&) = delete;

  // One transaction may be active. Group IDs are opaque; previously committed
  // groups remain protected. Duplicate exact ranges in next are counted once.
  // Checks the complete currently-owned + next union before changing any pins.
  void begin_group(uint64_t group_id, const std::vector<TensorInfo>& next);
  // Call while retaining the requested page lease. Nondesignated pages are
  // ignored; desired pages are observed and pinned exactly once globally.
  // A pin failure cancels this transaction, preserving committed group pins.
  void observe_loaded(const TensorInfo&);
  // Requires every desired range to have been observed, including prior overlap.
  // Commits next before releasing obsolete pins. A release failure is reported,
  // remains tracked/charged, and requires clear() before starting another group.
  void finish_group(uint64_t group_id);
  // Drops only this transaction's new pins, retaining committed groups.
  void cancel_group();
  // Attempts every owned release. Failed releases remain tracked for a retry;
  // successfully released pins are never released again by this helper.
  void clear();

  bool group_active() const noexcept;
  uint64_t active_group_id() const;
  uint64_t protected_bytes() const noexcept;
  uint64_t peak_protected_bytes() const noexcept;
  uint64_t protected_limit_bytes() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace pmi
