#pragma once
#include <cstddef>
#include <cstdint>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace pmi {
struct CacheStats { uint64_t hits{}, misses{}, evictions{}, bytes_loaded{}, bytes_reused{}; };
struct CacheEntry { uint64_t key{}; std::vector<uint8_t> bytes; bool pinned{}; };
class LruByteCache {
 public:
  explicit LruByteCache(size_t capacity) : capacity_(capacity) {}
  const std::vector<uint8_t>* get(uint64_t key);
  bool put(uint64_t key, std::vector<uint8_t> bytes);
  bool pin(uint64_t key); bool unpin(uint64_t key); bool erase(uint64_t key);
  size_t resident_bytes() const { return resident_bytes_; }
  size_t capacity() const { return capacity_; }
  const CacheStats& stats() const { return stats_; }
 private:
  size_t capacity_{}; size_t resident_bytes_{};
  std::list<CacheEntry> lru_;
  std::unordered_map<uint64_t, std::list<CacheEntry>::iterator> index_;
  CacheStats stats_{};
  bool evict_for(size_t needed);
};
} // namespace pmi
