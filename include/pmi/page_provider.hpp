#pragma once
#include "pmi/gguf.hpp"
#include "pmi/cache.hpp"
#include <future>
#include <memory>
#include <mutex>

namespace pmi {
class PageProvider {
 public:
  virtual ~PageProvider() = default;
  virtual std::shared_future<std::shared_ptr<const std::vector<uint8_t>>> request(const TensorInfo&) = 0;
  virtual std::shared_future<std::shared_ptr<const std::vector<uint8_t>>> prefetch(const TensorInfo&) = 0;
  virtual bool pin(const TensorInfo&) = 0;
  virtual bool release(const TensorInfo&) = 0;
};
class FilePageProvider final : public PageProvider {
 public:
  FilePageProvider(std::filesystem::path path, size_t cache_bytes);
  std::shared_future<std::shared_ptr<const std::vector<uint8_t>>> request(const TensorInfo&) override;
  std::shared_future<std::shared_ptr<const std::vector<uint8_t>>> prefetch(const TensorInfo&) override;
  bool pin(const TensorInfo&) override; bool release(const TensorInfo&) override;
  const CacheStats& stats() const { return cache_.stats(); }
 private:
  std::filesystem::path path_; size_t cache_bytes_{}; LruByteCache cache_;
  std::mutex mutex_;
  std::shared_ptr<const std::vector<uint8_t>> load(const TensorInfo&);
};
} // namespace pmi
