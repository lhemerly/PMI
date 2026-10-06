#pragma once
#include "pmi/gguf.hpp"
#include "pmi/runtime_ffi.h"
#include <future>
#include <memory>
#include <stdexcept>

namespace pmi {
// Immutable zero-copy view of a 64-byte-aligned Rust-owned payload. A lease
// keeps bytes alive independently of the provider and blocks their eviction.
class Page final {
 public:
  ~Page();
  Page(const Page&) = delete;
  Page& operator=(const Page&) = delete;
  size_t size() const noexcept;
  const uint8_t* data() const noexcept;
  const uint8_t& operator[](size_t index) const noexcept { return data()[index]; }
  bool operator==(const Page& other) const noexcept;
  bool operator!=(const Page& other) const noexcept { return !(*this == other); }
 private:
  friend class FilePageProvider;
  explicit Page(PmiPageLease* lease) noexcept : lease_(lease) {}
  PmiPageLease* lease_;
};

using PageLease = std::shared_ptr<const Page>;
using PageFuture = std::shared_future<PageLease>;
using RuntimeStats = PmiRuntimeStats;
using IoStats = PmiIoStats;
enum class HostReadPolicy { Buffered, DirectPreferred, DirectRequired };

class PageProvider {
 public:
  virtual ~PageProvider() = default;
  virtual PageFuture request(const TensorInfo&) = 0;
  virtual PageFuture prefetch(const TensorInfo&) = 0;
  virtual bool pin(const TensorInfo&) = 0;
  virtual bool release(const TensorInfo&) = 0;
  // Explicit opt-in: unsupported providers reject rather than silently retain.
  virtual PageFuture request_transient(const TensorInfo&) {
    throw std::runtime_error("provider does not support transient leases");
  }
  virtual PageFuture request_direct(const TensorInfo&, bool transient, bool required) {
    (void)transient; (void)required;
    throw std::runtime_error("provider does not support direct leases");
  }
};
class FilePageProvider final : public PageProvider {
 public:
  FilePageProvider(std::filesystem::path path, size_t cache_bytes);
  FilePageProvider(std::vector<std::filesystem::path> shards, size_t cache_bytes);
  ~FilePageProvider() override;
  FilePageProvider(const FilePageProvider&) = delete;
  FilePageProvider& operator=(const FilePageProvider&) = delete;
  // I/O is synchronous initially; both methods return ready futures and
  // propagate failures through future.get().
  PageFuture request(const TensorInfo&) override;
  PageFuture prefetch(const TensorInfo&) override;
  PageFuture request_transient(const TensorInfo&) override;
  bool pin(const TensorInfo&) override;
  bool release(const TensorInfo&) override;
  RuntimeStats stats() const;
  IoStats io_stats() const;
  PageFuture request_direct(const TensorInfo&, bool transient, bool required) override;
 private:
  PmiRuntime* runtime_{};
  PageLease load(const TensorInfo&, bool transient = false, HostReadPolicy io = HostReadPolicy::Buffered);
};
} // namespace pmi
