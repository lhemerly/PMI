#include "pmi/page_provider.hpp"
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <utility>
namespace pmi {
namespace {
void check_status(PmiStatus status, const char* error) {
  if (status != PMI_OK) throw std::runtime_error(std::string("PMI runtime: ") + error);
}
} // namespace
Page::~Page() { pmi_page_destroy(lease_); }
size_t Page::size() const noexcept { return static_cast<size_t>(pmi_page_size(lease_)); }
const uint8_t* Page::data() const noexcept { return pmi_page_data(lease_); }
bool Page::operator==(const Page& other) const noexcept {
  return size() == other.size() && std::equal(data(), data() + size(), other.data());
}

FilePageProvider::FilePageProvider(std::filesystem::path path, size_t cache_bytes)
    : FilePageProvider(std::vector<std::filesystem::path>{std::move(path)}, cache_bytes) {}
FilePageProvider::FilePageProvider(std::vector<std::filesystem::path> shards, size_t cache_bytes) {
  std::vector<std::string> paths;
  paths.reserve(shards.size());
  for (const auto& path : shards) {
    auto value = path.string();
    if (value.find('\0') != std::string::npos) throw std::invalid_argument("PMI shard path contains NUL");
    paths.push_back(std::move(value));
  }
  std::vector<const char*> pointers;
  pointers.reserve(paths.size());
  for (const auto& path : paths) pointers.push_back(path.c_str());
  char error[512]{};
  check_status(pmi_runtime_create(pointers.data(), pointers.size(), cache_bytes,
                                  &runtime_, error, sizeof(error)), error);
}
FilePageProvider::~FilePageProvider() { pmi_runtime_destroy(runtime_); }
PageLease FilePageProvider::load(const TensorInfo& tensor, bool transient, HostReadPolicy io) {
  PmiPageLease* lease = nullptr;
  char error[512]{};
  const auto request = transient ? pmi_runtime_request_transient : pmi_runtime_request;
  if (io == HostReadPolicy::Buffered)
    check_status(request(runtime_, tensor.shard_id, tensor.absolute_offset,
                         tensor.byte_length, &lease, error, sizeof(error)), error);
  else
    check_status(pmi_runtime_request_direct(runtime_, tensor.shard_id, tensor.absolute_offset,
                         tensor.byte_length, transient ? 0 : 1, io == HostReadPolicy::DirectRequired ? 1 : 0,
                         &lease, error, sizeof(error)), error);
  auto lease_owner = std::unique_ptr<PmiPageLease, decltype(&pmi_page_destroy)>(lease, &pmi_page_destroy);
  std::unique_ptr<Page> page(new Page(lease_owner.get()));
  lease_owner.release();
  return PageLease(std::move(page));
}
PageFuture FilePageProvider::request(const TensorInfo& tensor) {
  std::promise<PageLease> result;
  auto future = result.get_future().share();
  try { result.set_value(load(tensor)); }
  catch (...) { result.set_exception(std::current_exception()); }
  return future;
}
PageFuture FilePageProvider::request_transient(const TensorInfo& tensor) {
  std::promise<PageLease> result;
  auto future = result.get_future().share();
  try { result.set_value(load(tensor, true)); }
  catch (...) { result.set_exception(std::current_exception()); }
  return future;
}
PageFuture FilePageProvider::request_direct(const TensorInfo& tensor, bool transient, bool required) {
  std::promise<PageLease> result; auto future=result.get_future().share();
  try {result.set_value(load(tensor,transient,required ? HostReadPolicy::DirectRequired : HostReadPolicy::DirectPreferred));}
  catch (...) {result.set_exception(std::current_exception());}
  return future;
}
IoStats FilePageProvider::io_stats() const {
  IoStats stats{};char error[512]{};
  check_status(pmi_runtime_get_io_stats(runtime_,&stats,error,sizeof(error)),error);
  return stats;
}
PageFuture FilePageProvider::prefetch(const TensorInfo& tensor) { return request(tensor); }
bool FilePageProvider::pin(const TensorInfo& tensor) {
  return pmi_runtime_pin(runtime_, tensor.shard_id, tensor.absolute_offset,
                          tensor.byte_length, nullptr, 0) == PMI_OK;
}
bool FilePageProvider::release(const TensorInfo& tensor) {
  return pmi_runtime_release(runtime_, tensor.shard_id, tensor.absolute_offset,
                              tensor.byte_length, nullptr, 0) == PMI_OK;
}
RuntimeStats FilePageProvider::stats() const {
  RuntimeStats stats{};
  char error[512]{};
  check_status(pmi_runtime_get_stats(runtime_, &stats, error, sizeof(error)), error);
  return stats;
}
} // namespace pmi
