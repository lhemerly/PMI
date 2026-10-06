#pragma once
#include <cstdint>
#include <cstddef>
#include <cerrno>
#include <limits>
#include <stdexcept>
#include <system_error>
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace pmi {
struct OwnedPageRange { uintptr_t begin{}; size_t bytes{}; };
inline OwnedPageRange owned_page_interior(void* data, size_t size, size_t page) {
  if (!page || (page & (page - 1)) || (!data && size))
    throw std::invalid_argument("invalid owned-buffer range or page size");
  const auto first = reinterpret_cast<uintptr_t>(data);
  if (size > std::numeric_limits<uintptr_t>::max() - first)
    throw std::overflow_error("owned-buffer address range overflow");
  const auto end = first + size;
  const auto remainder = first % page;
  const auto skip = remainder ? page - remainder : 0;
  if (skip > size) return {};
  const auto begin = first + skip;
  const auto last = end - end % page;
  return last > begin ? OwnedPageRange{begin, last - begin} : OwnedPageRange{};
}
#ifdef __linux__
using MemoryAdvice = int (*)(void*, size_t, int);
inline OwnedPageRange advise_owned_base_pages(void* data, size_t size,
                                             MemoryAdvice apply = ::madvise) {
  const auto page = ::sysconf(_SC_PAGESIZE);
  if (page <= 0) throw std::runtime_error("cannot determine host page size");
  const auto range = owned_page_interior(data, size, static_cast<size_t>(page));
  if (range.bytes && apply(reinterpret_cast<void*>(range.begin), range.bytes, MADV_NOHUGEPAGE))
    throw std::system_error(errno, std::generic_category(), "owned CPU weight MADV_NOHUGEPAGE");
  return range;
}
#endif
} // namespace pmi
