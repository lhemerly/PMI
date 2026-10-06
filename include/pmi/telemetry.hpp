#pragma once
#include <cstdint>
#include <optional>
namespace pmi {
struct ProcessMemory {
  std::optional<uint64_t> rss_bytes, peak_rss_bytes, storage_read_bytes;
};
ProcessMemory process_memory();
}
