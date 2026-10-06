#ifndef PMI_RUNTIME_FFI_H
#define PMI_RUNTIME_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PmiRuntime PmiRuntime;
typedef struct PmiPageLease PmiPageLease;
typedef int32_t PmiStatus;
enum {
  PMI_OK = 0,
  PMI_INVALID_ARGUMENT = 1,
  PMI_IO_ERROR = 2,
  PMI_CAPACITY = 3,
  PMI_NOT_FOUND = 4,
  PMI_INTERNAL_ERROR = 5,
  PMI_OUT_OF_MEMORY = 6
};

/* current_bytes/peak_bytes include the owned payload allocation, rounded up
 * to 64-byte alignment, but exclude allocator/cache metadata, backend buffers,
 * model state and OS file caches. io_bytes includes partial failed reads.
 * failures counts failed page requests; pin/release misses are not requests. */
typedef struct PmiRuntimeStats {
  uint64_t hits;
  uint64_t misses;
  uint64_t evictions;
  uint64_t failures;
  uint64_t io_bytes;
  uint64_t bytes_reused;
  uint64_t current_bytes;
  uint64_t peak_bytes;
  uint64_t capacity_bytes;
  uint64_t entries;
  uint64_t pinned_entries;
  uint64_t active_leases;
} PmiRuntimeStats;

/* Per-provider page I/O only; excludes dense loader/metadata. Read bytes are
 * syscall-returned bytes including alignment overlap and partial failed reads.
 * O_DIRECT descriptor success does not prove a filesystem never buffers inside
 * its kernel implementation. staging_peak_bytes is max single temporary charged
 * against the same host payload quota. Existing PmiRuntimeStats ABI is intact. */
typedef struct PmiIoStats {
  uint64_t direct_requests, direct_completed, fallback_count;
  uint64_t direct_read_bytes, buffered_read_bytes, direct_logical_bytes;
  uint64_t staging_peak_bytes, alignment_statx_requests, alignment_known_fs_requests;
} PmiIoStats;

/* Pointer arguments must reference valid memory for the call. Error output
 * is optional and NUL terminated when capacity > 0. No function unwinds
 * across the C ABI. Operations serialize internally; finish concurrent calls
 * before destroying the runtime. Leases may outlive their runtime. */
PmiStatus pmi_runtime_create(const char* const* shard_paths, size_t shard_count,
                             uint64_t capacity_bytes, PmiRuntime** output,
                             char* error, size_t error_capacity);
void pmi_runtime_destroy(PmiRuntime* runtime);
PmiStatus pmi_runtime_request(PmiRuntime* runtime, uint32_t shard_id,
                              uint64_t offset, uint64_t length,
                              PmiPageLease** output,
                              char* error, size_t error_capacity);
/* Transient miss: same alignment/range/budget checks and lease lifetime, but
 * no retained cache entry. Existing exact cached hits/pins remain unchanged.
 * Live transient bytes still consume capacity until the final lease dies. */
PmiStatus pmi_runtime_request_transient(PmiRuntime* runtime, uint32_t shard_id,
                              uint64_t offset, uint64_t length,
                              PmiPageLease** output,
                              char* error, size_t error_capacity);
/* Opt-in O_DIRECT miss with retained (1) or transient (0) payload ownership.
 * required=1 never falls back in userspace; required=0 falls back only for
 * unsupported platform/filesystem/alignment before any direct bytes arrive.
 * Permission, identity, real I/O and short-read failures propagate. */
PmiStatus pmi_runtime_request_direct(PmiRuntime* runtime, uint32_t shard_id,
                              uint64_t offset, uint64_t length,
                              uint8_t retain, uint8_t required,
                              PmiPageLease** output, char* error, size_t error_capacity);
PmiStatus pmi_runtime_get_io_stats(const PmiRuntime* runtime, PmiIoStats* output,
                                  char* error, size_t error_capacity);
/* Pins are counted and apply only to already cached exact ranges. */
PmiStatus pmi_runtime_pin(PmiRuntime* runtime, uint32_t shard_id,
                          uint64_t offset, uint64_t length,
                          char* error, size_t error_capacity);
PmiStatus pmi_runtime_release(PmiRuntime* runtime, uint32_t shard_id,
                              uint64_t offset, uint64_t length,
                              char* error, size_t error_capacity);
PmiStatus pmi_runtime_get_stats(const PmiRuntime* runtime, PmiRuntimeStats* output,
                                char* error, size_t error_capacity);
/* Data is immutable, aligned to at least 64 bytes, and valid until this lease
 * is destroyed. A cache hit returns the same underlying allocation. */
const uint8_t* pmi_page_data(const PmiPageLease* lease);
uint64_t pmi_page_size(const PmiPageLease* lease);
void pmi_page_destroy(PmiPageLease* lease);

#ifdef __cplusplus
}
#endif
#endif
