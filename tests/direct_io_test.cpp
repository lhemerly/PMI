#include "llama.h"
#include "llama-mmap.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef __linux__
#error This test requires Linux and GNU linker wrapping.
#endif

namespace {

constexpr size_t alignment = 4096;

struct tracked_fd {
    int value = -1;
};

struct tracked_allocation {
    void * value = nullptr;
    size_t bytes = 0;
};

struct wrapper_state {
    bool enabled = false;
    bool fail_open = false;
    bool fail_stat = false;
    bool fail_seek = false;
    int fail_seek_whence = -1;
    bool fail_allocation = false;
    int fail_read_call = 0;
    int read_errno = EINVAL;
    size_t first_read_limit = 0;
    uint64_t open_attempts = 0;
    uint64_t read_calls = 0;
    uint64_t direct_bytes = 0;
    uint64_t fread_calls = 0;
    uint64_t buffered_bytes = 0;
    uint64_t allocation_calls = 0;
    uint64_t allocation_frees = 0;
    size_t allocation_live = 0;
    size_t allocation_peak = 0;
    std::array<tracked_fd, 8> descriptors;
    std::array<tracked_allocation, 8> allocations;
};

wrapper_state state;

tracked_fd * find_fd(int fd) {
    if (fd < 0) {
        return nullptr;
    }
    for (auto & entry : state.descriptors) {
        if (entry.value == fd) {
            return &entry;
        }
    }
    return nullptr;
}

size_t open_count() {
    size_t result = 0;
    for (const auto & entry : state.descriptors) {
        result += entry.value != -1;
    }
    return result;
}

void require(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

template<class F> void rejects(F action) {
    bool caught = false;
    try {
        action();
    } catch (const std::exception &) {
        caught = true;
    }
    require(caught, "expected direct-I/O failure");
}

void require_cleanup() {
    require(open_count() == 0, "direct descriptor leaked");
    require(state.allocation_live == 0, "aligned allocation leaked");
}

void begin_case() {
    require_cleanup();
    state = wrapper_state{};
    state.enabled = true;
}

void expect_counts(const llama_file & file, uint64_t requested, uint64_t completed,
                   uint64_t direct, uint64_t buffered, uint64_t staging, uint64_t fallback) {
    const auto info = file.load_io_info();
    require(info.payload_requested_bytes == requested, "requested payload accounting mismatch");
    require(info.payload_completed_bytes == completed, "completed payload accounting mismatch");
    require(info.direct_read_bytes == direct, "direct byte accounting mismatch");
    require(info.buffered_read_bytes == buffered, "buffered byte accounting mismatch");
    require(info.staging_peak_bytes == staging, "staging peak accounting mismatch");
    require(info.fallback_count == fallback, "fallback accounting mismatch");
    require(info.staging_peak_bytes <= LLAMA_DIRECT_IO_BUFFER_SIZE, "staging exceeded configured maximum");
}

void expect_mode(const llama_file & file, bool requested, bool required, bool active) {
    const auto info = file.load_io_info();
    require(info.direct_io_requested == requested, "requested mode mismatch");
    require(info.direct_io_required == required, "required mode mismatch");
    require(info.direct_io_active == active, "effective mode mismatch");
    require(file.has_direct_io() == active, "effective mode disagrees with file handle");
}

struct fixture {
    std::filesystem::path directory;
    std::array<std::filesystem::path, 2> paths;
    std::array<std::vector<uint8_t>, 2> bytes;

    fixture() {
        std::string pattern = (std::filesystem::temp_directory_path() / "pmi-direct-io-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back('\0');
        require(mkdtemp(name.data()) != nullptr, "failed to create fixture directory");
        directory = name.data();
        try {
            for (size_t shard = 0; shard < paths.size(); ++shard) {
                paths[shard] = directory / ("shard-" + std::to_string(shard) + ".bin");
                bytes[shard].resize(8 * alignment + 13);
                for (size_t i = 0; i < bytes[shard].size(); ++i) {
                    bytes[shard][i] = static_cast<uint8_t>((i * 37 + i / 251 + shard * 97) % 256);
                }
                std::ofstream output(paths[shard], std::ios::binary);
                output.write(reinterpret_cast<const char *>(bytes[shard].data()), bytes[shard].size());
                output.close();
                require(output.good(), "failed to write fixture");
            }
        } catch (...) {
            std::error_code ignored;
            std::filesystem::remove_all(directory, ignored);
            throw;
        }
    }

    ~fixture() {
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
    }
};

void expect_read(llama_file & file, const std::vector<uint8_t> & source, size_t offset, size_t length) {
    std::vector<uint8_t> result(length + 2, 0xa5);
    file.read_raw(result.data() + 1, length);
    require(result.front() == 0xa5 && result.back() == 0xa5, "read changed destination guards");
    require(std::equal(result.begin() + 1, result.end() - 1, source.begin() + offset), "payload bytes differ");
    require(file.tell() == offset + length, "read did not preserve logical file position");
}

void test_buffered(const fixture & data) {
    begin_case();
    {
        llama_file file(data.paths[0].c_str(), "rb");
        expect_mode(file, false, false, false);
        file.seek(11, SEEK_SET);
        expect_read(file, data.bytes[0], 11, 1000);
        file.seek(101, SEEK_SET);
        expect_read(file, data.bytes[0], 101, 703);
        expect_counts(file, 1703, 1703, 0, 1703, 0, 0);
        require(state.open_attempts == 0 && state.read_calls == 0, "buffered mode requested direct I/O");
        require(state.buffered_bytes == 1703, "buffered wrapper byte mismatch");
        file.seek(data.bytes[0].size() - 1, SEEK_SET);
        uint8_t beyond_end[2] = {};
        const auto fread_calls = state.fread_calls;
        rejects([&] { file.read_raw(beyond_end, sizeof(beyond_end)); });
        expect_counts(file, 1705, 1703, 0, 1703, 0, 0);
        require(state.fread_calls == fread_calls, "buffered out-of-range payload attempted I/O");
        file.seek(data.bytes[0].size(), SEEK_SET);
        file.read_raw(nullptr, 0);
        expect_counts(file, 1705, 1703, 0, 1703, 0, 0);
        require(state.fread_calls == fread_calls, "buffered zero-length read attempted I/O");
    }
    require_cleanup();
}

void test_direct_reads(const fixture & data, bool strict) {
    begin_case();
    {
        llama_file file(data.paths[0].c_str(), "rb", true, strict);
        expect_mode(file, true, strict, true);
        require(file.read_alignment() == alignment, "fixture alignment mismatch");
        require(file.size() == data.bytes[0].size(), "file size mismatch");
        file.seek(alignment, SEEK_SET);
        expect_read(file, data.bytes[0], alignment, alignment);
        file.seek(37, SEEK_SET);
        expect_read(file, data.bytes[0], 37, 5000);
        file.seek(23, SEEK_SET);
        expect_read(file, data.bytes[0], 23, 113);
        file.seek(data.bytes[0].size() - 29, SEEK_SET);
        expect_read(file, data.bytes[0], data.bytes[0].size() - 29, 29);
        expect_counts(file, 9238, 9238, 20493, 0, 8192, 0);
        const auto read_calls = state.read_calls;
        const auto allocation_calls = state.allocation_calls;
        file.read_raw(nullptr, 0);
        require(file.tell() == data.bytes[0].size(), "zero-length read changed file position");
        require(state.read_calls == read_calls && state.allocation_calls == allocation_calls,
                "zero-length read issued I/O or staging allocation");
        file.seek(37, SEEK_SET);
        expect_read(file, data.bytes[0], 37, 31);
        expect_read(file, data.bytes[0], 68, 511);
        expect_counts(file, 9780, 9780, 28685, 0, 8192, 0);
        require(state.direct_bytes == 28685 && state.fread_calls == 0, "direct reads used buffered I/O");
        require(state.allocation_peak == 8192 && state.allocation_live == 0, "staging lifetime mismatch");
        file.seek(data.bytes[0].size() - 1, SEEK_SET);
        uint8_t beyond_end[2] = {};
        const auto reads_before_error = state.read_calls;
        const auto allocations_before_error = state.allocation_calls;
        rejects([&] { file.read_raw(beyond_end, sizeof(beyond_end)); });
        expect_counts(file, 9782, 9780, 28685, 0, 8192, 0);
        require(state.read_calls == reads_before_error && state.allocation_calls == allocations_before_error,
                "out-of-range payload attempted I/O or allocation");
    }
    require_cleanup();
}

void test_shards(const fixture & data) {
    begin_case();
    {
        llama_file second(data.paths[1].c_str(), "rb", true, true);
        {
            llama_file first(data.paths[0].c_str(), "rb", true, true);
            require(open_count() == 2, "shard descriptors were not independent");
            first.seek(61, SEEK_SET);
            second.seek(61, SEEK_SET);
            expect_read(first, data.bytes[0], 61, 123);
            expect_read(second, data.bytes[1], 61, 123);
            expect_counts(first, 123, 123, 4096, 0, 4096, 0);
            expect_counts(second, 123, 123, 4096, 0, 4096, 0);
        }
        require(open_count() == 1, "closing one shard changed the other descriptor");
        expect_read(second, data.bytes[1], 184, 77);
        expect_counts(second, 200, 200, 8192, 0, 4096, 0);
    }
    require_cleanup();
}

void test_open_failure(const fixture & data, bool strict, bool stat_failure) {
    begin_case();
    state.fail_open = !stat_failure;
    state.fail_stat = stat_failure;
    if (strict) {
        rejects([&] { llama_file file(data.paths[0].c_str(), "rb", true, true); });
        require(state.fread_calls == 0 && state.allocation_calls == 0, "strict open failure used buffered I/O");
    } else {
        llama_file file(data.paths[0].c_str(), "rb", true, false);
        expect_mode(file, true, false, false);
        file.seek(17, SEEK_SET);
        expect_read(file, data.bytes[0], 17, 3001);
        expect_counts(file, 3001, 3001, 0, 3001, 0, 1);
    }
    require(state.open_attempts == 1, "direct open attempt was not observed");
    require_cleanup();
}

void test_read_failure(const fixture & data, bool strict, int error, bool partial) {
    begin_case();
    {
        llama_file file(data.paths[0].c_str(), "rb", true, strict);
        file.seek(37, SEEK_SET);
        state.first_read_limit = partial ? 1024 : 0;
        state.fail_read_call = partial ? 2 : 1;
        state.read_errno = error;
        const uint64_t direct = partial ? 1024 : 0;
        if (strict) {
            std::vector<uint8_t> output(7000);
            rejects([&] { file.read_raw(output.data(), output.size()); });
            expect_counts(file, 7000, 0, direct, 0, 8192, 0);
            require(state.fread_calls == 0 && state.buffered_bytes == 0, "strict read failure performed buffered I/O");
        } else {
            expect_read(file, data.bytes[0], 37, 7000);
            expect_mode(file, true, false, false);
            expect_counts(file, 7000, 7000, direct, 8192 - direct, 8192, 1);
            require(state.buffered_bytes == 8192 - direct, "fallback did not read only the remainder");
            expect_read(file, data.bytes[0], 7037, 257);
            expect_counts(file, 7257, 7257, direct, 8449 - direct, 8192, 1);
        }
        require(state.direct_bytes == direct, "partial direct read was not counted exactly");
        require(state.allocation_live == 0 && state.allocation_frees == 1, "read failure leaked staging");
    }
    require_cleanup();
}

void test_interrupted_read(const fixture & data) {
    begin_case();
    {
        llama_file file(data.paths[0].c_str(), "rb", true, true);
        state.fail_read_call = 1;
        state.read_errno = EINTR;
        expect_read(file, data.bytes[0], 0, 123);
        expect_counts(file, 123, 123, 4096, 0, 4096, 0);
        require(state.read_calls == 2 && state.fread_calls == 0, "interrupted direct read was not retried");
    }
    require_cleanup();
}

void test_cleanup_failures(const fixture & data) {
    begin_case();
    state.fail_seek = true;
    rejects([&] { llama_file file(data.paths[0].c_str(), "rb", true, true); });
    require_cleanup();
    require(state.open_attempts == 1 && state.fread_calls == 0, "constructor seek failure used buffered I/O");

    begin_case();
    {
        llama_file file(data.paths[0].c_str(), "rb", true, true);
        state.fail_allocation = true;
        uint8_t output[123] = {};
        rejects([&] { file.read_raw(output, sizeof(output)); });
        expect_counts(file, 123, 0, 0, 0, 0, 0);
        require(state.allocation_calls == 1 && state.allocation_live == 0, "allocation failure accounting mismatch");
        require(state.read_calls == 0 && state.fread_calls == 0, "allocation failure issued I/O");
    }
    require_cleanup();

    begin_case();
    {
        llama_file file(data.paths[0].c_str(), "rb", true, true);
        state.fail_seek = true;
        state.fail_seek_whence = SEEK_SET;
        uint8_t output[123] = {};
        rejects([&] { file.read_raw(output, sizeof(output)); });
        expect_counts(file, 123, 0, 0, 0, 4096, 0);
        require(state.allocation_calls == 1 && state.allocation_frees == 1 && state.allocation_live == 0,
                "seek failure leaked aligned staging");
        require(state.read_calls == 0 && state.fread_calls == 0, "seek failure issued payload I/O");
    }
    require_cleanup();
}

} // namespace

extern "C" {
int __real_open(const char * path, int flags, ...);
ssize_t __real_read(int fd, void * buffer, size_t count);
int __real_close(int fd);
size_t __real_fread(void * buffer, size_t size, size_t count, FILE * stream);
int __real_posix_memalign(void ** pointer, size_t alignment_value, size_t size);
void __real_free(void * pointer);
int __real_fstat(int fd, struct stat * info);
off_t __real_lseek(int fd, off_t offset, int whence);

int __wrap_open(const char * path, int flags, ...) {
    mode_t mode = 0;
    if ((flags & O_CREAT) != 0 || (flags & O_TMPFILE) == O_TMPFILE) {
        va_list arguments;
        va_start(arguments, flags);
        mode = va_arg(arguments, mode_t);
        va_end(arguments);
    }
    if (!state.enabled || (flags & O_DIRECT) == 0) {
        return __real_open(path, flags, mode);
    }
    ++state.open_attempts;
    if (state.fail_open) {
        errno = EINVAL;
        return -1;
    }
    // Strip O_DIRECT only in this test so faults do not depend on the filesystem.
    const int fd = __real_open(path, flags & ~O_DIRECT, mode);
    if (fd != -1) {
        for (auto & entry : state.descriptors) {
            if (entry.value == -1) {
                entry.value = fd;
                return fd;
            }
        }
        __real_close(fd);
        errno = EMFILE;
        return -1;
    }
    return fd;
}

ssize_t __wrap_read(int fd, void * buffer, size_t count) {
    if (find_fd(fd) == nullptr) {
        return __real_read(fd, buffer, count);
    }
    ++state.read_calls;
    if (state.read_calls == static_cast<uint64_t>(state.fail_read_call)) {
        errno = state.read_errno;
        return -1;
    }
    if (state.read_calls == 1 && state.first_read_limit != 0) {
        count = std::min(count, state.first_read_limit);
    }
    const ssize_t result = __real_read(fd, buffer, count);
    if (result > 0) {
        state.direct_bytes += static_cast<uint64_t>(result);
    }
    return result;
}

int __wrap_close(int fd) {
    auto * entry = find_fd(fd);
    const int result = __real_close(fd);
    if (entry != nullptr && result == 0) {
        entry->value = -1;
    }
    return result;
}

size_t __wrap_fread(void * buffer, size_t size, size_t count, FILE * stream) {
    const size_t result = __real_fread(buffer, size, count, stream);
    if (state.enabled) {
        ++state.fread_calls;
        state.buffered_bytes += result * size;
    }
    return result;
}

int __wrap_posix_memalign(void ** pointer, size_t alignment_value, size_t size) {
    if (!state.enabled) {
        return __real_posix_memalign(pointer, alignment_value, size);
    }
    ++state.allocation_calls;
    if (state.fail_allocation) {
        return ENOMEM;
    }
    const int result = __real_posix_memalign(pointer, alignment_value, size);
    if (result == 0) {
        for (auto & entry : state.allocations) {
            if (entry.value == nullptr) {
                entry.value = *pointer;
                entry.bytes = size;
                state.allocation_live += size;
                state.allocation_peak = std::max(state.allocation_peak, state.allocation_live);
                return result;
            }
        }
        __real_free(*pointer);
        *pointer = nullptr;
        return ENOMEM;
    }
    return result;
}

void __wrap_free(void * pointer) {
    if (pointer != nullptr) {
        for (auto & entry : state.allocations) {
            if (entry.value == pointer) {
                state.allocation_live -= entry.bytes;
                ++state.allocation_frees;
                entry = tracked_allocation{};
                break;
            }
        }
    }
    __real_free(pointer);
}

int __wrap_fstat(int fd, struct stat * info) {
    if (find_fd(fd) != nullptr && state.fail_stat) {
        state.fail_stat = false;
        errno = EIO;
        return -1;
    }
    const int result = __real_fstat(fd, info);
    if (result == 0 && find_fd(fd) != nullptr) {
        info->st_blksize = alignment;
    }
    return result;
}

off_t __wrap_lseek(int fd, off_t offset, int whence) {
    if (find_fd(fd) != nullptr && state.fail_seek &&
        (state.fail_seek_whence == -1 || state.fail_seek_whence == whence)) {
        state.fail_seek = false;
        errno = EIO;
        return -1;
    }
    return __real_lseek(fd, offset, whence);
}
} // extern "C"

int main() {
    try {
        const fixture data;
        test_buffered(data);
        test_direct_reads(data, false);
        test_direct_reads(data, true);
        test_shards(data);
        for (bool strict : {false, true}) {
            test_open_failure(data, strict, false);
            test_open_failure(data, strict, true);
            for (int error : {EINVAL, EFAULT}) {
                test_read_failure(data, strict, error, false);
                test_read_failure(data, strict, error, true);
            }
        }
        test_interrupted_read(data);
        test_cleanup_failures(data);
        state.enabled = false;
        std::cout << "direct-I/O fixture tests passed (bytes, seeks, EOF, shards, accounting, fallback, strict failures, cleanup)\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
