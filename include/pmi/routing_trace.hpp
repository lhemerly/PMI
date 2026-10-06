#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace pmi {

// Optional, bounded JSONL capture. fields_json contains trusted, already escaped
// JSON object members, without the surrounding braces. The caller owns the
// privacy allowlist and must not supply schema, type, or run_id members. This
// class validates line framing, not JSON syntax or application field contents.
// Calls are synchronous and must be serialized by the caller.
class RoutingTraceWriter {
 public:
  static constexpr uint64_t default_byte_limit = 8 * 1024 * 1024;
  static constexpr uint64_t default_row_limit = 65536;
  static constexpr uint64_t trailer_reserve_bytes = 4096;
  static constexpr uint64_t max_fields_bytes = 64 * 1024;

  // Creates a new owner-readable/writable file exclusively. Existing files and
  // symlinks are never followed or replaced. run_id is 1..128 printable ASCII
  // bytes. The byte budget includes the run record and completion trailer;
  // the row budget counts routing and metrics records only.
  RoutingTraceWriter(const std::filesystem::path& path, std::string run_id,
                     std::string_view header_fields_json,
                     uint64_t byte_limit = default_byte_limit,
                     uint64_t row_limit = default_row_limit);
  ~RoutingTraceWriter() noexcept;
  RoutingTraceWriter(const RoutingTraceWriter&) = delete;
  RoutingTraceWriter& operator=(const RoutingTraceWriter&) = delete;
  RoutingTraceWriter(RoutingTraceWriter&&) = delete;
  RoutingTraceWriter& operator=(RoutingTraceWriter&&) = delete;

  // type is "routing" or "metrics". Returns false after a byte, record-size,
  // or row limit; hitting a limit never throws. Other malformed arguments and
  // I/O failures throw. No further records are accepted after the first limit.
  bool append(std::string_view type, std::string_view fields_json);

  // A caller-side capture limit (for example, bounded pending routing tensors)
  // omitted data. Stop accepting records and mark the eventual trailer as
  // incomplete/truncated with reason scope_limit. The first limit reason wins.
  // This does not close the file; finish still reports final I/O errors.
  void stop_capture() noexcept;

  // Explicit completion flushes/closes the file and reports I/O errors. A cap
  // forces complete=false and truncated=true. reason is a diagnostic code,
  // bounded to 128 bytes and normalized to ASCII; use identifier characters
  // accepted by the dataset importer. Repeated finish calls are harmless.
  // Destruction without finish records incomplete completion best effort.
  void finish(bool complete = true, const std::string& reason = {});

  bool truncated() const noexcept { return truncated_; }
  bool accepting() const noexcept { return !finished_ && !truncated_ && !failed_; }
  uint64_t records_written() const noexcept { return records_written_; }
  uint64_t bytes_written() const noexcept { return bytes_written_; }

 private:
  std::string line(std::string_view type, std::string_view fields_json) const;
  void write_line(const std::string& value);
  void truncate(const char* reason) noexcept;
  int fd_{-1};
  std::string run_id_json_;
  uint64_t byte_limit_{};
  uint64_t row_limit_{};
  uint64_t records_written_{};
  uint64_t bytes_written_{};
  bool finished_{};
  bool truncated_{};
  bool failed_{};
  const char* truncation_reason_{};
};

}  // namespace pmi
