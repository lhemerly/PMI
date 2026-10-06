#include "pmi/routing_trace.hpp"
#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <stdexcept>
#include <system_error>
#include <unistd.h>

namespace pmi {
namespace {
std::string quoted_ascii(std::string_view value) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result = "\"";
  for (unsigned char c : value) {
    if (c == '\\' || c == '"') {
      result += '\\';
      result += static_cast<char>(c);
    } else if (c < 0x20 || c == 0x7f) {
      result += "\\u00";
      result += hex[c >> 4];
      result += hex[c & 15];
    } else {
      result += c < 0x80 ? static_cast<char>(c) : '?';
    }
  }
  result += '"';
  return result;
}

void validate_fields(std::string_view fields) {
  for (unsigned char c : fields) {
    if (c < 0x20) {
      throw std::invalid_argument("routing JSON fields must use escaped control characters");
    }
  }
}
}  // namespace

RoutingTraceWriter::RoutingTraceWriter(const std::filesystem::path& path,
                                     std::string run_id,
                                     std::string_view header_fields_json,
                                     uint64_t byte_limit, uint64_t row_limit)
    : byte_limit_(byte_limit), row_limit_(row_limit) {
  if (run_id.empty() || run_id.size() > 128 ||
      std::any_of(run_id.begin(), run_id.end(), [](unsigned char c) {
        return c < 0x20 || c > 0x7e;
      })) {
    throw std::invalid_argument("routing run_id must be 1..128 printable ASCII bytes");
  }
  if (header_fields_json.size() > max_fields_bytes) {
    throw std::invalid_argument("routing run metadata exceeds 64 KiB");
  }
  validate_fields(header_fields_json);
  run_id_json_ = quoted_ascii(run_id);
  const auto header = line("run", header_fields_json);
  if (byte_limit_ < header.size() || byte_limit_ - header.size() < trailer_reserve_bytes) {
    throw std::invalid_argument("routing byte limit cannot hold metadata and completion reserve");
  }
  fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (fd_ < 0) {
    throw std::system_error(errno, std::generic_category(), "create routing JSONL");
  }
  try {
    write_line(header);
  } catch (...) {
    ::close(fd_);
    fd_ = -1;
    throw;
  }
}

RoutingTraceWriter::~RoutingTraceWriter() noexcept {
  if (fd_ < 0) return;
  try {
    if (!failed_) finish(false, "cancelled");
  } catch (...) {
    // Destructors cannot surface I/O failures. Explicit finish is required to
    // observe them; a missing/partial trailer remains incomplete to importers.
  }
  if (fd_ >= 0) ::close(fd_);
}

std::string RoutingTraceWriter::line(std::string_view type, std::string_view fields_json) const {
  std::string result = "{\"schema\":\"pmi.routing.v1\",\"type\":\"";
  result.append(type.data(), type.size());
  result += "\",\"run_id\":";
  result += run_id_json_;
  if (!fields_json.empty()) {
    result += ',';
    result.append(fields_json.data(), fields_json.size());
  }
  result += "}\n";
  return result;
}

void RoutingTraceWriter::write_line(const std::string& value) {
  size_t offset = 0;
  while (offset < value.size()) {
    const auto written = ::write(fd_, value.data() + offset, value.size() - offset);
    if (written < 0 && errno == EINTR) continue;
    if (written <= 0) {
      const auto error = written == 0 ? EIO : errno;
      failed_ = true;
      throw std::system_error(error, std::generic_category(), "write routing JSONL");
    }
    offset += static_cast<size_t>(written);
    bytes_written_ += static_cast<uint64_t>(written);
  }
}

void RoutingTraceWriter::truncate(const char* reason) noexcept {
  truncated_ = true;
  if (!truncation_reason_) truncation_reason_ = reason;
}

void RoutingTraceWriter::stop_capture() noexcept {
  if (!finished_ && !failed_) truncate("scope_limit");
}

bool RoutingTraceWriter::append(std::string_view type, std::string_view fields_json) {
  if (failed_) throw std::runtime_error("routing JSONL has a previous I/O failure");
  if (finished_) throw std::logic_error("routing JSONL is already finished");
  if (type != "routing" && type != "metrics") {
    throw std::invalid_argument("routing JSONL record type must be routing or metrics");
  }
  if (truncated_) return false;
  if (fields_json.size() > max_fields_bytes) {
    truncate("record_size_limit");
    return false;
  }
  validate_fields(fields_json);
  if (records_written_ >= row_limit_) {
    truncate("row_limit");
    return false;
  }
  const auto record = line(type, fields_json);
  // The constructor and this check preserve bytes_written <= limit-reserve.
  if (record.size() > byte_limit_ - bytes_written_ - trailer_reserve_bytes) {
    truncate("byte_limit");
    return false;
  }
  write_line(record);
  ++records_written_;
  return true;
}

void RoutingTraceWriter::finish(bool complete, const std::string& reason) {
  if (failed_) throw std::runtime_error("routing JSONL has a previous I/O failure");
  if (finished_) return;
  std::string fields = "\"complete\":";
  fields += complete && !truncated_ ? "true" : "false";
  fields += ",\"truncated\":";
  fields += truncated_ ? "true" : "false";
  fields += ",\"records\":" + std::to_string(records_written_);
  fields += ",\"bytes_before_complete\":" + std::to_string(bytes_written_);
  const std::string_view completion_reason = truncated_ ? std::string_view(truncation_reason_) :
      reason.empty() && !complete ? std::string_view("incomplete") : std::string_view(reason);
  if (!completion_reason.empty()) {
    fields += ",\"reason\":" + quoted_ascii(completion_reason.substr(0, 128));
  }
  const auto trailer = line("complete", fields);
  if (trailer.size() > trailer_reserve_bytes || trailer.size() > byte_limit_ - bytes_written_) {
    throw std::logic_error("routing completion exceeded reserved space");
  }
  write_line(trailer);
  const auto descriptor = fd_;
  fd_ = -1;
  if (::close(descriptor) < 0) {
    failed_ = true;
    throw std::system_error(errno, std::generic_category(), "close routing JSONL");
  }
  finished_ = true;
}

}  // namespace pmi
