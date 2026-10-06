#pragma once
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string_view>
namespace pmi {
// Optional process-interruption trace, not a power-loss journal. Completed
// write(2) records survive TERM/KILL; a killed write may leave one partial tail.
// Caller supplies trusted JSON members without braces, paths or model content.
// Append is serialized; callers must never reenter it. I/O errors abort capture
// explicitly. Detail limits preserve space for evaluation/completion records.
class ProgressWriter final {
 public:
  ProgressWriter(const std::filesystem::path&, uint64_t bytes = 32*1024*1024,
                 uint64_t rows = 262144);
  ~ProgressWriter() noexcept;
  bool append(std::string_view phase, std::string_view fields = {}, bool detail = false);
  void finish();
 private:
  void write(std::string_view phase, std::string_view fields);
  int fd_{-1};
  uint64_t bytes_{}, rows_{}, byte_limit_, row_limit_, detail_bytes_, detail_rows_;
  bool details_truncated_{}, finished_{}, failed_{};
  std::mutex mutex_;
};
}
