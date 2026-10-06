#include "pmi/routing_trace.hpp"
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
  bool caught = false;
  try { action(); } catch (const std::exception&) { caught = true; }
  require(caught, "expected routing writer failure");
}
struct Directory {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("pmi-routing-" + std::to_string(::getpid()) + "-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  Directory() { std::filesystem::create_directories(path); }
  ~Directory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};
std::string read(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "could not read routing file");
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
std::vector<std::string> lines(const std::filesystem::path& path) {
  std::ifstream input(path);
  std::vector<std::string> result;
  for (std::string line; std::getline(input, line);) result.push_back(line);
  return result;
}
bool contains(const std::string& value, const std::string& part) {
  return value.find(part) != std::string::npos;
}
void is_incomplete(const std::filesystem::path& path, bool truncated) {
  const auto records = lines(path);
  require(!records.empty(), "missing routing records");
  const auto& trailer = records.back();
  require(contains(trailer, "\"type\":\"complete\""), "missing explicit completion trailer");
  require(contains(trailer, "\"complete\":false"), "incomplete run marked complete");
  require(contains(trailer, truncated ? "\"truncated\":true" : "\"truncated\":false"),
          "incorrect truncation flag");
}
}  // namespace

int main() {
  try {
    Directory directory;
    const auto normal = directory.path / "normal.jsonl";
    {
      pmi::RoutingTraceWriter writer(normal, "run-1", "\"model\":{\"revision\":null}");
      require(writer.accepting() && !writer.truncated(), "new writer not accepting records");
      require(writer.append("routing", "\"eval_id\":0,\"layer\":3,\"expert_ids\":[7,1],\"routing_weights\":null"),
              "routing record rejected");
      require(writer.append("metrics", "\"metrics\":{\"io_bytes\":128}"), "metrics record rejected");
      require(writer.records_written() == 2, "incorrect record count");
      writer.finish();
      const auto complete_size = writer.bytes_written();
      writer.finish(false, "second finish must not change the existing result");
      writer.stop_capture();
      require(!writer.truncated(), "scope cap retroactively changed a finished run");
      require(writer.bytes_written() == complete_size, "finish was not idempotent");
      require(!writer.accepting(), "finished writer still accepting");
      require(std::filesystem::file_size(normal) == complete_size, "byte accounting mismatch");
      rejects([&] { writer.append("routing", ""); });
    }
    const auto normal_lines = lines(normal);
    require(normal_lines.size() == 4, "unexpected JSONL record count");
    require(normal_lines[0] == "{\"schema\":\"pmi.routing.v1\",\"type\":\"run\",\"run_id\":\"run-1\",\"model\":{\"revision\":null}}",
            "run schema or metadata changed");
    require(contains(normal_lines[1], "\"expert_ids\":[7,1],\"routing_weights\":null"),
            "routing payload changed");
    require(contains(normal_lines.back(), "\"complete\":true,\"truncated\":false,\"records\":2"),
            "successful completion missing");
    struct stat info {};
    require(::stat(normal.c_str(), &info) == 0 && (info.st_mode & 0077) == 0,
            "routing capture is accessible to group or other users");

    const auto original = read(normal);
    rejects([&] { pmi::RoutingTraceWriter writer(normal, "other", ""); });
    require(read(normal) == original, "exclusive creation overwrote an existing capture");
    const auto link = directory.path / "link.jsonl";
    std::filesystem::create_symlink(normal, link);
    rejects([&] { pmi::RoutingTraceWriter writer(link, "other", ""); });
    require(read(normal) == original, "exclusive creation followed a symlink");
    rejects([&] { pmi::RoutingTraceWriter writer(directory.path / "missing" / "run.jsonl", "run", ""); });

    const auto invalid = directory.path / "invalid.jsonl";
    rejects([&] { pmi::RoutingTraceWriter writer(invalid, "", ""); });
    rejects([&] { pmi::RoutingTraceWriter writer(invalid, std::string(129, 'r'), ""); });
    rejects([&] { pmi::RoutingTraceWriter writer(invalid, "run\nid", ""); });
    rejects([&] { pmi::RoutingTraceWriter writer(invalid, "run", "\"metadata\":1\n"); });
    rejects([&] { pmi::RoutingTraceWriter writer(invalid, "run", std::string(65537, ' ')); });
    rejects([&] { pmi::RoutingTraceWriter writer(invalid, "run", "", 1); });
    require(!std::filesystem::exists(invalid), "invalid constructor arguments created a file");

    const auto row_cap = directory.path / "row-cap.jsonl";
    {
      pmi::RoutingTraceWriter writer(row_cap, "run", "", 8192, 1);
      require(writer.append("routing", "\"layer\":0"), "first row did not fit");
      require(!writer.append("metrics", "\"metrics\":{}"), "record cap ignored");
      require(writer.truncated() && !writer.accepting(), "record cap did not stop capture");
      require(!writer.append("routing", "\"layer\":1"), "capture resumed after cap");
      require(writer.records_written() == 1, "capped records were counted");
      writer.finish(true);
    }
    require(lines(row_cap).size() == 3, "unexpected data beyond row cap");
    is_incomplete(row_cap, true);
    require(contains(read(row_cap), "\"reason\":\"row_limit\""), "row cap reason missing");

    const auto byte_cap = directory.path / "byte-cap.jsonl";
    {
      pmi::RoutingTraceWriter measure(directory.path / "measure.jsonl", "run", "");
      const auto limit = measure.bytes_written() + pmi::RoutingTraceWriter::trailer_reserve_bytes;
      pmi::RoutingTraceWriter writer(byte_cap, "run", "", limit);
      require(!writer.append("routing", "\"layer\":0"), "byte cap ignored");
      writer.finish();
      require(writer.bytes_written() <= limit && std::filesystem::file_size(byte_cap) <= limit,
              "completion trailer escaped byte budget");
      require(writer.records_written() == 0, "oversized record was counted");
      measure.finish();
    }
    is_incomplete(byte_cap, true);
    require(contains(read(byte_cap), "\"reason\":\"byte_limit\""), "byte cap reason missing");

    const auto record_cap = directory.path / "record-cap.jsonl";
    {
      pmi::RoutingTraceWriter writer(record_cap, "run", "");
      require(!writer.append("routing", std::string(65537, ' ')), "record size cap ignored");
      writer.finish();
    }
    is_incomplete(record_cap, true);
    require(contains(read(record_cap), "\"reason\":\"record_size_limit\""), "record size cap reason missing");

    const auto unfinished = directory.path / "unfinished.jsonl";
    {
      pmi::RoutingTraceWriter writer(unfinished, "run", "");
      require(writer.append("routing", "\"layer\":0"), "destructor test append failed");
    }
    is_incomplete(unfinished, false);
    require(contains(read(unfinished), "\"reason\":\"cancelled\""), "destructor reason missing");

    const auto incomplete = directory.path / "incomplete.jsonl";
    {
      pmi::RoutingTraceWriter writer(incomplete, "run", "");
      writer.finish(false);
    }
    is_incomplete(incomplete, false);
    require(contains(read(incomplete), "\"reason\":\"incomplete\""), "default incomplete reason missing");

    const auto scope_cap = directory.path / "scope-cap.jsonl";
    {
      pmi::RoutingTraceWriter writer(scope_cap, "run", "");
      require(writer.append("routing", "\"layer\":0"), "scope cap initial row failed");
      writer.stop_capture();
      writer.stop_capture();
      require(writer.truncated() && !writer.accepting(), "scope cap did not stop capture");
      require(!writer.append("routing", "\"layer\":1"), "capture resumed after scope cap");
      require(writer.records_written() == 1, "scope-capped record was counted");
      writer.finish(true);
      const auto complete_size = writer.bytes_written();
      writer.stop_capture();
      require(writer.bytes_written() == complete_size, "scope cap changed a finished capture");
    }
    is_incomplete(scope_cap, true);
    require(contains(read(scope_cap), "\"reason\":\"scope_limit\""), "scope cap reason missing");

    const auto prior_cap = directory.path / "prior-cap.jsonl";
    {
      pmi::RoutingTraceWriter writer(prior_cap, "run", "", 8192, 0);
      require(!writer.append("routing", "\"layer\":0"), "zero-row limit ignored");
      writer.stop_capture();
      writer.finish();
    }
    is_incomplete(prior_cap, true);
    require(contains(read(prior_cap), "\"reason\":\"row_limit\""), "scope cap replaced first limit reason");

    const auto escaped = directory.path / "escaped.jsonl";
    {
      pmi::RoutingTraceWriter writer(escaped, "quote\"slash\\", "");
      rejects([&] { writer.append("complete", ""); });
      rejects([&] { writer.append("routing", "\"layer\":0\n{}"); });
      require(writer.accepting(), "invalid argument disabled an otherwise healthy writer");
      require(writer.append("routing", "\"note\":\"escaped\\nnewline\""), "escaped newline rejected");
      writer.finish(false, std::string(200, '\1'));
    }
    const auto escaped_lines = lines(escaped);
    require(escaped_lines.size() == 3 && contains(escaped_lines[0], "quote\\\"slash\\\\"),
            "run ID escaping broke JSONL framing");
    std::string bounded_reason;
    for (size_t i = 0; i < 128; ++i) bounded_reason += "\\u0001";
    require(contains(escaped_lines.back(), "\"reason\":\"" + bounded_reason + "\""),
            "reason was not escaped and bounded");
    is_incomplete(escaped, false);

    // A process-local file size limit gives a deterministic write error without
    // depending on device files, storage exhaustion, or descriptor internals.
    const auto failure = directory.path / "write-failure.jsonl";
    const auto child = ::fork();
    require(child >= 0, "fork failed");
    if (child == 0) {
      int status = 1;
      try {
        pmi::RoutingTraceWriter writer(failure, "run", "");
        std::signal(SIGXFSZ, SIG_IGN);
        const rlimit limit {1, 1};
        if (::setrlimit(RLIMIT_FSIZE, &limit) != 0) ::_exit(2);
        rejects([&] { writer.append("routing", "\"layer\":0"); });
        require(!writer.accepting(), "I/O failure did not disable capture");
        rejects([&] { writer.finish(); });
        status = 0;
      } catch (...) {}
      ::_exit(status);
    }
    int child_status = 0;
    require(::waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) &&
            WEXITSTATUS(child_status) == 0, "explicit I/O failure handling failed");
    require(!contains(read(failure), "\"type\":\"complete\""), "failed I/O fabricated completion");

    std::cout << "routing JSONL writer tests passed (exclusive capture, caps, completion, errors)\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
