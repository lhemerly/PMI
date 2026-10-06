#include "pmi/telemetry.hpp"
#include <fstream>
#include <sstream>
#include <string>
namespace pmi {
ProcessMemory process_memory() {
  ProcessMemory result;
#ifdef __linux__
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    std::istringstream fields(line);
    std::string key;
    uint64_t value;
    if (fields >> key >> value) {
      if (key == "VmRSS:") result.rss_bytes = value * 1024;
      if (key == "VmHWM:") result.peak_rss_bytes = value * 1024;
    }
  }
  std::ifstream io("/proc/self/io");
  while (std::getline(io, line)) {
    std::istringstream fields(line);
    std::string key;
    uint64_t value;
    if ((fields >> key >> value) && key == "read_bytes:") result.storage_read_bytes = value;
  }
#endif
  return result;
}
}
