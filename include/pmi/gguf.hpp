#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace pmi {

enum class GgufType : uint32_t {
  UINT8=0, INT8=1, UINT16=2, INT16=3, UINT32=4, INT32=5,
  FLOAT32=6, BOOL=7, STRING=8, ARRAY=9, UINT64=10, INT64=11,
  FLOAT64=12
};

struct MetadataValue {
  GgufType type{};
  std::string text;
};

struct TensorInfo {
  std::string name;
  std::vector<uint64_t> shape;
  uint32_t ggml_type{};
  uint64_t offset{};
  uint64_t byte_length{};
  uint64_t absolute_offset{};
};

struct GgufFile {
  uint32_t version{};
  std::unordered_map<std::string, MetadataValue> metadata;
  std::vector<TensorInfo> tensors;
  uint64_t tensor_data_offset{};
  uint64_t file_size{};
};

class GgufReader {
 public:
  explicit GgufReader(const std::filesystem::path& path);
  GgufFile inspect() const;
 private:
  std::filesystem::path path_;
  mutable std::ifstream in_;
  uint64_t file_size_{};
  template<class T> T read_scalar() const;
  std::string read_string() const;
  MetadataValue read_value(GgufType type) const;
  void seek(uint64_t offset) const;
};

std::string ggml_type_name(uint32_t type);
uint64_t ggml_tensor_nbytes(uint32_t type, const std::vector<uint64_t>& shape);

} // namespace pmi
