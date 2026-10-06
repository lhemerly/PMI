#pragma once
#include "pmi/gguf.hpp"
#include <string>
#include <vector>

namespace pmi {
enum class TensorClass { ALWAYS_ACTIVE, ROUTED_EXPERT, SPARSE_LOOKUP, OPTIONAL, UNKNOWN };
struct ManifestTensor : TensorInfo {
  // Name-based classes are advisory. Connectors validate architecture layouts.
  TensorClass semantic_class{TensorClass::UNKNOWN};
  std::string source_tensor_name;
  int layer{-1};
  int expert_id{-1};
  uint64_t required_alignment{1};
};
struct ManifestShard {
  uint32_t shard_id{};
  std::filesystem::path source_path;
  uint64_t file_size{};
  uint64_t tensor_data_offset{};
  uint64_t tensor_count{}; // Physical GGUF tensors, before expert slicing.
};
struct ModelManifest {
  std::string model_architecture;
  std::string quantization;
  uint64_t file_size{}; // Sum of shard file sizes.
  uint64_t tensor_data_offset{}; // Shard 0; use shards for individual offsets.
  std::vector<ManifestTensor> tensors;
  std::vector<ManifestShard> shards; // Ordered by shard_id.
};
ModelManifest build_manifest(const GgufFile& file);
ModelManifest build_manifest(const std::vector<GgufFile>& files);
std::string class_name(TensorClass c);
std::string manifest_json(const ModelManifest& manifest);
} // namespace pmi
