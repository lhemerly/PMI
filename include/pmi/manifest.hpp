#pragma once
#include "pmi/gguf.hpp"
#include <string>
#include <vector>

namespace pmi {
enum class TensorClass { ALWAYS_ACTIVE, ROUTED_EXPERT, SPARSE_LOOKUP, OPTIONAL, UNKNOWN };
struct ManifestTensor : TensorInfo {
  TensorClass semantic_class{TensorClass::UNKNOWN};
  std::string source_tensor_name;
  int layer{-1};
  int expert_id{-1};
  uint64_t required_alignment{1};
};
struct ModelManifest {
  std::string model_architecture;
  std::string quantization;
  uint64_t file_size{};
  uint64_t tensor_data_offset{};
  std::vector<ManifestTensor> tensors;
};
ModelManifest build_manifest(const GgufFile& file);
std::string class_name(TensorClass c);
std::string manifest_json(const ModelManifest& manifest);
} // namespace pmi
