#include "pmi/manifest.hpp"
#include <algorithm>
#include <charconv>
#include <cctype>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace pmi {
namespace {
const MetadataValue* metadata(const GgufFile& file, const std::string& key) {
  const auto it = file.metadata.find(key);
  return it == file.metadata.end() ? nullptr : &it->second;
}

std::string text_metadata(const GgufFile& file, const std::string& key) {
  const auto* value = metadata(file, key);
  return value ? value->text : "";
}

uint64_t unsigned_value(const MetadataValue& value, const std::string& key) {
  switch (value.type) {
    case GgufType::UINT8: case GgufType::INT8: case GgufType::UINT16:
    case GgufType::INT16: case GgufType::UINT32: case GgufType::INT32:
    case GgufType::UINT64: case GgufType::INT64: break;
    default: throw std::runtime_error("GGUF metadata must be an integer: " + key);
  }
  uint64_t result = 0;
  const auto parsed = std::from_chars(value.text.data(), value.text.data() + value.text.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.text.data() + value.text.size())
    throw std::runtime_error("invalid nonnegative GGUF metadata: " + key);
  return result;
}

uint64_t typed_value(const MetadataValue& value, GgufType type, uint64_t maximum,
                     const std::string& key) {
  const auto result = unsigned_value(value, key);
  if (value.type != type || result > maximum)
    throw std::runtime_error("invalid GGUF metadata type or value: " + key);
  return result;
}

uint64_t checked_add(uint64_t a, uint64_t b, const char* what) {
  if (b > std::numeric_limits<uint64_t>::max() - a) throw std::runtime_error(what);
  return a + b;
}

uint64_t alignment_of(const GgufFile& file) {
  const auto* value = metadata(file, "general.alignment");
  const auto alignment = value ? typed_value(*value, GgufType::UINT32,
      std::numeric_limits<uint32_t>::max(), "general.alignment") : 32;
  if (alignment == 0 || (alignment & (alignment - 1)) != 0)
    throw std::runtime_error("GGUF alignment must be a nonzero power of two");
  return alignment;
}

struct SplitInfo {
  uint32_t id{};
  uint32_t count{1};
  std::optional<uint64_t> tensor_count;
};

SplitInfo split_info(const GgufFile& file) {
  const auto* count = metadata(file, "split.count");
  const auto* id = metadata(file, "split.no");
  const auto* tensors = metadata(file, "split.tensors.count");
  SplitInfo info;
  if (!count) {
    if (id || tensors) throw std::runtime_error("incomplete GGUF split metadata: missing split.count");
    return info;
  }
  const auto raw_count = typed_value(*count, GgufType::UINT16,
      std::numeric_limits<uint16_t>::max(), "split.count");
  // The upstream merger leaves split.count=0; it denotes one unsplit file.
  info.count = static_cast<uint32_t>(raw_count == 0 ? 1 : raw_count);
  if (info.count > 1 && (!id || !tensors)) throw std::runtime_error("incomplete GGUF split metadata");
  if (id) info.id = static_cast<uint32_t>(typed_value(*id, GgufType::UINT16,
      std::numeric_limits<uint16_t>::max(), "split.no"));
  if (info.id >= info.count) throw std::runtime_error("GGUF split.no outside split.count");
  if (tensors) info.tensor_count = typed_value(*tensors, GgufType::INT32,
      std::numeric_limits<int32_t>::max(), "split.tensors.count");
  return info;
}

void check_shared_metadata(const GgufFile& primary, const GgufFile& file, const std::string& key) {
  const auto* value = metadata(file, key);
  if (!value) return; // Official split files normally only have metadata in shard 0.
  const auto* expected = metadata(primary, key);
  if (!expected || value->type != expected->type || value->text != expected->text)
    throw std::runtime_error("GGUF shards disagree on " + key);
}

int name_index(const std::ssub_match& match) {
  const auto value = match.str();
  int result = -1;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  return parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() ? result : -1;
}

void classify(ManifestTensor& tensor) {
  static const std::regex layer_re(".*\\.layers\\.([0-9]+)\\..*|blk\\.([0-9]+)\\..*");
  static const std::regex expert_re(".*(?:experts|expert)[._]([0-9]+)(?:[._].*)?");
  std::smatch match;
  if (std::regex_match(tensor.name, match, layer_re))
    tensor.layer = name_index(match[match[1].matched ? 1 : 2]);
  if (std::regex_match(tensor.name, match, expert_re)) tensor.expert_id = name_index(match[1]);
  auto name = tensor.name;
  for (auto& c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (name.find("vision") != std::string::npos || name.find("mtp") != std::string::npos)
    tensor.semantic_class = TensorClass::OPTIONAL;
  else if (name.find("shexp") != std::string::npos || name.find("shared_expert") != std::string::npos)
    tensor.semantic_class = TensorClass::ALWAYS_ACTIVE;
  else if (tensor.expert_id >= 0 || name.find("experts") != std::string::npos ||
           name.find("_exps") != std::string::npos || name.find("expert_") != std::string::npos)
    tensor.semantic_class = TensorClass::ROUTED_EXPERT;
  else if (name.find("ngram") != std::string::npos || name.find("ple_") != std::string::npos ||
           name.find("per_layer_token_embd") != std::string::npos || name.find("sparse_embedding") != std::string::npos)
    tensor.semantic_class = TensorClass::SPARSE_LOOKUP;
  else if (tensor.layer >= 0 || name.find("router") != std::string::npos ||
           name.find("norm") != std::string::npos || name.find("token_embd") != std::string::npos ||
           name.find("output") != std::string::npos)
    tensor.semantic_class = TensorClass::ALWAYS_ACTIVE;
}

std::string escape(const std::string& text) {
  std::string out;
  for (unsigned char c : text) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          const char* hex = "0123456789abcdef";
          out += "\\u00";
          out += hex[c >> 4];
          out += hex[c & 15];
        } else out += static_cast<char>(c);
    }
  }
  return out;
}
} // namespace

ModelManifest build_manifest(const GgufFile& file) {
  return build_manifest(std::vector<GgufFile>{file});
}

ModelManifest build_manifest(const std::vector<GgufFile>& files) {
  if (files.empty()) throw std::runtime_error("cannot build a manifest without GGUF files");
  std::vector<const GgufFile*> ordered(files.size(), nullptr);
  std::optional<uint64_t> expected_tensor_count;
  std::unordered_set<std::string> source_paths;
  for (const auto& file : files) {
    const auto split = split_info(file);
    if (split.count != files.size()) throw std::runtime_error("GGUF split count does not match supplied files");
    if (ordered[split.id]) throw std::runtime_error("duplicate GGUF split.no");
    ordered[split.id] = &file;
    if (split.tensor_count) {
      if (expected_tensor_count && expected_tensor_count != split.tensor_count)
        throw std::runtime_error("GGUF shards disagree on split.tensors.count");
      expected_tensor_count = split.tensor_count;
    }
    if (!file.source_path.empty() && !source_paths.insert(
        std::filesystem::weakly_canonical(file.source_path).string()).second)
      throw std::runtime_error("duplicate GGUF shard source path");
  }
  for (const auto* file : ordered)
    if (!file) throw std::runtime_error("missing GGUF shard");

  const auto& primary = *ordered.front();
  const auto* architecture = metadata(primary, "general.architecture");
  if ((architecture && (architecture->type != GgufType::STRING || architecture->text.empty())) ||
      (files.size() > 1 && !architecture))
    throw std::runtime_error("missing or invalid general.architecture in GGUF shard 0");
  ModelManifest manifest;
  manifest.model_architecture = text_metadata(primary, "general.architecture");
  manifest.quantization = text_metadata(primary, "general.file_type");
  manifest.tensor_data_offset = primary.tensor_data_offset;
  std::unordered_set<std::string> source_names;
  uint64_t physical_tensor_count = 0;
  for (size_t id = 0; id < ordered.size(); ++id) {
    const auto& file = *ordered[id];
    if (file.version != primary.version) throw std::runtime_error("GGUF shards have mismatched versions");
    check_shared_metadata(primary, file, "general.architecture");
    check_shared_metadata(primary, file, "general.file_type");
    for (const char* key : {"expert_count", "expert_used_count", "expert_weights_scale", "expert_group_count"})
      check_shared_metadata(primary, file, manifest.model_architecture + "." + key);
    const auto alignment = alignment_of(file);
    if (file.tensor_data_offset % alignment != 0) throw std::runtime_error("unaligned GGUF data offset");
    manifest.file_size = checked_add(manifest.file_size, file.file_size, "aggregate GGUF file size overflow");
    physical_tensor_count = checked_add(physical_tensor_count, file.tensors.size(), "GGUF tensor count overflow");
    manifest.shards.push_back({static_cast<uint32_t>(id), file.source_path, file.file_size,
                               file.tensor_data_offset, static_cast<uint64_t>(file.tensors.size())});
    std::vector<const TensorInfo*> ranges;
    for (const auto& tensor : file.tensors) {
      if (tensor.name.empty() || !source_names.insert(tensor.name).second)
        throw std::runtime_error("empty or duplicate GGUF tensor name: " + tensor.name);
      if (tensor.offset % alignment != 0) throw std::runtime_error("unaligned GGUF tensor offset");
      const auto absolute_offset = checked_add(file.tensor_data_offset, tensor.offset, "tensor offset overflow");
      if (tensor.absolute_offset != absolute_offset || absolute_offset > file.file_size ||
          tensor.byte_length > file.file_size - absolute_offset)
        throw std::runtime_error("tensor range outside GGUF or inconsistent offset");
      if (ggml_tensor_nbytes(tensor.ggml_type, tensor.shape) != tensor.byte_length)
        throw std::runtime_error("tensor byte length disagrees with shape and quantization");
      ranges.push_back(&tensor);
    }
    std::sort(ranges.begin(), ranges.end(), [](const auto* a, const auto* b) {
      return a->absolute_offset < b->absolute_offset;
    });
    for (size_t i = 1; i < ranges.size(); ++i)
      if (ranges[i]->absolute_offset - ranges[i - 1]->absolute_offset < ranges[i - 1]->byte_length)
        throw std::runtime_error("overlapping GGUF tensor ranges");
  }
  if (expected_tensor_count && *expected_tensor_count != physical_tensor_count)
    throw std::runtime_error("GGUF split.tensors.count disagrees with physical tensor count");

  std::unordered_set<std::string> logical_names;
  for (size_t id = 0; id < ordered.size(); ++id) {
    const auto& file = *ordered[id];
    for (const auto& tensor : file.tensors) {
      ManifestTensor base;
      static_cast<TensorInfo&>(base) = tensor;
      base.shard_id = static_cast<uint32_t>(id);
      base.source_tensor_name = tensor.name;
      base.required_alignment = alignment_of(file);
      classify(base);
      auto name = base.name;
      for (auto& c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      // Preserve the legacy packed-expert convention. A model connector must
      // validate these advisory slices against its exact tensor layout.
      if (base.semantic_class == TensorClass::ROUTED_EXPERT && base.expert_id < 0 &&
          name.find("exps") != std::string::npos && base.shape.size() >= 2) {
        const auto count = base.shape.back();
        if (count > static_cast<uint64_t>(std::numeric_limits<int>::max()))
          throw std::runtime_error("packed expert count exceeds representable expert IDs");
        const auto* expected = metadata(primary, manifest.model_architecture + ".expert_count");
        if (expected && unsigned_value(*expected, manifest.model_architecture + ".expert_count") != count)
          throw std::runtime_error("expert tensor dimension disagrees with model expert_count");
        auto slice_shape = base.shape;
        slice_shape.pop_back();
        const auto slice_bytes = ggml_tensor_nbytes(base.ggml_type, slice_shape);
        if (slice_bytes > std::numeric_limits<uint64_t>::max() / count || slice_bytes * count != base.byte_length)
          throw std::runtime_error("packed expert tensor size mismatch");
        for (uint64_t expert = 0; expert < count; ++expert) {
          auto slice = base;
          slice.expert_id = static_cast<int>(expert);
          slice.name += "#expert:" + std::to_string(expert);
          if (source_names.count(slice.name) || !logical_names.insert(slice.name).second)
            throw std::runtime_error("duplicate logical GGUF tensor name: " + slice.name);
          slice.shape = slice_shape;
          const auto displacement = slice_bytes * expert; // Bounded by checked total above.
          slice.offset = checked_add(tensor.offset, displacement, "expert slice offset overflow");
          slice.absolute_offset = checked_add(tensor.absolute_offset, displacement, "expert slice offset overflow");
          slice.byte_length = slice_bytes;
          slice.required_alignment = 1;
          manifest.tensors.push_back(std::move(slice));
        }
      } else {
        if (!logical_names.insert(base.name).second) throw std::runtime_error("duplicate logical GGUF tensor name");
        manifest.tensors.push_back(std::move(base));
      }
    }
  }
  return manifest;
}

std::string class_name(TensorClass type) {
  switch (type) {
    case TensorClass::ALWAYS_ACTIVE: return "ALWAYS_ACTIVE";
    case TensorClass::ROUTED_EXPERT: return "ROUTED_EXPERT";
    case TensorClass::SPARSE_LOOKUP: return "SPARSE_LOOKUP";
    case TensorClass::OPTIONAL: return "OPTIONAL";
    default: return "UNKNOWN";
  }
}

std::string manifest_json(const ModelManifest& manifest) {
  std::ostringstream out;
  out << "{\"model_architecture\":\"" << escape(manifest.model_architecture)
      << "\",\"quantization\":\"" << escape(manifest.quantization)
      << "\",\"file_size\":" << manifest.file_size
      << ",\"tensor_data_offset\":" << manifest.tensor_data_offset << ",\"shards\":[";
  for (size_t i = 0; i < manifest.shards.size(); ++i) {
    const auto& shard = manifest.shards[i];
    if (i) out << ',';
    out << "{\"shard_id\":" << shard.shard_id << ",\"source_path\":\"" << escape(shard.source_path.string())
        << "\",\"file_size\":" << shard.file_size << ",\"tensor_data_offset\":" << shard.tensor_data_offset
        << ",\"tensor_count\":" << shard.tensor_count << '}';
  }
  out << "],\"tensors\":[";
  for (size_t i = 0; i < manifest.tensors.size(); ++i) {
    const auto& tensor = manifest.tensors[i];
    if (i) out << ',';
    out << "{\"name\":\"" << escape(tensor.name) << "\",\"source_tensor\":\"" << escape(tensor.source_tensor_name)
        << "\",\"shard_id\":" << tensor.shard_id << ",\"dtype\":\"" << ggml_type_name(tensor.ggml_type)
        << "\",\"offset\":" << tensor.absolute_offset << ",\"relative_offset\":" << tensor.offset
        << ",\"byte_length\":" << tensor.byte_length << ",\"shape\":[";
    for (size_t dimension = 0; dimension < tensor.shape.size(); ++dimension) {
      if (dimension) out << ',';
      out << tensor.shape[dimension];
    }
    out << "],\"class\":\"" << class_name(tensor.semantic_class) << "\",\"layer\":" << tensor.layer
        << ",\"expert_id\":" << tensor.expert_id << '}';
  }
  out << "]}";
  return out.str();
}
} // namespace pmi
