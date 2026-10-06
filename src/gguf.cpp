#include "pmi/gguf.hpp"
#include <algorithm>
#include <charconv>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace pmi {
namespace {
constexpr uint32_t MAGIC = 0x46554747;

template<class T> std::string floating_metadata(T value) {
  // Metadata is consumed by execution code, so preserve the original float
  // exactly on a text round trip, including tiny nonzero routing scales.
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(std::numeric_limits<T>::max_digits10) << value;
  return out.str();
}

uint64_t align_up(uint64_t n, uint64_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0)
    throw std::runtime_error("GGUF alignment must be a nonzero power of two");
  if (n > std::numeric_limits<uint64_t>::max() - (alignment - 1))
    throw std::runtime_error("GGUF alignment overflow");
  return (n + alignment - 1) / alignment * alignment;
}

uint64_t integer_metadata(const MetadataValue& value, GgufType expected,
                          const std::string& key) {
  uint64_t result = 0;
  const auto parsed = std::from_chars(value.text.data(), value.text.data() + value.text.size(), result);
  if (value.type != expected || parsed.ec != std::errc{} ||
      parsed.ptr != value.text.data() + value.text.size())
    throw std::runtime_error("invalid GGUF integer metadata: " + key);
  return result;
}
} // namespace

GgufReader::GgufReader(const std::filesystem::path& path)
    : path_(path), in_(path, std::ios::binary) {
  if (!in_) throw std::runtime_error("cannot open GGUF: " + path.string());
  in_.seekg(0, std::ios::end);
  const auto size = in_.tellg();
  if (size < 0) throw std::runtime_error("cannot determine GGUF file size");
  file_size_ = static_cast<uint64_t>(size);
  in_.seekg(0);
}

void GgufReader::seek(uint64_t offset) const {
  if (offset > file_size_ || offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()))
    throw std::runtime_error("GGUF offset outside file");
  in_.clear();
  in_.seekg(static_cast<std::streamoff>(offset));
  if (!in_) throw std::runtime_error("cannot seek in GGUF");
}

template<class T> T GgufReader::read_scalar() const {
  T value{};
  in_.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in_) throw std::runtime_error("truncated GGUF");
  return value;
}

std::string GgufReader::read_string() const {
  const auto length = read_scalar<uint64_t>();
  const auto position = in_.tellg();
  if (position < 0 || static_cast<uint64_t>(position) > file_size_ ||
      length > file_size_ - static_cast<uint64_t>(position) ||
      length > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
      length > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()))
    throw std::runtime_error("invalid GGUF string length");
  std::string value(static_cast<size_t>(length), '\0');
  in_.read(value.data(), static_cast<std::streamsize>(length));
  if (!in_) throw std::runtime_error("truncated GGUF string");
  return value;
}

MetadataValue GgufReader::read_value(GgufType type) const {
  MetadataValue value{type, {}};
  switch (type) {
    case GgufType::UINT8: value.text = std::to_string(read_scalar<uint8_t>()); break;
    case GgufType::INT8: value.text = std::to_string(read_scalar<int8_t>()); break;
    case GgufType::UINT16: value.text = std::to_string(read_scalar<uint16_t>()); break;
    case GgufType::INT16: value.text = std::to_string(read_scalar<int16_t>()); break;
    case GgufType::UINT32: value.text = std::to_string(read_scalar<uint32_t>()); break;
    case GgufType::INT32: value.text = std::to_string(read_scalar<int32_t>()); break;
    case GgufType::UINT64: value.text = std::to_string(read_scalar<uint64_t>()); break;
    case GgufType::INT64: value.text = std::to_string(read_scalar<int64_t>()); break;
    case GgufType::FLOAT32: value.text = floating_metadata(read_scalar<float>()); break;
    case GgufType::FLOAT64: value.text = floating_metadata(read_scalar<double>()); break;
    case GgufType::BOOL: value.text = read_scalar<uint8_t>() ? "true" : "false"; break;
    case GgufType::STRING: value.text = read_string(); break;
    case GgufType::ARRAY: {
      const auto element_type = static_cast<GgufType>(read_scalar<uint32_t>());
      const auto count = read_scalar<uint64_t>();
      if (element_type == GgufType::ARRAY ||
          static_cast<uint32_t>(element_type) > static_cast<uint32_t>(GgufType::FLOAT64) ||
          count > file_size_)
        throw std::runtime_error("invalid GGUF metadata array");
      for (uint64_t i = 0; i < count; ++i) {
        const auto item = read_value(element_type);
        if (i < 16) {
          if (!value.text.empty()) value.text += ",";
          value.text += item.text;
        }
      }
      value.text = "[" + value.text + (count > 16 ? ",..." : "") + "]";
      break;
    }
    default: throw std::runtime_error("unknown GGUF metadata type");
  }
  return value;
}

GgufFile GgufReader::inspect() const {
  seek(0);
  if (read_scalar<uint32_t>() != MAGIC) throw std::runtime_error("not a GGUF file");
  GgufFile file;
  file.source_path = std::filesystem::absolute(path_).lexically_normal();
  file.version = read_scalar<uint32_t>();
  if (file.version < 2 || file.version > 3) throw std::runtime_error("unsupported GGUF version");
  const auto tensor_count = read_scalar<uint64_t>();
  const auto metadata_count = read_scalar<uint64_t>();
  // A tensor descriptor is at least 24 bytes; a metadata entry at least 13.
  if (tensor_count > 100000000 || tensor_count > file_size_ / 24 ||
      metadata_count > 1000000 || metadata_count > file_size_ / 13)
    throw std::runtime_error("implausible GGUF counts");
  for (uint64_t i = 0; i < metadata_count; ++i) {
    auto key = read_string();
    const auto type = static_cast<GgufType>(read_scalar<uint32_t>());
    auto value = read_value(type);
    if (!file.metadata.emplace(key, std::move(value)).second)
      throw std::runtime_error("duplicate GGUF metadata key: " + key);
  }
  uint32_t shard_id = 0;
  if (const auto it = file.metadata.find("split.no"); it != file.metadata.end()) {
    const auto id = integer_metadata(it->second, GgufType::UINT16, "split.no");
    if (id > std::numeric_limits<uint16_t>::max()) throw std::runtime_error("invalid GGUF split.no");
    shard_id = static_cast<uint32_t>(id);
  }
  file.tensors.reserve(static_cast<size_t>(tensor_count));
  std::unordered_set<std::string> tensor_names;
  for (uint64_t i = 0; i < tensor_count; ++i) {
    TensorInfo tensor;
    tensor.shard_id = shard_id;
    tensor.name = read_string();
    if (tensor.name.empty() || !tensor_names.insert(tensor.name).second)
      throw std::runtime_error("empty or duplicate GGUF tensor name: " + tensor.name);
    const auto dimensions = read_scalar<uint32_t>();
    if (dimensions == 0 || dimensions > 4) throw std::runtime_error("invalid GGUF dimensions");
    tensor.shape.resize(dimensions);
    for (auto& dimension : tensor.shape) dimension = read_scalar<uint64_t>();
    tensor.ggml_type = read_scalar<uint32_t>();
    tensor.offset = read_scalar<uint64_t>();
    file.tensors.push_back(std::move(tensor));
  }
  uint64_t alignment = 32;
  if (const auto it = file.metadata.find("general.alignment"); it != file.metadata.end())
    alignment = integer_metadata(it->second, GgufType::UINT32, "general.alignment");
  const auto position = in_.tellg();
  if (position < 0) throw std::runtime_error("invalid GGUF data offset");
  file.tensor_data_offset = align_up(static_cast<uint64_t>(position), alignment);
  file.file_size = file_size_;
  // A metadata-only first shard can legitimately have no trailing data padding.
  if (!file.tensors.empty() && file.tensor_data_offset > file_size_)
    throw std::runtime_error("GGUF tensor data outside file");
  for (auto& tensor : file.tensors) {
    if (tensor.offset % alignment != 0) throw std::runtime_error("unaligned GGUF tensor offset");
    if (tensor.offset > std::numeric_limits<uint64_t>::max() - file.tensor_data_offset)
      throw std::runtime_error("tensor offset overflow");
    tensor.absolute_offset = file.tensor_data_offset + tensor.offset;
    tensor.byte_length = ggml_tensor_nbytes(tensor.ggml_type, tensor.shape);
    if (tensor.absolute_offset > file_size_ || tensor.byte_length > file_size_ - tensor.absolute_offset)
      throw std::runtime_error("tensor range outside GGUF");
  }
  std::vector<const TensorInfo*> ranges;
  ranges.reserve(file.tensors.size());
  for (const auto& tensor : file.tensors) ranges.push_back(&tensor);
  std::sort(ranges.begin(), ranges.end(), [](const auto* a, const auto* b) {
    return a->absolute_offset < b->absolute_offset;
  });
  for (size_t i = 1; i < ranges.size(); ++i)
    if (ranges[i]->absolute_offset - ranges[i - 1]->absolute_offset < ranges[i - 1]->byte_length)
      throw std::runtime_error("overlapping GGUF tensor ranges");
  return file;
}
std::string ggml_type_name(uint32_t t){switch(t){case 0:return"F32";case 1:return"F16";case 2:return"Q4_0";case 3:return"Q4_1";case 4:case 5:return"DEPRECATED";case 6:return"Q5_0";case 7:return"Q5_1";case 8:return"Q8_0";case 9:return"Q8_1";case 10:return"Q2_K";case 11:return"Q3_K";case 12:return"Q4_K";case 13:return"Q5_K";case 14:return"Q6_K";case 15:return"Q8_K";case 16:return"IQ2_XXS";case 17:return"IQ2_XS";case 18:return"IQ3_XXS";case 19:return"IQ1_S";case 20:return"IQ4_NL";case 21:return"IQ3_S";case 22:return"IQ2_S";case 23:return"IQ4_XS";case 24:return"I8";case 25:return"I16";case 26:return"I32";case 27:return"I64";case 28:return"F64";case 29:return"IQ1_M";case 30:return"BF16";case 34:return"TQ1_0";case 35:return"TQ2_0";case 39:return"MXFP4";case 40:return"NVFP4";case 41:return"Q1_0";case 42:return"Q2_0";default:return"TYPE_"+std::to_string(t);}}
uint64_t ggml_tensor_nbytes(uint32_t type,const std::vector<uint64_t>& shape){
 if(shape.empty()||shape.size()>4)throw std::runtime_error("invalid GGUF tensor dimensions");
 for(auto dimension:shape)if(dimension==0)throw std::runtime_error("GGUF tensor dimension is zero");
 uint64_t block=1,bytes=0;
 switch(type){case 0:block=1;bytes=4;break;case 1:case 30:block=1;bytes=2;break;case 2:block=32;bytes=18;break;case 3:block=32;bytes=20;break;case 6:block=32;bytes=22;break;case 7:block=32;bytes=24;break;case 8:block=32;bytes=34;break;case 9:block=32;bytes=36;break;case 10:block=256;bytes=84;break;case 11:block=256;bytes=110;break;case 12:block=256;bytes=144;break;case 13:block=256;bytes=176;break;case 14:block=256;bytes=210;break;case 15:block=256;bytes=292;break;case 16:block=256;bytes=66;break;case 17:block=256;bytes=74;break;case 18:block=256;bytes=98;break;case 19:block=256;bytes=50;break;case 20:block=32;bytes=18;break;case 21:block=256;bytes=110;break;case 22:block=256;bytes=82;break;case 23:block=256;bytes=136;break;case 24:block=1;bytes=1;break;case 25:block=1;bytes=2;break;case 26:block=1;bytes=4;break;case 27:case 28:block=1;bytes=8;break;case 29:block=256;bytes=56;break;case 34:block=256;bytes=54;break;case 35:block=256;bytes=66;break;case 39:block=32;bytes=17;break;case 40:block=64;bytes=36;break;case 41:block=128;bytes=18;break;case 42:block=64;bytes=18;break;default:throw std::runtime_error("GGML tensor type size is unsupported: "+std::to_string(type));}
 if(shape[0]%block!=0)throw std::runtime_error("tensor row dimension is not divisible by GGML quantization block");
 uint64_t rows=1;for(size_t i=1;i<shape.size();i++){if(shape[i]&&rows>std::numeric_limits<uint64_t>::max()/shape[i])throw std::runtime_error("tensor element count overflow");rows*=shape[i];}
 const uint64_t row_blocks=shape[0]/block;
 if(row_blocks>std::numeric_limits<uint64_t>::max()/bytes)throw std::runtime_error("tensor row byte size overflow");
 uint64_t row_bytes=row_blocks*bytes;if(row_bytes>std::numeric_limits<uint64_t>::max()/rows)throw std::runtime_error("tensor byte size overflow");return row_bytes*rows;
}
} // namespace pmi
