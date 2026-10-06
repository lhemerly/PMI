#include "pmi/gguf.hpp"
#include "pmi/manifest.hpp"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Metadata = std::vector<std::pair<std::string, pmi::MetadataValue>>;
struct Tensor {
  std::string name;
  std::vector<uint64_t> shape;
  uint32_t type{};
  uint64_t offset{};
};

void require(bool condition, const std::string& description) {
  if (!condition) throw std::runtime_error(description);
}

size_t rejections = 0;
void rejects(const std::function<void()>& action, const std::string& message) {
  try {
    action();
  } catch (const std::runtime_error& error) {
    require(std::string(error.what()).find(message) != std::string::npos,
            "unexpected rejection: " + std::string(error.what()) + "; expected " + message);
    ++rejections;
    return;
  }
  throw std::runtime_error("expected rejection: " + message);
}

void integer(std::vector<uint8_t>& bytes, uint64_t value, size_t width) {
  for (size_t i = 0; i < width; ++i) bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
}

void string(std::vector<uint8_t>& bytes, const std::string& value) {
  integer(bytes, value.size(), 8);
  bytes.insert(bytes.end(), value.begin(), value.end());
}

void fixture(const std::filesystem::path& path, const Metadata& metadata,
             const std::vector<Tensor>& tensors, size_t payload_size,
             uint8_t payload_start = 0, size_t alignment = 32) {
  std::vector<uint8_t> bytes;
  integer(bytes, 0x46554747, 4);
  integer(bytes, 3, 4);
  integer(bytes, tensors.size(), 8);
  integer(bytes, metadata.size(), 8);
  for (const auto& entry : metadata) {
    string(bytes, entry.first);
    integer(bytes, static_cast<uint32_t>(entry.second.type), 4);
    if (entry.second.type == pmi::GgufType::STRING) string(bytes, entry.second.text);
    else if (entry.second.type == pmi::GgufType::FLOAT32) {
      const auto value = std::stof(entry.second.text);
      uint32_t bits;
      static_assert(sizeof(bits) == sizeof(value), "FLOAT32 fixture requires 32-bit float");
      std::memcpy(&bits, &value, sizeof(bits));
      integer(bytes, bits, sizeof(bits));
    } else if (entry.second.type == pmi::GgufType::FLOAT64) {
      const auto value = std::stod(entry.second.text);
      uint64_t bits;
      static_assert(sizeof(bits) == sizeof(value), "FLOAT64 fixture requires 64-bit double");
      std::memcpy(&bits, &value, sizeof(bits));
      integer(bytes, bits, sizeof(bits));
    }
    else {
      size_t width = 0;
      switch (entry.second.type) {
        case pmi::GgufType::UINT16: case pmi::GgufType::INT16: width = 2; break;
        case pmi::GgufType::UINT32: case pmi::GgufType::INT32: width = 4; break;
        case pmi::GgufType::UINT64: case pmi::GgufType::INT64: width = 8; break;
        default: throw std::runtime_error("test fixture uses unsupported metadata type");
      }
      integer(bytes, static_cast<uint64_t>(std::stoll(entry.second.text)), width);
    }
  }
  for (const auto& tensor : tensors) {
    string(bytes, tensor.name);
    integer(bytes, tensor.shape.size(), 4);
    for (const auto dimension : tensor.shape) integer(bytes, dimension, 8);
    integer(bytes, tensor.type, 4);
    integer(bytes, tensor.offset, 8);
  }
  while (bytes.size() % alignment) bytes.push_back(0);
  for (size_t i = 0; i < payload_size; ++i) bytes.push_back(static_cast<uint8_t>(payload_start + i));
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  require(static_cast<bool>(output), "could not write test fixture");
}

Metadata model_metadata() {
  return {{"general.architecture", {pmi::GgufType::STRING, "toy_moe"}},
          {"general.alignment", {pmi::GgufType::UINT32, "32"}}};
}

Metadata split_metadata(uint32_t id, uint32_t count, int32_t tensors, bool primary = false) {
  Metadata metadata = primary ? model_metadata() : Metadata{};
  metadata.push_back({"split.no", {pmi::GgufType::UINT16, std::to_string(id)}});
  metadata.push_back({"split.count", {pmi::GgufType::UINT16, std::to_string(count)}});
  metadata.push_back({"split.tensors.count", {pmi::GgufType::INT32, std::to_string(tensors)}});
  return metadata;
}

struct Fixtures {
  std::filesystem::path directory;
  Fixtures() {
    directory = std::filesystem::temp_directory_path() /
        ("pmi-shard-manifest-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    require(std::filesystem::create_directory(directory), "could not create fixture directory");
  }
  ~Fixtures() {
    std::error_code error;
    std::filesystem::remove_all(directory, error);
  }
};

void test_manifest(const Fixtures& fixtures) {
  const auto single_path = fixtures.directory / "legacy\".gguf";
  fixture(single_path, model_metadata(),
      {{"blk.0.ffn_up_exps.weight", {32, 1, 4}, 8, 0},
       {"blk.0.attn_norm.weight", {2, 2}, 0, 160}}, 176);
  const auto single_file = pmi::GgufReader(single_path).inspect();
  const auto single = pmi::build_manifest(single_file);
  require(single.shards.size() == 1 && single.shards[0].tensor_count == 2, "legacy shard descriptor");
  require(single.shards[0].source_path == single_path, "source path retained");
  require(single.tensors.size() == 5 && single.tensors[3].byte_length == 34 &&
          single.tensors[3].expert_id == 3 && single.tensors[3].shard_id == 0 &&
          single.tensors[3].absolute_offset == single_file.tensor_data_offset + 102 &&
          single.tensors[3].shape == std::vector<uint64_t>({32, 1}) &&
          single.tensors[3].source_tensor_name == "blk.0.ffn_up_exps.weight", "legacy expert slices");
  const auto json = pmi::manifest_json(single);
  require(json.find("legacy\\\".gguf") != std::string::npos &&
          json.find("\"shape\":[32,1]") != std::string::npos &&
          json.find("\"shard_id\":0") != std::string::npos &&
          json.find("\"tensor_count\":2") != std::string::npos, "manifest JSON shape/source/shard fields");

  const auto first_path = fixtures.directory / "model-00001-of-00002.gguf";
  const auto second_path = fixtures.directory / "model-00002-of-00002.gguf";
  auto primary_metadata = split_metadata(0, 2, 2, true);
  primary_metadata.push_back({"toy_moe.expert_count", {pmi::GgufType::UINT32, "2"}});
  fixture(first_path, primary_metadata, {{"token_embd.weight", {2, 2}, 0, 0}}, 16, 0x11);
  fixture(second_path, split_metadata(1, 2, 2), {{"blk.0.ffn_up_exps.weight", {32, 1, 2}, 8, 0}}, 68, 0x22);
  const auto first = pmi::GgufReader(first_path).inspect();
  const auto second = pmi::GgufReader(second_path).inspect();
  require(second.tensors[0].shard_id == 1, "reader preserves split.no");
  const auto merged = pmi::build_manifest(std::vector<pmi::GgufFile>{second, first});
  require(merged.shards.size() == 2 && merged.shards[0].source_path == first_path &&
          merged.shards[1].source_path == second_path, "out-of-order shards sorted by ID");
  require(merged.file_size == first.file_size + second.file_size && merged.tensors.size() == 3,
          "aggregate file size and logical tensor count");
  require(merged.tensors[0].shard_id == 0 && merged.tensors[1].shard_id == 1 &&
          merged.tensors[2].shard_id == 1 && merged.tensors[2].offset == 34 &&
          merged.tensors[2].absolute_offset == second.tensor_data_offset + 34,
          "expert slices keep shard-local offsets");
  std::ifstream payload(merged.shards[1].source_path, std::ios::binary);
  payload.seekg(static_cast<std::streamoff>(merged.tensors[2].absolute_offset));
  require(payload.get() == 0x22 + 34, "expert slice addresses actual shard bytes");

  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{}); }, "without GGUF files");
  rejects([&] { pmi::build_manifest(first); }, "split count");
  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{first, first}); }, "duplicate GGUF split.no");
  auto bad = second;
  bad.metadata["split.count"].text = "3";
  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{first, bad}); }, "split count");
  bad = second;
  bad.metadata["split.no"].text = "2";
  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{first, bad}); }, "split.no outside");
  bad = second;
  bad.metadata["split.tensors.count"].text = "3";
  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{first, bad}); }, "disagree on split.tensors.count");
  auto bad_first = first;
  bad_first.metadata["split.tensors.count"].text = "3";
  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{bad_first, bad}); }, "physical tensor count");
  bad = second;
  bad.metadata["split.tensors.count"].text = "-1";
  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{first, bad}); }, "nonnegative");
  bad = second;
  bad.metadata.erase("split.tensors.count");
  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{first, bad}); }, "incomplete GGUF split metadata");
  bad = second;
  bad.metadata.erase("split.count");
  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{first, bad}); }, "missing split.count");
  bad = second;
  bad.metadata["split.count"].type = pmi::GgufType::STRING;
  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{first, bad}); }, "must be an integer");
  bad = second;
  bad.metadata["split.count"].type = pmi::GgufType::UINT32;
  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{first, bad}); }, "metadata type or value");
  bad = second;
  bad.metadata["general.architecture"] = {pmi::GgufType::STRING, "different_model"};
  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{first, bad}); }, "disagree on general.architecture");
  bad_first = first;
  bad_first.metadata.erase("general.architecture");
  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{bad_first, second}); }, "general.architecture in GGUF shard 0");
  bad = second;
  bad.metadata["toy_moe.expert_count"] = {pmi::GgufType::UINT32, "3"};
  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{first, bad}); }, "disagree on toy_moe.expert_count");
  bad_first = first;
  bad_first.metadata["toy_moe.expert_count"].text = "3";
  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{bad_first, second}); }, "dimension disagrees");
  bad = second;
  bad.version = 2;
  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{first, bad}); }, "mismatched versions");
  bad = second;
  bad.tensors[0].name = first.tensors[0].name;
  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{first, bad}); }, "duplicate GGUF tensor name");
  bad = second;
  bad.source_path = first.source_path;
  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{first, bad}); }, "duplicate GGUF shard source path");
  bad = second;
  bad.tensors[0].byte_length -= 1;
  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{first, bad}); }, "byte length disagrees");
  bad = second;
  bad.tensors[0].absolute_offset += 1;
  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{first, bad}); }, "inconsistent offset");
  bad_first = first;
  bad_first.file_size = std::numeric_limits<uint64_t>::max();
  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{bad_first, second}); }, "aggregate GGUF file size overflow");

  // Upstream's merge command leaves split.count=0 and may retain the other keys.
  for (const auto count : {0, 1}) {
    auto unsplit = single_file;
    unsplit.metadata["split.count"] = {pmi::GgufType::UINT16, std::to_string(count)};
    unsplit.metadata["split.no"] = {pmi::GgufType::UINT16, "0"};
    unsplit.metadata["split.tensors.count"] = {pmi::GgufType::INT32, "2"};
    require(pmi::build_manifest(unsplit).tensors.size() == 5, "merged single-file split metadata compatibility");
  }
  // A metadata-only first split is an upstream supported option.
  fixture(first_path, split_metadata(0, 2, 1, true), {}, 0);
  fixture(second_path, split_metadata(1, 2, 1), {{"output.weight", {2, 2}, 0, 0}}, 16);
  const auto empty_first = pmi::build_manifest(std::vector<pmi::GgufFile>{
      pmi::GgufReader(first_path).inspect(), pmi::GgufReader(second_path).inspect()});
  require(empty_first.shards[0].tensor_count == 0 && empty_first.tensors[0].shard_id == 1,
          "metadata-only first shard");
}

void test_reader_and_sizes(const Fixtures& fixtures) {
  const auto path = fixtures.directory / "malformed.gguf";
  const auto max = std::numeric_limits<uint64_t>::max();
  const auto inspect = [&] { pmi::GgufReader(path).inspect(); };
  fixture(path, model_metadata(), {{"weight", {2}, 0, 1}}, 32);
  rejects(inspect, "unaligned GGUF tensor offset");
  fixture(path, model_metadata(), {{"a", {2}, 0, 0}, {"b", {2}, 0, 0}}, 32);
  rejects(inspect, "overlapping GGUF tensor ranges");
  fixture(path, model_metadata(), {{"a", {2}, 0, 0}, {"a", {2}, 0, 32}}, 64);
  rejects(inspect, "duplicate GGUF tensor name");
  fixture(path, model_metadata(), {{"weight", {32}, 0, 0}}, 8);
  rejects(inspect, "tensor range outside GGUF");
  fixture(path, model_metadata(), {{"weight", {}, 0, 0}}, 8);
  rejects(inspect, "invalid GGUF dimensions");
  fixture(path, model_metadata(), {{"weight", {0}, 0, 0}}, 8);
  rejects(inspect, "dimension is zero");
  fixture(path, model_metadata(), {{"weight", {33}, 8, 0}}, 64);
  rejects(inspect, "not divisible");
  fixture(path, model_metadata(), {{"weight", {max / 4 + 1}, 0, 0}}, 8);
  rejects(inspect, "row byte size overflow");
  fixture(path, model_metadata(), {{"weight", {1, max}, 0, 0}}, 8);
  rejects(inspect, "tensor byte size overflow");
  fixture(path, model_metadata(), {{"weight", {1, max, 2}, 24, 0}}, 8);
  rejects(inspect, "element count overflow");
  fixture(path, model_metadata(), {{"weight", {2}, 999, 0}}, 8);
  rejects(inspect, "unsupported");
  fixture(path, model_metadata(), {{"weight", {2}, 0, max - 31}}, 8);
  rejects(inspect, "tensor offset overflow");
  auto duplicate = model_metadata();
  duplicate.push_back(duplicate.front());
  fixture(path, duplicate, {{"weight", {2}, 0, 0}}, 8);
  rejects(inspect, "duplicate GGUF metadata key");
  auto bad_alignment = model_metadata();
  bad_alignment[1].second.text = "3";
  fixture(path, bad_alignment, {{"weight", {2}, 0, 0}}, 8);
  rejects(inspect, "power of two");
  auto custom_alignment = model_metadata();
  custom_alignment[1].second.text = "64";
  fixture(path, custom_alignment, {{"weight", {2}, 0, 0}}, 8, 0, 64);
  const auto custom = pmi::build_manifest(pmi::GgufReader(path).inspect());
  require(custom.tensors[0].required_alignment == 64 && custom.tensor_data_offset % 64 == 0,
          "actual file alignment preserved");

  require(pmi::ggml_tensor_nbytes(8, {32, 3}) == 102, "Q8_0 rows use serialized block bytes");
  require(pmi::ggml_tensor_nbytes(12, {256, 2, 3}) == 864, "Q4_K multi-dimensional byte count");
  rejects([&] { pmi::ggml_tensor_nbytes(0, {1, 1, 1, 1, 1}); }, "invalid GGUF tensor dimensions");
  rejects([&] { pmi::ggml_tensor_nbytes(0, {1, 0}); }, "dimension is zero");
  rejects([&] { pmi::ggml_tensor_nbytes(8, {32, max}); }, "byte size overflow");
}

class CommaDecimal : public std::numpunct<char> {
 protected:
  char do_decimal_point() const override { return ','; }
};

void test_metadata_precision(const Fixtures& fixtures) {
  const auto path = fixtures.directory / "floating-metadata.gguf";
  const Metadata values{
      {"tiny_float", {pmi::GgufType::FLOAT32, "1e-7"}},
      {"precise_float", {pmi::GgufType::FLOAT32, "1.00000011920928955078125"}},
      {"tiny_double", {pmi::GgufType::FLOAT64, "1e-200"}},
      {"precise_double", {pmi::GgufType::FLOAT64, "1.0000000000000002"}},
      {"largest_double", {pmi::GgufType::FLOAT64, "1.7976931348623157e308"}}};
  auto metadata = model_metadata();
  metadata.insert(metadata.end(), values.begin(), values.end());
  fixture(path, metadata, {{"weight", {2}, 0, 0}}, 8);

  pmi::GgufFile file;
  {
    struct RestoreLocale {
      std::locale saved = std::locale();
      ~RestoreLocale() { std::locale::global(saved); }
    } restore;
    std::locale::global(std::locale(restore.saved, new CommaDecimal));
    file = pmi::GgufReader(path).inspect();
  }
  for (const auto& entry : values) {
    const auto& actual = file.metadata.at(entry.first);
    require(actual.type == entry.second.type && actual.text.find(',') == std::string::npos,
            "floating metadata is independent of the process locale");
    if (actual.type == pmi::GgufType::FLOAT32)
      require(std::stof(actual.text) == std::stof(entry.second.text), "FLOAT32 metadata exact round trip");
    else require(std::stod(actual.text) == std::stod(entry.second.text), "FLOAT64 metadata exact round trip");
  }

  const auto first_path = fixtures.directory / "routing-00001-of-00002.gguf";
  const auto second_path = fixtures.directory / "routing-00002-of-00002.gguf";
  const Metadata routing{
      {"toy_moe.expert_used_count", {pmi::GgufType::UINT32, "1"}},
      {"toy_moe.expert_weights_scale", {pmi::GgufType::FLOAT32, "1e-7"}},
      {"toy_moe.expert_group_count", {pmi::GgufType::UINT32, "1"}}};
  auto first_metadata = split_metadata(0, 2, 2, true);
  first_metadata.insert(first_metadata.end(), routing.begin(), routing.end());
  fixture(first_path, first_metadata, {{"a.weight", {2}, 0, 0}}, 8);
  const auto first = pmi::GgufReader(first_path).inspect();
  const auto load_second = [&](const Metadata& copied_routing) {
    auto second_metadata = split_metadata(1, 2, 2);
    second_metadata.insert(second_metadata.end(), copied_routing.begin(), copied_routing.end());
    fixture(second_path, second_metadata, {{"b.weight", {2}, 0, 0}}, 8);
    return pmi::GgufReader(second_path).inspect();
  };
  require(pmi::build_manifest(std::vector<pmi::GgufFile>{load_second(routing), first}).tensors.size() == 2,
          "identical repeated routing metadata accepted");
  for (size_t i = 0; i < routing.size(); ++i) {
    auto conflicting = routing;
    // Adjacent nonzero FLOAT32 scales previously both rounded to "0.000000".
    conflicting[i].second.text = conflicting[i].second.type == pmi::GgufType::FLOAT32 ? "1.0000001e-7" : "2";
    const auto second = load_second(conflicting);
    rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{first, second}); },
            "disagree on " + conflicting[i].first);
  }
  auto wrong_type = routing;
  wrong_type[1].second.type = pmi::GgufType::FLOAT64;
  const auto second = load_second(wrong_type);
  rejects([&] { pmi::build_manifest(std::vector<pmi::GgufFile>{first, second}); },
          "disagree on toy_moe.expert_weights_scale");
}
} // namespace

int main() {
  try {
    Fixtures fixtures;
    test_manifest(fixtures);
    test_reader_and_sizes(fixtures);
    test_metadata_precision(fixtures);
    std::cout << "shard manifest tests passed (legacy, complete/reordered/missing/conflicting shards, "
                 "real slice bytes, JSON, alignment, exact floating metadata, " << rejections << " rejection cases)\n";
  } catch (const std::exception& error) {
    std::cerr << "shard manifest test failed: " << error.what() << '\n';
    return 1;
  }
}
