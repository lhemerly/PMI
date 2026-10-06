#include "pmi/gguf.hpp"
#include "pmi/inference.hpp"
#include "pmi/manifest.hpp"
#include "pmi/page_provider.hpp"
#include "pmi/connector.hpp"
#include "pmi/telemetry.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
void usage() {
  std::cerr << "usage:\n"
            << "  pmi inspect SHARD.gguf [SHARD.gguf ...]\n"
            << "  pmi reference FIRST_SHARD.gguf --prompt TEXT [--max-tokens N] [--gpu-layers N]\n"
            << "      [--reference-mode default|demand] (demand requires CPU and disables repacking)\n"
            << "  pmi run SHARD.gguf [SHARD.gguf ...] --prompt TEXT [--cache-bytes N]\n"
            << "      [--max-tokens N] [--prefill-chunk N] [--context N] [--trace-json PATH]\n"
            << "      [--device-upload-host-io buffered|direct-preferred|direct-required] [--device-upload-host-policy retained|transient] [--progress-jsonl NEW_PATH] [--cache-policy lru|previous-top4] [--profile-paging on|off]\n"
            << "      [--dense-load default|direct-required] (dense tensor payloads only)\n"
      << "      [--dense-pages native|base] [--load-trace-jsonl PATH] (run, direct-required only)\n"
            << "      [--projection-backend cpu|device-cpu|device-vulkan] [--projection-device-index N]\n"
            << "      [--projection-activation backend-native|cpu-compatible] (compatible requires device-vulkan)\n"
            << "      [--projection-dispatch all|split] (split: frozen first-evaluation residency, CPU misses)\n"
            << "      [--device-cache-bytes N --device-graph-bytes N] (required for device backends)\n"
            << "  generation options: [--trace-mode full|compact] [--cpu-repack on|off]\n"
            << "      [--teacher-forcing-file IDS.txt] (exactly max-tokens IDs; benchmark replay)\n"
            << "      [--routing-jsonl NEW_PATH] (bounded routing observations; no token text/IDs)\n"
            << "  pmi moe-layer SHARD.gguf [SHARD.gguf ...] --input FLOATS.txt\n"
            << "      [--layer N] [--experts-used N] [--cache-bytes N] [--repeat N]\n"
            << "  pmi self-test TOY_FIXTURE.gguf\n";
}
uint64_t number(const std::string& value) {
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("expected a nonnegative integer: " + value);
  return std::stoull(value);
}
template<class T> void array(const std::vector<T>& values) {
  std::cout << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << values[i];
  }
  std::cout << ']';
}
void nullable(const std::optional<uint64_t>& value) {
  if (value) std::cout << *value; else std::cout << "null";
}
void memory_json(const pmi::ProcessMemory& memory) {
  std::cout << "\"rss_bytes\":"; nullable(memory.rss_bytes);
  std::cout << ",\"peak_rss_bytes\":"; nullable(memory.peak_rss_bytes);
  std::cout << ",\"process_storage_read_bytes\":"; nullable(memory.storage_read_bytes);
}
int moe_layer(int argc, char** argv) {
  std::vector<pmi::GgufFile> files;
  std::string input_path;
  uint64_t layer = 0, used = 0, capacity = 64 * 1024 * 1024, repeat = 1;
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--", 0) != 0) files.push_back(pmi::GgufReader(arg).inspect());
    else {
      if (++i == argc) throw std::invalid_argument("missing option value: " + arg);
      if (arg == "--input") input_path = argv[i];
      else if (arg == "--layer") layer = number(argv[i]);
      else if (arg == "--experts-used") used = number(argv[i]);
      else if (arg == "--cache-bytes") capacity = number(argv[i]);
      else if (arg == "--repeat") repeat = number(argv[i]);
      else throw std::invalid_argument("unknown moe-layer option: " + arg);
    }
  }
  if (files.empty() || input_path.empty() || !capacity || capacity > SIZE_MAX ||
      !repeat || repeat > 10000 || layer > INT32_MAX)
    throw std::invalid_argument("moe-layer requires shards, input, positive capacity and repeat <= 10000");
  auto manifest = pmi::build_manifest(files);
  auto find = [&](const std::string& suffix, int expert = -1) -> const pmi::ManifestTensor& {
    const auto name = "blk." + std::to_string(layer) + "." + suffix;
    for (const auto& tensor : manifest.tensors)
      if (tensor.source_tensor_name == name && tensor.expert_id == expert) return tensor;
    throw std::runtime_error("missing supported layer tensor: " + name);
  };
  const auto& router = find("ffn_gate_inp.weight");
  const auto& expert = find("ffn_up_exps.weight", 0);
  const auto& shared = find("ffn_up_shexp.weight");
  if (router.shape.size() != 2 || expert.shape.size() != 2 || shared.shape.size() != 2)
    throw std::runtime_error("unsupported layer dimensions");
  pmi::Qwen4ExpMoeConfig config;
  config.layer = static_cast<int>(layer);
  config.embedding_length = router.shape[0]; config.expert_count = router.shape[1];
  config.expert_hidden_length = expert.shape[1]; config.shared_hidden_length = shared.shape[1];
  // Canonical metadata comes from split.no=0 regardless of argument order.
  for (const auto& file : files) {
    if (file.source_path != manifest.shards.front().source_path) continue;
    if (!used) {
      auto value = file.metadata.find("qwen4exp.expert_used_count");
      if (value != file.metadata.end()) used = number(value->second.text);
    }
    auto scale = file.metadata.find("qwen4exp.expert_weights_scale");
    if (scale != file.metadata.end()) config.expert_weights_scale = std::stof(scale->second.text);
    auto groups = file.metadata.find("qwen4exp.expert_group_count");
    if (groups != file.metadata.end()) config.expert_groups = number(groups->second.text);
  }
  if (!used) throw std::invalid_argument("--experts-used is required when GGUF metadata omits it");
  config.experts_used = used;
  pmi::Qwen4ExpMoeConnector connector(manifest, config);
  std::ifstream input_file(input_path);
  if (!input_file) throw std::runtime_error("cannot open input vector");
  std::vector<float> input;
  float value;
  for (;;) {
    input_file >> std::ws;
    if (input_file.eof()) break;
    if (!(input_file >> value)) throw std::invalid_argument("malformed float in input vector");
    if (!std::isfinite(value) || input.size() >= config.embedding_length)
      throw std::invalid_argument("invalid or oversized input vector");
    input.push_back(value);
  }
  if (!input_file.eof() || input.size() != config.embedding_length)
    throw std::invalid_argument("input must be one whitespace-separated token embedding");
  std::vector<std::filesystem::path> paths;
  for (const auto& shard : manifest.shards) paths.push_back(shard.source_path);
  pmi::FilePageProvider pages(paths, static_cast<size_t>(capacity));
  std::cout << std::setprecision(9);
  for (uint64_t iteration = 0; iteration < repeat; ++iteration) {
    const auto before = pages.stats();
    const auto start = std::chrono::steady_clock::now();
    const auto trace = connector.forward(pages, input);
    const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const auto after = pages.stats();
    const auto hits = after.hits - before.hits, misses = after.misses - before.misses;
    std::cout << "{\"path\":\"pmi_qwen_moe_layer\",\"backend\":\"cpu\",\"iteration\":" << iteration
              << ",\"elapsed_ms\":" << elapsed << ",\"layer_evaluations_per_second\":" << 1000.0 / elapsed
              << ",\"io_bytes\":" << after.io_bytes - before.io_bytes
              << ",\"cache_hits\":" << hits << ",\"cache_misses\":" << misses
              << ",\"hit_rate\":" << (hits + misses ? double(hits) / (hits + misses) : 0)
              << ",\"evictions\":" << after.evictions - before.evictions
              << ",\"cache_reserved_bytes\":" << after.current_bytes
              << ",\"cache_peak_reserved_bytes\":" << after.peak_bytes << ",\"vram_bytes\":0,";
    memory_json(pmi::process_memory());
    std::cout << ",\"router_logits\":"; array(trace.router_logits);
    std::cout << ",\"router_probabilities\":"; array(trace.router_probabilities);
    std::cout << ",\"expert_ids\":"; array(trace.expert_ids);
    std::cout << ",\"expert_weights\":"; array(trace.expert_weights);
    std::cout << ",\"output\":"; array(trace.output);
    std::cout << "}\n";
  }
  return 0;
}
int self_test(const char* fixture) {
  if (!fixture) throw std::invalid_argument("self-test requires a generated toy fixture");

  if (fixture) {
    pmi::GgufReader reader(fixture);
    auto file = reader.inspect();
    auto manifest = pmi::build_manifest(file);
    if (manifest.tensors.size() != 5 || manifest.tensors[0].byte_length != 34 ||
        manifest.tensors[3].byte_length != 34 || manifest.tensors[3].expert_id != 3 ||
        manifest.tensors[3].absolute_offset != 256 + 34 * 3 || manifest.tensors[4].byte_length != 16 ||
        manifest.tensors[0].layer != 0 ||
        manifest.tensors[0].semantic_class != pmi::TensorClass::ROUTED_EXPERT)
      throw std::runtime_error("GGUF expert slice manifest check failed");
    pmi::FilePageProvider pages(fixture, 128);
    auto payload = pages.request(manifest.tensors[2]).get();
    if (payload->size() != 34 || (*payload)[0] != 68 || (*payload)[33] != 101)
      throw std::runtime_error("explicit expert page read mismatch");
    auto reused = pages.request(manifest.tensors[2]).get();
    if (*reused != *payload || pages.stats().hits == 0)
      throw std::runtime_error("page cache reuse check failed");
  }
  std::cout << "self-test ok\n";
  return 0;
}
int generation(int argc, char** argv, bool paged) {
  std::vector<std::filesystem::path> paths;
  std::string prompt;
  bool prompt_seen = false;
  std::filesystem::path forcing_file;
  pmi::GenerationOptions options;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind("--", 0) != 0) { paths.emplace_back(arg); continue; }
    if (++i == argc) throw std::invalid_argument("missing generation option value: " + arg);
    const std::string value = argv[i];
    if (arg == "--prompt") { prompt = value; prompt_seen = true; }
    else if (arg == "--trace-json") options.trace_json = value;
    else if (arg == "--device-upload-host-io") {
      if (!paged) throw std::invalid_argument("device-upload-host-io requires run");
      if (value == "buffered") options.upload_host_io = pmi::HostReadPolicy::Buffered;
      else if (value == "direct-preferred") options.upload_host_io = pmi::HostReadPolicy::DirectPreferred;
      else if (value == "direct-required") options.upload_host_io = pmi::HostReadPolicy::DirectRequired;
      else throw std::invalid_argument("device-upload-host-io requires buffered|direct-preferred|direct-required");
    }
    else if (arg == "--device-upload-host-policy") {
      if (!paged || (value != "retained" && value != "transient"))
        throw std::invalid_argument("device-upload-host-policy requires run and retained|transient");
      options.transient_upload_host = value == "transient";
    }
    else if (arg == "--progress-jsonl") {
      if (!paged) throw std::invalid_argument("progress-jsonl is only valid for run");
      options.progress_jsonl = value;
    }
    else if (arg == "--routing-jsonl") options.routing_jsonl = value;
    else if (arg == "--teacher-forcing-file") forcing_file = value;
    else if (arg == "--dense-pages") {
      if (!paged) throw std::invalid_argument("dense-pages is only valid for run");
      if (value == "base") options.dense_base_pages = true;
      else if (value == "native") options.dense_base_pages = false;
      else throw std::invalid_argument("dense-pages must be native or base");
    } else if (arg == "--load-trace-jsonl") {
      if (!paged) throw std::invalid_argument("load-trace-jsonl is only valid for run");
      options.load_trace_jsonl = value;
    }
    else if (arg == "--dense-load") {
      if (!paged) throw std::invalid_argument("dense-load is only valid for the run command");
      if (value == "default") options.dense_load = pmi::DenseLoadPolicy::Default;
      else if (value == "direct-required") options.dense_load = pmi::DenseLoadPolicy::DirectRequired;
      else throw std::invalid_argument("dense-load must be default or direct-required");
    }
    else if (arg == "--projection-backend") {
      if (!paged) throw std::invalid_argument("projection-backend is only valid for the run command");
      if (value == "cpu") options.projection_backend = pmi::ProjectionBackend::Cpu;
      else if (value == "device-cpu") options.projection_backend = pmi::ProjectionBackend::DeviceCpu;
      else if (value == "device-vulkan") options.projection_backend = pmi::ProjectionBackend::DeviceVulkan;
      else throw std::invalid_argument("projection-backend must be cpu, device-cpu or device-vulkan");
    } else if (arg == "--projection-dispatch") {
      if (!paged) throw std::invalid_argument("projection-dispatch is only valid for the run command");
      if (value == "all") options.split_cpu_misses = false;
      else if (value == "split") options.split_cpu_misses = true;
      else throw std::invalid_argument("projection-dispatch must be all or split");
    } else if (arg == "--projection-activation") {
      if (!paged) throw std::invalid_argument("projection-activation is only valid for the run command");
      if (value == "backend-native") options.projection_activation = pmi::ProjectionActivationMode::BackendNative;
      else if (value == "cpu-compatible") options.projection_activation = pmi::ProjectionActivationMode::CpuCompatible;
      else throw std::invalid_argument("projection-activation must be backend-native or cpu-compatible");
    } else if (arg == "--projection-device-index") {
      const auto index = number(value);
      if (!paged || index > UINT32_MAX)
        throw std::invalid_argument("projection-device-index requires paged run and a uint32 index");
      options.projection_device_index = static_cast<uint32_t>(index);
    } else if (arg == "--device-cache-bytes" || arg == "--device-graph-bytes") {
      const auto capacity = number(value);
      if (!paged || !capacity || capacity > SIZE_MAX)
        throw std::invalid_argument("device budgets require paged run and positive capacities");
      if (arg == "--device-cache-bytes") options.device_cache_bytes = capacity;
      else options.device_graph_bytes = capacity;
    }
    else if (arg == "--cache-policy") {
      if (!paged) throw std::invalid_argument("cache-policy is only valid for the run command");
      if (value == "lru") options.cache_policy = pmi::CachePolicy::Lru;
      else if (value == "previous-top4") options.cache_policy = pmi::CachePolicy::PreviousTop4;
      else throw std::invalid_argument("cache-policy must be lru or previous-top4");
    } else if (arg == "--profile-paging") {
      if (!paged) throw std::invalid_argument("profile-paging is only valid for the run command");
      if (value == "on") options.profile_paging = true;
      else if (value == "off") options.profile_paging = false;
      else throw std::invalid_argument("profile-paging must be on or off");
    }
    else if (arg == "--trace-mode") {
      if (value == "full") options.trace_mode = pmi::TraceMode::Full;
      else if (value == "compact") options.trace_mode = pmi::TraceMode::Compact;
      else throw std::invalid_argument("trace-mode must be full or compact");
    } else if (arg == "--reference-mode") {
      if (paged) throw std::invalid_argument("reference-mode is only valid for the reference command");
      if (value == "default") options.reference_mode = pmi::ReferenceMode::Default;
      else if (value == "demand") options.reference_mode = pmi::ReferenceMode::Demand;
      else throw std::invalid_argument("reference-mode must be default or demand");
    } else if (arg == "--cpu-repack") {
      if (value == "on") options.cpu_repack = true;
      else if (value == "off") options.cpu_repack = false;
      else throw std::invalid_argument("cpu-repack must be on or off");
    }
    else if (arg == "--gpu-layers" && value == "-1") options.gpu_layers = -1;
    else {
      const auto count = number(value);
      if (arg == "--cache-bytes") {
        if (!paged || !count || count > SIZE_MAX) throw std::invalid_argument("cache-bytes requires paged run and positive capacity");
        options.cache_bytes = static_cast<size_t>(count);
      } else {
        if (count > INT32_MAX) throw std::invalid_argument("generation option out of range: " + arg);
        if (arg == "--gpu-layers") options.gpu_layers = static_cast<int32_t>(count);
        else if (arg == "--max-tokens" && count) options.max_tokens = static_cast<int32_t>(count);
        else if (arg == "--prefill-chunk" && count) options.prefill_chunk = static_cast<uint32_t>(count);
        else if (arg == "--context" && count) options.context_length = static_cast<uint32_t>(count);
        else throw std::invalid_argument("unknown or invalid generation option: " + arg);
      }
    }
  }
  if (!prompt_seen || paths.empty()) throw std::invalid_argument("model shards and --prompt are required");
  if (!forcing_file.empty()) {
    constexpr uint64_t limit = 65536;
    if (!std::filesystem::is_regular_file(forcing_file) ||
        std::filesystem::file_size(forcing_file) > limit)
      throw std::invalid_argument("teacher-forcing-file must be a regular file <=65536 bytes");
    std::ifstream file(forcing_file, std::ios::binary);
    std::string raw(static_cast<size_t>(limit + 1), '\0');
    file.read(raw.data(), static_cast<std::streamsize>(raw.size()));
    const auto bytes = file.gcount();
    if (file.bad() || bytes > static_cast<std::streamsize>(limit))
      throw std::invalid_argument("teacher-forcing-file read exceeds byte limit or failed");
    raw.resize(static_cast<size_t>(bytes));
    std::istringstream stream(raw);
    std::string item;
    while (stream >> item) {
      const auto id = number(item);
      if (id > INT32_MAX || options.forced_tokens.size() >= limit)
        throw std::invalid_argument("teacher-forcing token ID/count is out of range");
      options.forced_tokens.push_back(static_cast<int32_t>(id));
    }
    if (options.forced_tokens.size() != static_cast<size_t>(options.max_tokens))
      throw std::invalid_argument("teacher-forcing-file must contain exactly max-tokens IDs");
  }
  if (!paged && paths.size() != 1) throw std::invalid_argument("reference takes the first shard; llama.cpp discovers its siblings");
  if (options.projection_activation == pmi::ProjectionActivationMode::CpuCompatible &&
      options.projection_backend != pmi::ProjectionBackend::DeviceVulkan)
    throw std::invalid_argument("cpu-compatible projection-activation requires run with device-vulkan");
  if (options.projection_backend != pmi::ProjectionBackend::Cpu) {
    if (!options.device_cache_bytes || !options.device_graph_bytes)
      throw std::invalid_argument("device projection requires explicit device-cache-bytes and device-graph-bytes");
    if (options.cache_policy != pmi::CachePolicy::Lru)
      throw std::invalid_argument("device projection currently requires lru; previous-top4 protects only host pages");
  } else if (options.device_cache_bytes || options.device_graph_bytes) {
    throw std::invalid_argument("device budgets require projection-backend device-cpu or device-vulkan");
  }
  if (options.projection_device_index && options.projection_backend != pmi::ProjectionBackend::DeviceVulkan)
    throw std::invalid_argument("projection-device-index is only valid for device-vulkan");
  auto result = paged ? pmi::run_paged(paths, prompt, options)
                      : pmi::run_reference(paths.front().string(), prompt, options);
  std::cout << result.text << "\nTOKENS";
  for (auto token : result.tokens) std::cout << ' ' << token;
  if (!result.predictions.empty()) {
    std::cout << "\nPREDICTIONS";
    for (auto token : result.predictions) std::cout << ' ' << token;
  }
  std::cout << '\n' << (paged ? "PMI_METRICS " : "REFERENCE_METRICS ")
            << pmi::generation_metrics_json(result) << '\n';
  return 0;
}
}

int main(int argc, char** argv) {
  try {
    if (argc < 2) { usage(); return 2; }
    const std::string command = argv[1];
    if (command == "self-test") return self_test(argc > 2 ? argv[2] : nullptr);
    if (command == "reference") return generation(argc, argv, false);
    if (command == "moe-layer") return moe_layer(argc, argv);
    if (command == "run") return generation(argc, argv, true);
    if (command == "inspect" && argc >= 3) {
      std::vector<pmi::GgufFile> shards;
      for (int i = 2; i < argc; ++i) shards.push_back(pmi::GgufReader(argv[i]).inspect());
      std::cout << pmi::manifest_json(pmi::build_manifest(shards)) << "\n";
      return 0;
    }
    usage();
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
