#include "pmi/cache.hpp"
#include "pmi/gguf.hpp"
#include "pmi/inference.hpp"
#include "pmi/manifest.hpp"
#include "pmi/page_provider.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void usage() {
  std::cerr << "usage:\n"
            << "  pmi inspect MODEL.gguf\n"
            << "  pmi reference MODEL.gguf --prompt TEXT [--max-tokens N] [--gpu-layers N]\n"
            << "  pmi self-test [MODEL.gguf]\n";
}
int self_test(const char* fixture) {
  pmi::LruByteCache cache(8);
  if (!cache.put(1, {1, 2, 3, 4}) || !cache.get(1) || cache.resident_bytes() != 4)
    throw std::runtime_error("cache basic test failed");
  if (!cache.pin(1) || cache.put(1, {9, 9}) || !cache.get(1) || (*cache.get(1))[0] != 1)
    throw std::runtime_error("cache pin invariant failed");
  if (cache.put(2, {5, 6, 7, 8, 9}) || cache.resident_bytes() > cache.capacity())
    throw std::runtime_error("pinned cache eviction invariant failed");
  cache.unpin(1);
  if (!cache.put(2, {5, 6, 7, 8, 9}) || cache.resident_bytes() > cache.capacity() || cache.get(1))
    throw std::runtime_error("cache bounded LRU test failed");

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
int reference(int argc, char** argv) {
  if (argc < 5) { usage(); return 2; }
  std::string model = argv[2], prompt;
  int32_t max_tokens = 32, gpu_layers = 0;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--prompt" && i + 1 < argc) prompt = argv[++i];
    else if (arg == "--max-tokens" && i + 1 < argc) max_tokens = std::stoi(argv[++i]);
    else if (arg == "--gpu-layers" && i + 1 < argc) gpu_layers = std::stoi(argv[++i]);
    else throw std::runtime_error("unknown or incomplete reference option: " + arg);
  }
  if (prompt.empty()) throw std::runtime_error("--prompt is required");
  auto result = pmi::run_reference(model, prompt, max_tokens, gpu_layers);
  std::cout << result.text << "\nTOKENS";
  for (auto token : result.tokens) std::cout << " " << token;
  std::cout << "\n";
  return 0;
}
}

int main(int argc, char** argv) {
  try {
    if (argc < 2) { usage(); return 2; }
    const std::string command = argv[1];
    if (command == "self-test") return self_test(argc > 2 ? argv[2] : nullptr);
    if (command == "reference") return reference(argc, argv);
    if (command == "run") {
      throw std::runtime_error("PMI graph execution is not implemented yet; use `pmi reference` for the llama.cpp baseline");
    }
    if (command == "inspect" && argc >= 3) {
      pmi::GgufReader reader(argv[2]);
      std::cout << pmi::manifest_json(pmi::build_manifest(reader.inspect())) << "\n";
      return 0;
    }
    usage();
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
