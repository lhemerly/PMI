#include "pmi/gguf.hpp"
#include "ggml.h"
#include <iostream>
#include <vector>
int main() {
  for (int id = 0; id < GGML_TYPE_COUNT; ++id) {
    const auto type = static_cast<ggml_type>(id);
    const auto block = ggml_blck_size(type);
    const auto bytes = ggml_type_size(type);
    if (block <= 0 || bytes == 0) continue; // removed/deprecated GGUF types
    try {
      const auto actual = pmi::ggml_tensor_nbytes(static_cast<uint32_t>(id), {static_cast<uint64_t>(block)});
      if (actual != bytes) {
        std::cerr << "GGML type " << id << " size mismatch: PMI=" << actual << " ggml=" << bytes << "\n";
        return 1;
      }
    } catch (const std::exception& e) {
      std::cerr << "GGML type " << id << " unsupported by PMI: " << e.what() << "\n";
      return 1;
    }
  }
  std::cout << "all serialized GGML type sizes match pinned ggml\n";
}
