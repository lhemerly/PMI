#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace pmi {
struct GenerationResult { std::vector<int32_t> tokens; std::string text; };
GenerationResult run_reference(const std::string& model_path, const std::string& prompt,
                               int32_t max_tokens, int32_t gpu_layers);
}
