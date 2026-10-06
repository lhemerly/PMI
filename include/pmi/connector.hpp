#pragma once

#include "pmi/manifest.hpp"
#include "pmi/page_provider.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace pmi {

// This boundary executes ONE token's MoE sublayer after the model's HC mixer.
// It is not a decoder: attention, HC, PLE/ngram, tokenizer, and MTP are absent.
struct Qwen4ExpMoeConfig {
  int layer{};
  uint64_t embedding_length{};
  uint64_t expert_count{};
  uint64_t experts_used{};
  uint64_t expert_hidden_length{};
  uint64_t shared_hidden_length{};
  float expert_weights_scale{1.0f}; // upstream treats both 0 and 1 as identity
  std::string activation{"silu"};
  std::string routing{"softmax"};
  uint64_t expert_groups{1};
  bool normalize_topk{true};
  float swiglu_clamp{};
  int cpu_threads{1};
};

struct MoeExpertTrace {
  int32_t expert_id{-1}; // -1 denotes the shared expert
  std::vector<float> gate, up, activation, down;
};

struct MoeLayerTrace {
  std::vector<float> router_logits, router_probabilities;
  std::vector<int32_t> expert_ids;
  std::vector<float> expert_weights;
  std::vector<MoeExpertTrace> experts;
  std::vector<float> routed_output;
  float shared_gate_logit{}, shared_gate{};
  MoeExpertTrace shared;
  std::vector<float> output;
  // Weight leases only; trace vectors and ggml activation/workspace are separate.
  uint64_t largest_weight_page_bytes{};
  uint64_t weight_page_requests{};
};

class Qwen4ExpMoeConnector {
 public:
  // Requires the file/offset/overlap validation supplied by build_manifest().
  // Copies model-layout-validated descriptors; retains no manifest references.
  Qwen4ExpMoeConnector(const ModelManifest&, Qwen4ExpMoeConfig);
  const Qwen4ExpMoeConfig& config() const { return config_; }
  // GGML CPU consumes immutable pages directly. At most one weight page is
  // leased at a time, independently of expert count or provider cache capacity.
  MoeLayerTrace forward(PageProvider&, const std::vector<float>& input) const;
  // Payload sizes for resource planning; provider alignment padding is separate.
  uint64_t largest_weight_page_bytes() const;
  uint64_t total_weight_bytes() const;

 private:
  struct Expert { ManifestTensor gate, up, down; };
  Qwen4ExpMoeConfig config_;
  ManifestTensor router_, shared_gate_;
  std::vector<Expert> experts_;
  Expert shared_;
};

} // namespace pmi
