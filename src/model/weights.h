// TinyLlama (Llama-architecture) fp16 weights, loaded from tools/export_weights.py output.
#pragma once
#include <cuda_fp16.h>

#include <string>
#include <vector>

#include "cuda/cuda_utils.h"

namespace kvp {

struct LlamaConfig {
  int vocab = 0, hidden = 0, inter = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0;
  int max_pos = 0;
  float rope_theta = 10000.f, rms_eps = 1e-5f;
  int qkv_dim() const { return (heads + 2 * kv_heads) * head_dim; }
  int kv_dim() const { return kv_heads * head_dim; }
};

struct LayerWeights {
  const half* attn_norm;  // [H]
  const half* wqkv;       // [(heads + 2 kv_heads) * hd, H]
  const half* wo;         // [H, H]
  const half* ffn_norm;   // [H]
  const half* w_gate_up;  // [2I, H]
  const half* w_down;     // [H, I]
};

class LlamaWeights {
 public:
  static LlamaConfig read_config(const std::string& path);
  explicit LlamaWeights(const std::string& path);
  const LlamaConfig& config() const { return cfg_; }
  const half* embed() const { return embed_; }
  const LayerWeights& layer(int l) const { return layers_[l]; }
  const half* final_norm() const { return final_norm_; }
  const half* lm_head() const { return lm_head_; }
  size_t bytes() const { return buf_.bytes(); }

 private:
  LlamaConfig cfg_;
  DeviceBuffer<half> buf_;
  const half *embed_ = nullptr, *final_norm_ = nullptr, *lm_head_ = nullptr;
  std::vector<LayerWeights> layers_;
};

}  // namespace kvp
