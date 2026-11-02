// Llama forward pass over a flattened batch of "rows" (one per query token). Prefill and decode
// take the same path: each row carries its position, cache slot and context length.
#pragma once
#include <cublas_v2.h>

#include <vector>

#include "cuda/kv_cache.h"
#include "cuda/paged_attention.h"
#include "model/weights.h"

namespace kvp {

// Host-side description of one forward step.
struct ForwardBatch {
  std::vector<int32_t> tokens;        // [rows]
  std::vector<int32_t> positions;     // [rows]
  std::vector<int32_t> slot_mapping;  // [rows] block_id * B + offset
  std::vector<int32_t> row_seq;       // [rows] index into block_tables
  std::vector<int32_t> context_lens;  // [rows] position + 1
  std::vector<int32_t> block_tables;  // [num_seqs * max_blocks]
  std::vector<int32_t> logit_rows;    // rows whose logits are needed (usually last row per seq)
  int num_seqs = 0;
  int max_blocks = 0;
  int max_context_len = 0;

  int num_rows() const { return static_cast<int>(tokens.size()); }
  void clear() {
    tokens.clear(); positions.clear(); slot_mapping.clear(); row_seq.clear();
    context_lens.clear(); block_tables.clear(); logit_rows.clear();
    num_seqs = 0; max_context_len = 0;
  }
};

struct ModelLimits {
  int max_tokens = 4096;      // rows per step
  int max_seqs = 256;         // sequences per step
  int max_logit_rows = 256;   // rows we can return logits for
  int max_blocks_per_seq = 128;
  int max_context_len = 2048;
};

class LlamaModel {
 public:
  LlamaModel(const LlamaWeights& w, KVCache& cache, const ModelLimits& lim);
  ~LlamaModel();
  LlamaModel(const LlamaModel&) = delete;
  LlamaModel& operator=(const LlamaModel&) = delete;

  // Runs the batch, writes argmax token per logit row into `next`. If `logits` is non-null it
  // receives [logit_rows, vocab] fp32 logits.
  void forward(const ForwardBatch& b, std::vector<int32_t>& next, std::vector<float>* logits);

  void set_attention_version(AttnVersion v) { attn_version_ = v; }
  const LlamaConfig& config() const { return w_.config(); }
  cudaStream_t stream() const { return stream_; }
  size_t workspace_bytes() const { return workspace_bytes_; }

 private:
  // Y[rows, n] = X[rows, k] * W[n, k]^T (fp16 in, fp32 accumulate)
  void gemm(const half* X, const half* W, void* Y, int rows, int n, int k, bool fp32_out);

  const LlamaWeights& w_;
  KVCache& cache_;
  ModelLimits lim_;
  AttnVersion attn_version_ = AttnVersion::Auto;
  cudaStream_t stream_ = nullptr;
  cublasHandle_t cublas_ = nullptr;
  size_t workspace_bytes_ = 0;

  // Packed int inputs: tokens | positions | slots | row_seq | ctx | logit_rows | tables
  int32_t* h_ints_ = nullptr;  // pinned
  DeviceBuffer<int32_t> d_ints_;
  DeviceBuffer<float> x_;        // residual stream, fp32 [T, H]
  DeviceBuffer<half> h_;         // normed activations [T, H]
  DeviceBuffer<half> qkv_;       // [T, qkv_dim]
  DeviceBuffer<half> attn_;      // [T, H]
  DeviceBuffer<half> proj_;      // [T, H] (o_proj / down_proj output)
  DeviceBuffer<half> gate_up_;   // [T, 2I]
  DeviceBuffer<half> act_;       // [T, I]
  DeviceBuffer<half> h_last_;    // [L, H]
  DeviceBuffer<float> logits_;   // [L, V]
  DeviceBuffer<int32_t> argmax_; // [L]
  DeviceBuffer<float> attn_ws_;  // v2 scratch
};

}  // namespace kvp
