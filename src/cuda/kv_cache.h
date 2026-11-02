// Physical KV cache: one cudaMalloc per layer for K and one for V.
//   K[layer]: [num_blocks][num_kv_heads][block_size][head_dim]  (fp16), same for V.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>

#include "cuda/cuda_utils.h"

namespace kvp {

struct KVCacheConfig {
  int num_layers = 22;
  int num_blocks = 0;
  int block_size = 16;
  int num_kv_heads = 4;
  int head_dim = 64;

  size_t block_elems() const { return size_t(num_kv_heads) * block_size * head_dim; }
  // Bytes for K and V of one block across all layers.
  size_t bytes_per_block() const { return 2 * size_t(num_layers) * block_elems() * sizeof(half); }
  size_t bytes_per_token() const { return bytes_per_block() / block_size; }
};

class KVCache {
 public:
  explicit KVCache(const KVCacheConfig& cfg);
  const KVCacheConfig& config() const { return cfg_; }
  half* k(int layer) const { return k_[layer].get(); }
  half* v(int layer) const { return v_[layer].get(); }
  size_t total_bytes() const { return cfg_.bytes_per_block() * size_t(cfg_.num_blocks); }

 private:
  KVCacheConfig cfg_;
  std::vector<DeviceBuffer<half>> k_, v_;
};

// Scatters new K/V rows into the cache. k/v point at token 0's K/V inside a row-major buffer
// with `row_stride` halfs between tokens (lets us read straight out of the fused QKV output).
// slot = block_id * block_size + offset; slot < 0 => skip (padding).
void reshape_and_cache(const half* k, const half* v, int row_stride, half* k_cache, half* v_cache,
                       const int* slot_mapping, int num_tokens, int num_kv_heads, int head_dim,
                       int block_size, cudaStream_t stream);

}  // namespace kvp
