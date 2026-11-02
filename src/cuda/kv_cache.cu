#include "cuda/kv_cache.h"

namespace kvp {

KVCache::KVCache(const KVCacheConfig& cfg) : cfg_(cfg) {
  const size_t n = cfg.block_elems() * size_t(cfg.num_blocks);
  k_.reserve(cfg.num_layers);
  v_.reserve(cfg.num_layers);
  for (int l = 0; l < cfg.num_layers; ++l) {
    k_.emplace_back(n);
    v_.emplace_back(n);
    // Zero so that any (buggy) read of an unwritten slot is deterministic.
    CUDA_CHECK(cudaMemset(k_.back().get(), 0, k_.back().bytes()));
    CUDA_CHECK(cudaMemset(v_.back().get(), 0, v_.back().bytes()));
  }
}

// One thread block per token; threads stride over (kv_head, head_dim) in half2 units.
__global__ void reshape_and_cache_kernel(const half* __restrict__ k, const half* __restrict__ v,
                                         int row_stride, half* __restrict__ k_cache,
                                         half* __restrict__ v_cache,
                                         const int* __restrict__ slot_mapping, int num_kv_heads,
                                         int head_dim, int block_size) {
  const int t = blockIdx.x;
  const int slot = slot_mapping[t];
  if (slot < 0) return;
  const int block = slot / block_size, off = slot % block_size;
  const int n2 = num_kv_heads * head_dim / 2;
  const half2* ks = reinterpret_cast<const half2*>(k + size_t(t) * row_stride);
  const half2* vs = reinterpret_cast<const half2*>(v + size_t(t) * row_stride);
  for (int i = threadIdx.x; i < n2; i += blockDim.x) {
    const int e = 2 * i;
    const int h = e / head_dim, d = e % head_dim;
    const size_t dst = ((size_t(block) * num_kv_heads + h) * block_size + off) * head_dim + d;
    *reinterpret_cast<half2*>(k_cache + dst) = ks[i];
    *reinterpret_cast<half2*>(v_cache + dst) = vs[i];
  }
}

void reshape_and_cache(const half* k, const half* v, int row_stride, half* k_cache, half* v_cache,
                       const int* slot_mapping, int num_tokens, int num_kv_heads, int head_dim,
                       int block_size, cudaStream_t stream) {
  if (num_tokens == 0) return;
  const int threads = std::min(256, std::max(32, num_kv_heads * head_dim / 2));
  reshape_and_cache_kernel<<<num_tokens, threads, 0, stream>>>(
      k, v, row_stride, k_cache, v_cache, slot_mapping, num_kv_heads, head_dim, block_size);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace kvp
