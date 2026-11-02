// Small elementwise / reduction kernels for the Llama forward pass.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace kvp {

// x[t, :] = float(embed[ids[t], :])
void embedding_lookup(const int* ids, const half* table, float* x, int num_tokens, int hidden,
                      cudaStream_t s);
// out[r, :] = half(w * x[src_row, :] / rms(x[src_row, :])). rows == nullptr => identity.
void rmsnorm(const float* x, const int* rows, const half* w, half* out, int num_rows, int hidden,
             float eps, cudaStream_t s);
// Rotary embedding (HF "rotate_half" convention) applied in place to the q heads at column 0 and
// the k heads at column num_heads * head_dim of a [num_tokens, row_stride] buffer.
void rope_inplace(half* qkv, int row_stride, const int* positions, int num_tokens, int num_heads,
                  int num_kv_heads, int head_dim, float theta, cudaStream_t s);
// out[t, i] = silu(gu[t, i]) * gu[t, inter + i]
void silu_mul(const half* gate_up, half* out, int num_tokens, int inter, cudaStream_t s);
// x += float(y), elementwise over n
void add_residual(float* x, const half* y, size_t n, cudaStream_t s);
// out[r] = argmax_j logits[r, j] (first index on ties)
void argmax_rows(const float* logits, int* out, int num_rows, int vocab, cudaStream_t s);

}  // namespace kvp
