#include <cfloat>

#include "cuda/cuda_utils.h"
#include "cuda/kernels.h"

namespace kvp {

namespace {

template <int THREADS>
__device__ float block_sum(float v) {
  __shared__ float sh[THREADS / 32];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = v;
  __syncthreads();
  if (threadIdx.x < 32) {
    v = threadIdx.x < THREADS / 32 ? sh[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x == 0) sh[0] = v;
  }
  __syncthreads();
  v = sh[0];
  __syncthreads();
  return v;
}

__global__ void embedding_kernel(const int* ids, const half* table, float* x, int hidden) {
  const int t = blockIdx.x;
  const half* src = table + size_t(ids[t]) * hidden;
  float* dst = x + size_t(t) * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) dst[i] = __half2float(src[i]);
}

constexpr int kNormThreads = 256;
__global__ void __launch_bounds__(kNormThreads)
    rmsnorm_kernel(const float* x, const int* rows, const half* w, half* out, int hidden,
                   float eps) {
  const int r = blockIdx.x;
  const int src = rows ? rows[r] : r;
  const float* xr = x + size_t(src) * hidden;
  float ss = 0.f;
  for (int i = threadIdx.x; i < hidden; i += kNormThreads) ss += xr[i] * xr[i];
  ss = block_sum<kNormThreads>(ss);
  const float inv = rsqrtf(ss / hidden + eps);
  half* o = out + size_t(r) * hidden;
  // HF: weight * (x * inv).to(dtype); with an fp16 model the cast happens before the multiply.
  for (int i = threadIdx.x; i < hidden; i += kNormThreads)
    o[i] = __hmul(w[i], __float2half(xr[i] * inv));
}

__global__ void rope_kernel(half* qkv, int row_stride, const int* positions, int num_heads,
                            int num_kv_heads, int head_dim, float theta) {
  const int t = blockIdx.x;
  const int half_dim = head_dim / 2;
  const float pos = float(positions[t]);
  half* row = qkv + size_t(t) * row_stride;
  const int total = (num_heads + num_kv_heads) * half_dim;
  for (int i = threadIdx.x; i < total; i += blockDim.x) {
    const int h = i / half_dim, j = i % half_dim;
    // Same arithmetic as HF: inv_freq in fp32, angle = pos * inv_freq in fp32.
    const float inv_freq = 1.0f / powf(theta, float(2 * j) / float(head_dim));
    float sn, cs;
    sincosf(pos * inv_freq, &sn, &cs);
    half* base = row + size_t(h) * head_dim;  // k heads follow q heads contiguously
    const float a = __half2float(base[j]);
    const float b = __half2float(base[j + half_dim]);
    base[j] = __float2half(a * cs - b * sn);
    base[j + half_dim] = __float2half(b * cs + a * sn);
  }
}

__global__ void silu_mul_kernel(const half* gu, half* out, int inter) {
  const int t = blockIdx.x;
  const half* g = gu + size_t(t) * 2 * inter;
  const half* u = g + inter;
  half* o = out + size_t(t) * inter;
  for (int i = threadIdx.x; i < inter; i += blockDim.x) {
    const float x = __half2float(g[i]);
    const float silu = x / (1.f + __expf(-x));
    o[i] = __float2half(__half2float(__float2half(silu)) * __half2float(u[i]));
  }
}

__global__ void add_residual_kernel(float* x, const half* y, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    x[i] += __half2float(y[i]);
}

constexpr int kArgmaxThreads = 1024;
__global__ void __launch_bounds__(kArgmaxThreads)
    argmax_kernel(const float* logits, int* out, int vocab) {
  const float* l = logits + size_t(blockIdx.x) * vocab;
  float best = -FLT_MAX;
  int idx = 0x7fffffff;
  for (int i = threadIdx.x; i < vocab; i += kArgmaxThreads) {
    const float v = l[i];
    if (v > best) { best = v; idx = i; }
  }
  __shared__ float sv[kArgmaxThreads / 32];
  __shared__ int si[kArgmaxThreads / 32];
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
    if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
  }
  if (threadIdx.x % 32 == 0) { sv[threadIdx.x / 32] = best; si[threadIdx.x / 32] = idx; }
  __syncthreads();
  if (threadIdx.x < 32) {
    best = sv[threadIdx.x];
    idx = si[threadIdx.x];
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
      if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
    }
    if (threadIdx.x == 0) out[blockIdx.x] = idx;
  }
}

}  // namespace

void embedding_lookup(const int* ids, const half* table, float* x, int num_tokens, int hidden,
                      cudaStream_t s) {
  if (!num_tokens) return;
  embedding_kernel<<<num_tokens, 256, 0, s>>>(ids, table, x, hidden);
  CUDA_CHECK(cudaGetLastError());
}

void rmsnorm(const float* x, const int* rows, const half* w, half* out, int num_rows, int hidden,
             float eps, cudaStream_t s) {
  if (!num_rows) return;
  rmsnorm_kernel<<<num_rows, kNormThreads, 0, s>>>(x, rows, w, out, hidden, eps);
  CUDA_CHECK(cudaGetLastError());
}

void rope_inplace(half* qkv, int row_stride, const int* positions, int num_tokens, int num_heads,
                  int num_kv_heads, int head_dim, float theta, cudaStream_t s) {
  if (!num_tokens) return;
  rope_kernel<<<num_tokens, 256, 0, s>>>(qkv, row_stride, positions, num_heads, num_kv_heads,
                                         head_dim, theta);
  CUDA_CHECK(cudaGetLastError());
}

void silu_mul(const half* gate_up, half* out, int num_tokens, int inter, cudaStream_t s) {
  if (!num_tokens) return;
  silu_mul_kernel<<<num_tokens, 256, 0, s>>>(gate_up, out, inter);
  CUDA_CHECK(cudaGetLastError());
}

void add_residual(float* x, const half* y, size_t n, cudaStream_t s) {
  if (!n) return;
  const int blocks = static_cast<int>(std::min<size_t>((n + 255) / 256, 4096));
  add_residual_kernel<<<blocks, 256, 0, s>>>(x, y, n);
  CUDA_CHECK(cudaGetLastError());
}

void argmax_rows(const float* logits, int* out, int num_rows, int vocab, cudaStream_t s) {
  if (!num_rows) return;
  argmax_kernel<<<num_rows, kArgmaxThreads, 0, s>>>(logits, out, vocab);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace kvp
