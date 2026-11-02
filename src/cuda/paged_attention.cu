// Paged attention kernels.
//
// v1: grid = (kv_head, row). One thread block handles the GROUP query heads that share a KV head
//     (GQA), so every K/V element loaded from the cache is reused GROUP times.
//       pass 1: QK^T into a shared-memory score buffer, walking the block table. Each token is
//               handled by HD/8 lanes doing one 16-byte load each (4 tokens per warp for HD=64)
//       pass 2: block-wide softmax per query head
//       pass 3: P*V accumulated in registers, then reduced across lanes / warps
// v2: same body, but the context is split into fixed-size partitions across blockIdx.z (more
//     parallelism for long contexts / small batches), followed by a log-sum-exp merge kernel.
#include <algorithm>
#include <cfloat>

#include "cuda/cuda_utils.h"
#include "cuda/paged_attention.h"

namespace kvp {

namespace {

constexpr int kNumWarps = 4;
constexpr int kPartitionSize = 512;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Each token is handled by a "thread group" of TG = HD / 8 lanes, each lane owning 8 consecutive
// elements of head_dim (one 16-byte load). A warp therefore covers TPW = 32 / TG tokens per
// iteration, and a QK dot product needs only log2(TG) shuffles instead of 5.
struct alignas(16) Half8 { half2 h[4]; };

__device__ __forceinline__ void load8(const half* p, float* out) {
  const Half8 v = *reinterpret_cast<const Half8*>(p);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __half22float2(v.h[i]);
    out[2 * i] = f.x;
    out[2 * i + 1] = f.y;
  }
}

template <int HD, int GROUP, bool PARTITIONED, bool CONTIG>
__global__ void __launch_bounds__(kNumWarps * 32)
    paged_attention_kernel(PagedAttnParams p, int part_size, int score_stride, float* tmp_out,
                           float* part_max, float* part_sum, int num_parts) {
  constexpr int VEC = 8;           // halfs per lane per token
  constexpr int TG = HD / VEC;     // lanes per token
  constexpr int TPW = 32 / TG;     // tokens per warp iteration
  constexpr int STEP = kNumWarps * TPW;
  static_assert(HD % VEC == 0 && TG <= 32, "unsupported head_dim");

  const int kvh = blockIdx.x;
  const int row = blockIdx.y;
  const int part = blockIdx.z;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int sub = lane / TG, lid = lane % TG;

  const int ctx = p.context_lens[row];
  const int start = part * part_size;
  if (start >= ctx) return;
  const int end = min(ctx, start + part_size);

  extern __shared__ float scores[];  // [GROUP][score_stride]
  __shared__ float red[kNumWarps][GROUP][HD];
  __shared__ float s_max[GROUP], s_sum[GROUP];

  const int* table = p.block_tables + size_t(p.row_seq[row]) * p.max_blocks_per_seq;
  const int B = p.block_size;
  const size_t head_stride = size_t(B) * HD;
  const size_t block_stride = head_stride * p.num_kv_heads;

  float q[GROUP][VEC];
#pragma unroll
  for (int g = 0; g < GROUP; ++g) {
    load8(p.q + size_t(row) * p.q_stride + size_t(kvh * GROUP + g) * HD + lid * VEC, q[g]);
#pragma unroll
    for (int e = 0; e < VEC; ++e) q[g][e] *= p.scale;
  }

  auto token_ptr = [&](const half* cache, int t) {
    const int lb = t / B;
    const int phys = CONTIG ? table[0] + lb : table[lb];
    return cache + phys * block_stride + kvh * head_stride + size_t(t % B) * HD + lid * VEC;
  };

  // ---- pass 1: scores = q . k
  for (int base = start + warp * TPW; base < end; base += STEP) {
    const int t = base + sub;
    const bool valid = t < end;
    float k[VEC];
    if (valid) load8(token_ptr(p.k_cache, t), k);
    else {
#pragma unroll
      for (int e = 0; e < VEC; ++e) k[e] = 0.f;
    }
#pragma unroll
    for (int g = 0; g < GROUP; ++g) {
      float s = 0.f;
#pragma unroll
      for (int e = 0; e < VEC; ++e) s += q[g][e] * k[e];
#pragma unroll
      for (int o = TG / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
      if (valid && lid == 0) scores[g * score_stride + (t - start)] = s;
    }
  }
  __syncthreads();

  // ---- pass 2: softmax (one warp per query head)
  const int n = end - start;
  for (int g = warp; g < GROUP; g += kNumWarps) {
    float* sg = scores + g * score_stride;
    float m = -FLT_MAX;
    for (int i = lane; i < n; i += 32) m = fmaxf(m, sg[i]);
    m = warp_max(m);
    float sum = 0.f;
    for (int i = lane; i < n; i += 32) {
      const float e = __expf(sg[i] - m);
      sg[i] = e;
      sum += e;
    }
    sum = warp_sum(sum);
    if (lane == 0) { s_max[g] = m; s_sum[g] = sum; }
  }
  __syncthreads();

  // ---- pass 3: out = P . V  (each lane accumulates its 8 dims for its token subset)
  float acc[GROUP][VEC];
#pragma unroll
  for (int g = 0; g < GROUP; ++g)
#pragma unroll
    for (int e = 0; e < VEC; ++e) acc[g][e] = 0.f;
  for (int base = start + warp * TPW; base < end; base += STEP) {
    const int t = base + sub;
    if (t < end) {
      float v[VEC];
      load8(token_ptr(p.v_cache, t), v);
#pragma unroll
      for (int g = 0; g < GROUP; ++g) {
        const float w = scores[g * score_stride + (t - start)];
#pragma unroll
        for (int e = 0; e < VEC; ++e) acc[g][e] += w * v[e];
      }
    }
  }
  // Sum over the TPW token sub-groups of the warp.
#pragma unroll
  for (int g = 0; g < GROUP; ++g)
#pragma unroll
    for (int e = 0; e < VEC; ++e)
#pragma unroll
      for (int o = TG; o < 32; o <<= 1) acc[g][e] += __shfl_xor_sync(0xffffffffu, acc[g][e], o);
  if (sub == 0) {
#pragma unroll
    for (int g = 0; g < GROUP; ++g)
#pragma unroll
      for (int e = 0; e < VEC; ++e) red[warp][g][lid * VEC + e] = acc[g][e];
  }
  __syncthreads();

  for (int i = threadIdx.x; i < GROUP * HD; i += blockDim.x) {
    const int g = i / HD, d = i % HD;
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < kNumWarps; ++w) s += red[w][g][d];
    s /= s_sum[g];
    const int head = kvh * GROUP + g;
    if constexpr (!PARTITIONED) {
      p.out[size_t(row) * p.out_stride + size_t(head) * HD + d] = __float2half(s);
    } else {
      const size_t idx = (size_t(row) * p.num_heads + head) * num_parts + part;
      tmp_out[idx * HD + d] = s;
      if (d == 0) { part_max[idx] = s_max[g]; part_sum[idx] = s_sum[g]; }
    }
  }
}

// Merges v2 partitions: out = sum_p w_p * o_p / sum_p w_p, w_p = sum_p * exp(max_p - max).
template <int HD>
__global__ void paged_attention_reduce_kernel(PagedAttnParams p, int part_size,
                                              const float* tmp_out, const float* part_max,
                                              const float* part_sum, int num_parts) {
  const int head = blockIdx.x, row = blockIdx.y, d = threadIdx.x;
  const int ctx = p.context_lens[row];
  const int np = (ctx + part_size - 1) / part_size;
  const size_t base = (size_t(row) * p.num_heads + head) * num_parts;
  float m = -FLT_MAX;
  for (int i = 0; i < np; ++i) m = fmaxf(m, part_max[base + i]);
  float num = 0.f, den = 0.f;
  for (int i = 0; i < np; ++i) {
    const float w = part_sum[base + i] * __expf(part_max[base + i] - m);
    num += w * tmp_out[(base + i) * HD + d];
    den += w;
  }
  p.out[size_t(row) * p.out_stride + size_t(head) * HD + d] = __float2half(num / den);
}

int smem_optin() {
  static int v = [] {
    int dev = 0, x = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&x, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    return x;
  }();
  return v;
}
int num_sms() {
  static int v = [] {
    int dev = 0, x = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&x, cudaDevAttrMultiProcessorCount, dev));
    return x;
  }();
  return v;
}
template <int HD, int GROUP>
constexpr int static_smem() {
  return kNumWarps * GROUP * HD * 4 + 2 * GROUP * 4;
}

// Static + dynamic smem above 48 KB needs an explicit opt-in; do it once per instantiation,
// using the compiler's actual static allocation. Returns the largest dynamic smem allowed.
template <int HD, int GROUP, bool PART, bool CONTIG>
size_t configure() {
  static const size_t max_dyn = [] {
    auto kernel = paged_attention_kernel<HD, GROUP, PART, CONTIG>;
    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    const size_t m = size_t(smem_optin()) - attr.sharedSizeBytes;
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(m)));
    return m;
  }();
  return max_dyn;
}

template <bool PART>
void launch_geometry(const PagedAttnParams& p, int group, int* part_size, int* num_parts,
                     int* score_stride, size_t* dyn) {
  *part_size = PART ? kPartitionSize : std::max(p.max_context_len, 1);
  *num_parts = PART ? (p.max_context_len + kPartitionSize - 1) / kPartitionSize : 1;
  *score_stride = (*part_size + 3) & ~3;
  *dyn = size_t(group) * *score_stride * sizeof(float);
}

template <int HD, int GROUP, bool PART, bool CONTIG>
PagedAttnKernelInfo info(const PagedAttnParams& p) {
  auto kernel = paged_attention_kernel<HD, GROUP, PART, CONTIG>;
  configure<HD, GROUP, PART, CONTIG>();
  int part_size, num_parts, score_stride;
  size_t dyn;
  launch_geometry<PART>(p, GROUP, &part_size, &num_parts, &score_stride, &dyn);
  cudaFuncAttributes attr{};
  CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
  PagedAttnKernelInfo k;
  k.regs = attr.numRegs;
  k.static_smem = static_cast<int>(attr.sharedSizeBytes);
  k.dyn_smem = static_cast<int>(dyn);
  k.threads = kNumWarps * 32;
  k.ctas = p.num_kv_heads * p.num_rows * num_parts;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&k.blocks_per_sm, kernel, k.threads,
                                                           dyn));
  return k;
}

template <int HD, int GROUP, bool PART, bool CONTIG>
void launch(const PagedAttnParams& p, const PagedAttnWorkspace& ws, cudaStream_t stream) {
  auto kernel = paged_attention_kernel<HD, GROUP, PART, CONTIG>;
  int part_size, num_parts, score_stride;
  size_t dyn;
  launch_geometry<PART>(p, GROUP, &part_size, &num_parts, &score_stride, &dyn);
  if (dyn > configure<HD, GROUP, PART, CONTIG>())
    throw std::runtime_error("paged_attention: context too long for v1 shared memory");
  float *tmp = nullptr, *pmax = nullptr, *psum = nullptr;
  if (PART) {
    const size_t idx = size_t(p.num_rows) * p.num_heads * num_parts;
    if (ws.floats < idx * (HD + 2)) throw std::runtime_error("paged_attention: v2 workspace too small");
    tmp = ws.buf;
    pmax = tmp + idx * HD;
    psum = pmax + idx;
  }
  dim3 grid(p.num_kv_heads, p.num_rows, num_parts);
  kernel<<<grid, kNumWarps * 32, dyn, stream>>>(p, part_size, score_stride, tmp, pmax, psum,
                                               num_parts);
  if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
    throw std::runtime_error(std::string("paged_attention launch failed: ") +
                             cudaGetErrorString(e) + " grid=(" + std::to_string(grid.x) + "," +
                             std::to_string(grid.y) + "," + std::to_string(grid.z) +
                             ") dyn_smem=" + std::to_string(dyn) + " static_smem=" +
                             std::to_string(static_smem<HD, GROUP>()) +
                             " max_ctx=" + std::to_string(p.max_context_len));
  if (PART) {
    paged_attention_reduce_kernel<HD><<<dim3(p.num_heads, p.num_rows), HD, 0, stream>>>(
        p, part_size, tmp, pmax, psum, num_parts);
    CUDA_CHECK(cudaGetLastError());
  }
}

template <int HD, int GROUP>
void dispatch_version(const PagedAttnParams& p, bool v2, const PagedAttnWorkspace& ws,
                      cudaStream_t s) {
  if (v2) {
    if (p.assume_contiguous) launch<HD, GROUP, true, true>(p, ws, s);
    else launch<HD, GROUP, true, false>(p, ws, s);
  } else {
    if (p.assume_contiguous) launch<HD, GROUP, false, true>(p, ws, s);
    else launch<HD, GROUP, false, false>(p, ws, s);
  }
}

template <int HD>
void dispatch_group(const PagedAttnParams& p, bool v2, const PagedAttnWorkspace& ws,
                    cudaStream_t s) {
  switch (p.num_heads / p.num_kv_heads) {
    case 1: return dispatch_version<HD, 1>(p, v2, ws, s);
    case 4: return dispatch_version<HD, 4>(p, v2, ws, s);
    case 8: return dispatch_version<HD, 8>(p, v2, ws, s);
    default: throw std::runtime_error("paged_attention: unsupported GQA group size");
  }
}

template <int HD>
PagedAttnKernelInfo info_group(const PagedAttnParams& p, bool v2) {
  switch (p.num_heads / p.num_kv_heads) {
    case 1: return v2 ? info<HD, 1, true, false>(p) : info<HD, 1, false, false>(p);
    case 4: return v2 ? info<HD, 4, true, false>(p) : info<HD, 4, false, false>(p);
    case 8: return v2 ? info<HD, 8, true, false>(p) : info<HD, 8, false, false>(p);
    default: throw std::runtime_error("paged_attention: unsupported GQA group size");
  }
}

}  // namespace

size_t paged_attention_v2_workspace(int num_rows, int num_heads, int head_dim, int max_context_len,
                                    int partition_size) {
  const size_t parts = (max_context_len + partition_size - 1) / partition_size;
  return size_t(num_rows) * num_heads * parts * (head_dim + 2);
}

int paged_attention_v1_max_context(int num_heads, int num_kv_heads) {
  const int group = num_heads / num_kv_heads;
  const int stat = kNumWarps * group * 128 * 4 + 2 * group * 4;  // conservative (head_dim 128)
  return ((smem_optin() - stat) / (group * 4)) & ~3;
}

void paged_attention(const PagedAttnParams& p, AttnVersion version, const PagedAttnWorkspace& ws,
                     cudaStream_t stream) {
  if (p.num_rows == 0) return;
  if (p.num_heads % p.num_kv_heads) throw std::runtime_error("heads % kv_heads != 0");
  bool v2 = version == AttnVersion::V2;
  if (version == AttnVersion::Auto) {
    const bool fits_v1 = p.max_context_len <= paged_attention_v1_max_context(p.num_heads, p.num_kv_heads);
    const bool ws_ok = ws.floats >= paged_attention_v2_workspace(p.num_rows, p.num_heads,
                                                                 p.head_dim, p.max_context_len);
    // Measured on RTX 4080 (results/kernel_attn.csv): v1's score buffer grows with the context
    // (64 KB at 2048 tokens, 1 CTA/SM), so from ~1K tokens the partitioned v2 wins at every batch
    // size; below that v1 avoids the extra merge kernel. Low-parallelism batches favour v2 too.
    const bool low_parallelism = p.num_rows * p.num_kv_heads < num_sms();
    v2 = ws_ok && (!fits_v1 || p.max_context_len >= 1024 ||
                   (p.max_context_len >= 2 * kPartitionSize && low_parallelism));
  }
  if (p.head_dim == 64) dispatch_group<64>(p, v2, ws, stream);
  else if (p.head_dim == 128) dispatch_group<128>(p, v2, ws, stream);
  else throw std::runtime_error("paged_attention: head_dim must be 64 or 128");
}

}  // namespace kvp

namespace kvp {
PagedAttnKernelInfo paged_attention_kernel_info(const PagedAttnParams& p, AttnVersion v) {
  const bool v2 = v == AttnVersion::V2;
  if (p.head_dim == 64) return info_group<64>(p, v2);
  if (p.head_dim == 128) return info_group<128>(p, v2);
  throw std::runtime_error("paged_attention: head_dim must be 64 or 128");
}
}  // namespace kvp
