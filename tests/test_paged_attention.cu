// Kernel-level tests: random Q/K/V, shuffled block tables, naive fp32 CPU reference.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include "cuda/kv_cache.h"
#include "cuda/paged_attention.h"

using namespace kvp;

namespace {

struct Problem {
  int heads = 32, kv_heads = 4, head_dim = 64, block_size = 16;
  std::vector<int> ctx_lens;   // one sequence per entry
  bool all_rows = false;       // prefill-style: one row per position (else last position only)
  bool shuffle = true;
  uint32_t seed = 1;
};

struct Result {
  std::vector<float> gpu, ref;
};

float h2f(half h) { return __half2float(h); }

// Logical K/V for sequence s, token t, kv head h -> physical block via table.
Result run(const Problem& P, AttnVersion ver, bool assume_contig = false,
           std::vector<int>* tables_out = nullptr, const std::vector<int>* force_tables = nullptr) {
  std::mt19937 rng(P.seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  const int S = static_cast<int>(P.ctx_lens.size());
  const int B = P.block_size, HD = P.head_dim;
  const int max_ctx = *std::max_element(P.ctx_lens.begin(), P.ctx_lens.end());
  const int max_blocks = (max_ctx + B - 1) / B;
  int total_blocks = 0;
  for (int c : P.ctx_lens) total_blocks += (c + B - 1) / B;
  const int num_blocks = total_blocks + 7;  // a few spare blocks that nobody references

  // Logical K/V values (fp16-rounded so CPU and GPU see identical inputs).
  std::vector<std::vector<half>> Kl(S), Vl(S);
  for (int s = 0; s < S; ++s) {
    Kl[s].resize(size_t(P.ctx_lens[s]) * P.kv_heads * HD);
    Vl[s].resize(Kl[s].size());
    for (auto& x : Kl[s]) x = __float2half(nd(rng));
    for (auto& x : Vl[s]) x = __float2half(nd(rng));
  }

  // Block tables: physical ids either shuffled or sequential.
  std::vector<int> phys(num_blocks);
  std::iota(phys.begin(), phys.end(), 0);
  std::mt19937 shuf_rng(P.seed + 77);  // separate stream: shuffling must not change the data
  if (P.shuffle) std::shuffle(phys.begin(), phys.end(), shuf_rng);
  std::vector<int> tables(size_t(S) * max_blocks, 0);
  if (force_tables) {
    tables = *force_tables;
  } else {
    int next = 0;
    for (int s = 0; s < S; ++s)
      for (int b = 0; b < (P.ctx_lens[s] + B - 1) / B; ++b) tables[s * max_blocks + b] = phys[next++];
  }
  if (tables_out) *tables_out = tables;

  // Physical cache [num_blocks][kv_heads][B][HD].
  std::vector<half> Kc(size_t(num_blocks) * P.kv_heads * B * HD, __float2half(0.f));
  std::vector<half> Vc(Kc.size(), __float2half(0.f));
  for (int s = 0; s < S; ++s)
    for (int t = 0; t < P.ctx_lens[s]; ++t)
      for (int h = 0; h < P.kv_heads; ++h)
        for (int d = 0; d < HD; ++d) {
          const int blk = tables[s * max_blocks + t / B];
          const size_t dst = ((size_t(blk) * P.kv_heads + h) * B + t % B) * HD + d;
          const size_t src = (size_t(t) * P.kv_heads + h) * HD + d;
          Kc[dst] = Kl[s][src];
          Vc[dst] = Vl[s][src];
        }

  // Rows.
  std::vector<int> row_seq, ctx;
  for (int s = 0; s < S; ++s) {
    if (P.all_rows) {
      for (int t = 0; t < P.ctx_lens[s]; ++t) { row_seq.push_back(s); ctx.push_back(t + 1); }
    } else {
      row_seq.push_back(s); ctx.push_back(P.ctx_lens[s]);
    }
  }
  const int R = static_cast<int>(row_seq.size());
  const int qstride = P.heads * HD;
  std::vector<half> Q(size_t(R) * qstride);
  for (auto& x : Q) x = __float2half(nd(rng));

  // CPU reference (fp32).
  const float scale = 1.f / std::sqrt(float(HD));
  const int G = P.heads / P.kv_heads;
  Result res;
  res.ref.resize(size_t(R) * qstride);
  std::vector<float> sc;
  for (int r = 0; r < R; ++r) {
    const int s = row_seq[r], n = ctx[r];
    for (int h = 0; h < P.heads; ++h) {
      const int kvh = h / G;
      sc.assign(n, 0.f);
      float m = -1e30f;
      for (int t = 0; t < n; ++t) {
        float dot = 0;
        for (int d = 0; d < HD; ++d)
          dot += h2f(Q[size_t(r) * qstride + h * HD + d]) *
                 h2f(Kl[s][(size_t(t) * P.kv_heads + kvh) * HD + d]);
        sc[t] = dot * scale;
        m = std::max(m, sc[t]);
      }
      float sum = 0;
      for (auto& x : sc) { x = std::exp(x - m); sum += x; }
      for (int d = 0; d < HD; ++d) {
        float o = 0;
        for (int t = 0; t < n; ++t) o += sc[t] * h2f(Vl[s][(size_t(t) * P.kv_heads + kvh) * HD + d]);
        res.ref[size_t(r) * qstride + h * HD + d] = o / sum;
      }
    }
  }

  // GPU.
  DeviceBuffer<half> dQ(Q.size()), dO(Q.size()), dK(Kc.size()), dV(Vc.size());
  DeviceBuffer<int> dT(tables.size()), dRS(R), dC(R);
  CUDA_CHECK(cudaMemcpy(dQ.get(), Q.data(), Q.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK.get(), Kc.data(), Kc.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV.get(), Vc.data(), Vc.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dT.get(), tables.data(), tables.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dRS.get(), row_seq.data(), R * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dC.get(), ctx.data(), R * 4, cudaMemcpyHostToDevice));
  DeviceBuffer<float> ws(paged_attention_v2_workspace(R, P.heads, HD, max_ctx));
  PagedAttnParams p;
  p.q = dQ.get(); p.q_stride = qstride; p.out = dO.get(); p.out_stride = qstride;
  p.k_cache = dK.get(); p.v_cache = dV.get();
  p.block_tables = dT.get(); p.max_blocks_per_seq = max_blocks;
  p.row_seq = dRS.get(); p.context_lens = dC.get();
  p.num_rows = R; p.num_heads = P.heads; p.num_kv_heads = P.kv_heads; p.head_dim = HD;
  p.block_size = B; p.scale = scale; p.max_context_len = max_ctx;
  p.assume_contiguous = assume_contig;
  paged_attention(p, ver, {ws.get(), ws.size()}, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<half> O(Q.size());
  CUDA_CHECK(cudaMemcpy(O.data(), dO.get(), O.size() * 2, cudaMemcpyDeviceToHost));
  res.gpu.resize(O.size());
  for (size_t i = 0; i < O.size(); ++i) res.gpu[i] = h2f(O[i]);
  return res;
}

float max_abs_err(const Result& r) {
  float e = 0;
  for (size_t i = 0; i < r.gpu.size(); ++i) e = std::max(e, std::fabs(r.gpu[i] - r.ref[i]));
  return e;
}

}  // namespace

class PagedAttnSweep : public ::testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(PagedAttnSweep, MatchesCpuReference) {
  auto [block_size, ctx, version] = GetParam();
  Problem P;
  P.block_size = block_size;
  P.ctx_lens = {ctx, std::max(1, ctx / 2), 1, std::min(ctx + 3, 2047)};
  P.seed = block_size * 7919 + ctx;
  Result r = run(P, version == 1 ? AttnVersion::V1 : AttnVersion::V2);
  EXPECT_LT(max_abs_err(r), 1e-2) << "B=" << block_size << " ctx=" << ctx << " v" << version;
}

INSTANTIATE_TEST_SUITE_P(BlocksCtxVersion, PagedAttnSweep,
                         ::testing::Combine(::testing::Values(8, 16, 32, 64),
                                            ::testing::Values(1, 15, 16, 17, 2047),
                                            ::testing::Values(1, 2)));

TEST(PagedAttn, PrefillRowsCausal) {
  Problem P;
  P.ctx_lens = {33, 70, 5};
  P.all_rows = true;
  for (auto v : {AttnVersion::V1, AttnVersion::V2}) {
    Result r = run(P, v);
    EXPECT_LT(max_abs_err(r), 1e-2);
  }
}

// Regression: dynamic smem < 48 KB but dynamic + static > 48 KB still needs the opt-in.
// (Run alone with --gtest_filter=PagedAttn.SmemOptInBoundary to exercise the first launch.)
TEST(PagedAttn, SmemOptInBoundary) {
  Problem P;
  P.ctx_lens = {1300, 1000};
  Result r = run(P, AttnVersion::V1);
  EXPECT_LT(max_abs_err(r), 1e-2);
}

TEST(PagedAttn, OtherHeadConfigs) {
  for (auto [heads, kvh, hd] : {std::tuple{8, 8, 64}, std::tuple{16, 4, 128}, std::tuple{32, 4, 128}}) {
    Problem P;
    P.heads = heads; P.kv_heads = kvh; P.head_dim = hd;
    P.ctx_lens = {100, 1, 600};
    Result r = run(P, AttnVersion::V1);
    EXPECT_LT(max_abs_err(r), 1e-2) << heads << "/" << kvh << "/" << hd;
    Result r2 = run(P, AttnVersion::V2);
    EXPECT_LT(max_abs_err(r2), 1e-2) << heads << "/" << kvh << "/" << hd << " v2";
  }
}

TEST(PagedAttn, ShuffledLayoutIdenticalToSequential) {
  for (auto v : {AttnVersion::V1, AttnVersion::V2}) {
    Problem P;
    P.ctx_lens = {300, 17, 1200};
    P.shuffle = true;
    Result a = run(P, v);
    P.shuffle = false;
    Result b = run(P, v);
    ASSERT_EQ(a.gpu.size(), b.gpu.size());
    for (size_t i = 0; i < a.gpu.size(); ++i) ASSERT_EQ(a.gpu[i], b.gpu[i]) << "at " << i;
  }
}

TEST(PagedAttn, ContiguousFastPathMatchesTableWalk) {
  Problem P;
  P.ctx_lens = {500, 64, 1000};
  P.shuffle = false;  // sequential tables => blocks of each sequence are consecutive
  Result a = run(P, AttnVersion::V1, false);
  Result b = run(P, AttnVersion::V1, true);
  for (size_t i = 0; i < a.gpu.size(); ++i) ASSERT_EQ(a.gpu[i], b.gpu[i]);
}

TEST(ReshapeAndCache, ScattersToSlots) {
  const int T = 37, KVH = 4, HD = 64, B = 16, NB = 10, stride = 3 * KVH * HD;  // padded rows
  std::mt19937 rng(3);
  std::vector<half> src(size_t(T) * stride);
  for (auto& x : src) x = __float2half(float(rng() % 1000) / 100.f);
  std::vector<int> slots(T);
  std::vector<int> perm(NB * B);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);
  for (int t = 0; t < T; ++t) slots[t] = perm[t];
  slots[5] = -1;  // padding row is skipped
  DeviceBuffer<half> dsrc(src.size()), kc(size_t(NB) * KVH * B * HD), vc(kc.size());
  DeviceBuffer<int> dslots(T);
  CUDA_CHECK(cudaMemcpy(dsrc.get(), src.data(), src.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dslots.get(), slots.data(), T * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(kc.get(), 0, kc.bytes()));
  CUDA_CHECK(cudaMemset(vc.get(), 0, vc.bytes()));
  reshape_and_cache(dsrc.get(), dsrc.get() + KVH * HD, stride, kc.get(), vc.get(), dslots.get(), T,
                    KVH, HD, B, nullptr);
  std::vector<half> K(kc.size()), V(vc.size());
  CUDA_CHECK(cudaMemcpy(K.data(), kc.get(), kc.bytes(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(V.data(), vc.get(), vc.bytes(), cudaMemcpyDeviceToHost));
  for (int t = 0; t < T; ++t) {
    if (slots[t] < 0) continue;
    const int blk = slots[t] / B, off = slots[t] % B;
    for (int h = 0; h < KVH; ++h)
      for (int d = 0; d < HD; ++d) {
        const size_t dst = ((size_t(blk) * KVH + h) * B + off) * HD + d;
        ASSERT_EQ(h2f(K[dst]), h2f(src[size_t(t) * stride + h * HD + d]));
        ASSERT_EQ(h2f(V[dst]), h2f(src[size_t(t) * stride + KVH * HD + h * HD + d]));
      }
  }
  // The skipped row's natural slot stays zero.
  const int blk = perm[5] / B, off = perm[5] % B;
  EXPECT_EQ(h2f(K[((size_t(blk) * KVH) * B + off) * HD]), 0.f);
}
