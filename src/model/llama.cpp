#include "model/llama.h"

#include <cmath>
#include <cstring>

#include "cuda/kernels.h"

namespace kvp {

LlamaModel::LlamaModel(const LlamaWeights& w, KVCache& cache, const ModelLimits& lim)
    : w_(w), cache_(cache), lim_(lim) {
  const LlamaConfig& c = w.config();
  CUDA_CHECK(cudaStreamCreate(&stream_));
  CUBLAS_CHECK(cublasCreate(&cublas_));
  CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
  CUBLAS_CHECK(cublasSetMathMode(cublas_, CUBLAS_DEFAULT_MATH));

  const size_t T = lim.max_tokens, L = lim.max_logit_rows;
  const size_t ints = 5 * T + L + size_t(lim.max_seqs) * lim.max_blocks_per_seq;
  CUDA_CHECK(cudaMallocHost(&h_ints_, ints * sizeof(int32_t)));
  d_ints_.resize(ints);
  x_.resize(T * c.hidden);
  h_.resize(T * c.hidden);
  qkv_.resize(T * c.qkv_dim());
  attn_.resize(T * c.hidden);
  proj_.resize(T * c.hidden);
  gate_up_.resize(T * 2 * c.inter);
  act_.resize(T * c.inter);
  h_last_.resize(L * c.hidden);
  logits_.resize(L * c.vocab);
  argmax_.resize(L);
  attn_ws_.resize(paged_attention_v2_workspace(lim.max_seqs, c.heads, c.head_dim,
                                               lim.max_context_len));
  workspace_bytes_ = d_ints_.bytes() + x_.bytes() + h_.bytes() + qkv_.bytes() + attn_.bytes() +
                     proj_.bytes() + gate_up_.bytes() + act_.bytes() + h_last_.bytes() +
                     logits_.bytes() + argmax_.bytes() + attn_ws_.bytes();
}

LlamaModel::~LlamaModel() {
  if (h_ints_) cudaFreeHost(h_ints_);
  if (cublas_) cublasDestroy(cublas_);
  if (stream_) cudaStreamDestroy(stream_);
}

void LlamaModel::gemm(const half* X, const half* W, void* Y, int rows, int n, int k,
                      bool fp32_out) {
  // Row-major Y[rows,n] = X[rows,k] W[n,k]^T  <=>  column-major Y^T[n,rows] = W^T' X^T
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k, &alpha, W, CUDA_R_16F,
                            k, X, CUDA_R_16F, k, &beta, Y, fp32_out ? CUDA_R_32F : CUDA_R_16F, n,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

void LlamaModel::forward(const ForwardBatch& b, std::vector<int32_t>& next,
                         std::vector<float>* logits_out) {
  const LlamaConfig& c = w_.config();
  const int T = b.num_rows();
  const int S = b.num_seqs;
  const int L = static_cast<int>(b.logit_rows.size());
  if (T == 0) { next.clear(); return; }
  if (T > lim_.max_tokens || S > lim_.max_seqs || L > lim_.max_logit_rows ||
      b.max_blocks > lim_.max_blocks_per_seq || b.max_context_len > lim_.max_context_len)
    throw std::runtime_error("forward: batch exceeds model limits");

  // ---- upload all integer inputs with a single H2D copy
  int32_t* hp = h_ints_;
  auto put = [&](const std::vector<int32_t>& v) {
    std::memcpy(hp, v.data(), v.size() * sizeof(int32_t));
    int32_t* dst = d_ints_.get() + (hp - h_ints_);
    hp += v.size();
    return dst;
  };
  const int* d_tok = put(b.tokens);
  const int* d_pos = put(b.positions);
  const int* d_slot = put(b.slot_mapping);
  const int* d_rowseq = put(b.row_seq);
  const int* d_ctx = put(b.context_lens);
  const int* d_lrows = put(b.logit_rows);
  const int* d_tables = put(b.block_tables);
  CUDA_CHECK(cudaMemcpyAsync(d_ints_.get(), h_ints_, (hp - h_ints_) * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_));

  const int H = c.hidden, QKV = c.qkv_dim(), I = c.inter;
  embedding_lookup(d_tok, w_.embed(), x_.get(), T, H, stream_);

  PagedAttnParams ap;
  ap.q = qkv_.get();
  ap.q_stride = QKV;
  ap.out = attn_.get();
  ap.out_stride = H;
  ap.block_tables = d_tables;
  ap.max_blocks_per_seq = b.max_blocks;
  ap.row_seq = d_rowseq;
  ap.context_lens = d_ctx;
  ap.num_rows = T;
  ap.num_heads = c.heads;
  ap.num_kv_heads = c.kv_heads;
  ap.head_dim = c.head_dim;
  ap.block_size = cache_.config().block_size;
  ap.scale = 1.f / std::sqrt(float(c.head_dim));
  ap.max_context_len = b.max_context_len;
  PagedAttnWorkspace ws{attn_ws_.get(), attn_ws_.size()};

  for (int l = 0; l < c.layers; ++l) {
    const LayerWeights& lw = w_.layer(l);
    // attention block
    rmsnorm(x_.get(), nullptr, lw.attn_norm, h_.get(), T, H, c.rms_eps, stream_);
    gemm(h_.get(), lw.wqkv, qkv_.get(), T, QKV, H, false);
    rope_inplace(qkv_.get(), QKV, d_pos, T, c.heads, c.kv_heads, c.head_dim, c.rope_theta,
                 stream_);
    reshape_and_cache(qkv_.get() + H, qkv_.get() + H + c.kv_dim(), QKV, cache_.k(l), cache_.v(l),
                      d_slot, T, c.kv_heads, c.head_dim, cache_.config().block_size, stream_);
    ap.k_cache = cache_.k(l);
    ap.v_cache = cache_.v(l);
    paged_attention(ap, attn_version_, ws, stream_);
    gemm(attn_.get(), lw.wo, proj_.get(), T, H, H, false);
    add_residual(x_.get(), proj_.get(), size_t(T) * H, stream_);
    // MLP block
    rmsnorm(x_.get(), nullptr, lw.ffn_norm, h_.get(), T, H, c.rms_eps, stream_);
    gemm(h_.get(), lw.w_gate_up, gate_up_.get(), T, 2 * I, H, false);
    silu_mul(gate_up_.get(), act_.get(), T, I, stream_);
    gemm(act_.get(), lw.w_down, proj_.get(), T, H, I, false);
    add_residual(x_.get(), proj_.get(), size_t(T) * H, stream_);
  }

  // Only the rows we sample from go through the final norm and the LM head.
  rmsnorm(x_.get(), d_lrows, w_.final_norm(), h_last_.get(), L, H, c.rms_eps, stream_);
  gemm(h_last_.get(), w_.lm_head(), logits_.get(), L, c.vocab, H, true);
  argmax_rows(logits_.get(), argmax_.get(), L, c.vocab, stream_);

  next.resize(L);
  CUDA_CHECK(cudaMemcpyAsync(next.data(), argmax_.get(), L * sizeof(int32_t),
                             cudaMemcpyDeviceToHost, stream_));
  if (logits_out) {
    logits_out->resize(size_t(L) * c.vocab);
    CUDA_CHECK(cudaMemcpyAsync(logits_out->data(), logits_.get(),
                               logits_out->size() * sizeof(float), cudaMemcpyDeviceToHost,
                               stream_));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace kvp
