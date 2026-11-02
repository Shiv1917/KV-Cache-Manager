// Paged attention over a block-table-indexed KV cache.
//
// Every query token is a "row" with its own context_len (= position + 1). One kernel therefore
// serves prefill and decode, and causal masking falls out of context_len for free.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace kvp {

enum class AttnVersion { Auto, V1, V2 };

struct PagedAttnParams {
  const half* q = nullptr;     // [num_rows, q_stride], heads packed as [head][head_dim]
  int q_stride = 0;            // halfs between rows
  half* out = nullptr;         // [num_rows, out_stride]
  int out_stride = 0;
  const half* k_cache = nullptr;  // [num_blocks][num_kv_heads][block_size][head_dim]
  const half* v_cache = nullptr;
  const int* block_tables = nullptr;  // [num_seqs][max_blocks_per_seq]
  int max_blocks_per_seq = 0;
  const int* row_seq = nullptr;       // [num_rows] row -> sequence index
  const int* context_lens = nullptr;  // [num_rows]
  int num_rows = 0;
  int num_heads = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  int block_size = 0;
  float scale = 0;
  int max_context_len = 0;     // upper bound over rows (sizes shared memory / partitions)
  // Ablation: assume each sequence's blocks are physically consecutive starting at its first
  // table entry (the contiguous-layout baseline), so the kernel skips the per-block lookup.
  bool assume_contiguous = false;
};

struct PagedAttnWorkspace {
  float* buf = nullptr;   // v2 scratch
  size_t floats = 0;
};

// Floats of scratch v2 needs for this launch.
size_t paged_attention_v2_workspace(int num_rows, int num_heads, int head_dim, int max_context_len,
                                    int partition_size = 512);
// Largest context v1 can handle on this device (bounded by shared memory for the score buffer).
int paged_attention_v1_max_context(int num_heads, int num_kv_heads);

// Compiled-kernel resources and theoretical occupancy for a launch (V1 or V2 main kernel).
struct PagedAttnKernelInfo {
  int regs = 0, static_smem = 0, dyn_smem = 0, threads = 0, blocks_per_sm = 0, ctas = 0;
};
PagedAttnKernelInfo paged_attention_kernel_info(const PagedAttnParams& p, AttnVersion v);

// Dispatches on head_dim / GQA group size. Auto picks v2 for long contexts when the workspace
// is large enough, v1 otherwise.
void paged_attention(const PagedAttnParams& p, AttnVersion version, const PagedAttnWorkspace& ws,
                     cudaStream_t stream);

}  // namespace kvp
