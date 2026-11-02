# kvpage: design and evaluation

**Setup.** RTX 4080 (16 GB, sm_89, 717 GB/s), CUDA 13.1, TinyLlama-1.1B-Chat in fp16 (22 layers,
32 query heads, 4 KV heads, head_dim 64). One token of KV costs
2 × 22 × 4 × 64 × 2 B = **22,528 B**. The GPU was shared with another user's job holding
10.5 GB, so GPU experiments use KV pools of 0.5–2 GB. The CPU simulator covers 1–8 GB (it
reproduces the engine's scheduling within ~1%; see §5).

## 1. Architecture

```
 requests ─► Scheduler (FCFS, continuous batching, preempt-by-recompute)
                │ can_admit / admit / append_slot / commit / release
                ▼
            KVManager ◄── the only thing that changes between experiments
            ├─ PagedManager       BlockAllocator (free list + refcounts) + PrefixCache
            └─ ContiguousManager  first-fit over free extents, reserve ceil(max_len/B) blocks
                │ block tables (int32 physical block ids)
                ▼
            Engine ─► LlamaModel.forward(tokens, positions, slot_mapping, block_tables, ctx_lens)
                         cuBLAS GEMMs · RMSNorm · RoPE · reshape_and_cache · paged_attention
```

* **Physical cache.** One allocation per layer for K and one for V, laid out as
  `[num_blocks][kv_heads][B][head_dim]`. Logical token `t` of a sequence lives in
  block `table[t / B]` at offset `t % B`; `slot = block * B + offset`.
* **One code path for prefill and decode.** Every query token is a row with
  `context_len = pos + 1`. A step computes tokens `[num_computed, len)` of each scheduled
  sequence, so decode is the case of exactly one row. Causal masking falls out of
  `context_len`, and a prefill that resumes after a prefix hit just starts at `num_computed > 0`.
* **Fair baseline.** `ContiguousManager` uses the same block tables, kernel, scheduler and
  model. It reserves `ceil(max_seq_len / B)` *consecutive* blocks at admission (128 blocks for
  2048 tokens at B = 16), like a framework that pre-allocates a max-length KV buffer per
  request. Allocation policy is therefore the only variable. Both managers produce
  **bit-identical** tokens.
* **Scheduler.** Each step first grows every running sequence by one slot (oldest first). If
  the pool is exhausted it preempts the youngest sequence: its blocks are freed and it goes back
  to the front of the queue, to be recomputed later. It then admits waiting requests in strict
  FCFS order while `can_admit` succeeds and the token budget (4096) allows. Paged admission keeps
  a 1% free-block **watermark**, as vLLM does. Without it, a burst admits until the pool is full
  and then thrashes: the watermark cut preemptions from 178 to 62 in simulation at no
  throughput cost.

## 2. Prefix caching

* **Hash chain.** `h_i = xxh64(seed = h_{i-1}, tokens of block i)`. Only full blocks are
  cacheable. A hit verifies the stored parent hash and the block's tokens, so a 64-bit collision
  cannot alias two different prefixes.
* **Lookup at admission.** Walk the prompt block by block, `incref` each hit, and stop at the
  first miss. The block holding the last prompt token is never matched, because we need that
  token's logits. This also means a shared block is never written.
* **Registration happens after compute, not at admission.** `commit()` runs after each forward
  step and hashes newly completed blocks. If blocks were registered at admission, a second
  request admitted in the same step could "hit" a block whose K/V the first request is still
  writing in that same kernel launch.
* **Release.** At refcount 0, a hashed block goes onto an LRU list, where it stays valid and
  findable. An unhashed block goes onto the free list. `allocate()` takes from the free list
  first and only then evicts the LRU head. Blocks are released tail-first, so the end of a
  prefix is evicted before its head; the head is the part other requests are most likely to
  share.
* **No copy-on-write needed.** Shared blocks are always full and read-only. Writes only go to a
  sequence's private tail block.
* **Admission accounting.** `can_admit` counts a hit on an LRU block as consuming that
  evictable block. Otherwise admission could over-commit the pool.

## 3. Attention kernel

* **v1:** grid = (kv_head, row), 4 warps. One CTA serves the 8 query heads of a GQA group, so
  each K/V element is read from DRAM once and used 8 times.
  * Pass 1: QKᵀ into a shared-memory score buffer, walking the block table.
  * Pass 2: block-wide softmax.
  * Pass 3: PV accumulated in registers, reduced across lanes and warps.
* **Thread groups.** The first version used warp-per-token with lanes over head_dim: 5
  shuffles × 8 heads per token and 4-byte loads. That reached only 5–30% of bandwidth. Moving
  to **8-lane thread groups**, where each lane does one 16-byte load and a warp covers 4 tokens
  per iteration, made the kernel **3–4× faster**. (vLLM uses the same thread-group idea.)
* **v2 (flash-decoding):** the context is split into 512-token partitions across
  `blockIdx.z`, and a second kernel does a log-sum-exp merge.
  * Auto policy: v2 for contexts of 1024 or more, otherwise v1.
  * Reason: v1's score buffer grows with the context (64 KB at 2048 tokens). That allows
    1 CTA per SM, i.e. 8% occupancy, and bandwidth falls to ~36%. v2's per-CTA smem is fixed
    at 16 KB.
* **Ablation flag `assume_contiguous`.** It computes `block = table[0] + t / B` instead of
  `table[t / B]`. This gives an honest "same kernel, contiguous layout" baseline that measures
  the cost of indirection alone.

**Results** (`results/kernel_attn.csv`, one layer, B = 16):

| batch × ctx | paged v1 | paged v2 | contiguous (best) | paged, best % of peak BW |
|---|---|---|---|---|
| 1 × 2048 | 128 µs | 37 µs | 35 µs | 8% (16 CTAs for 76 SMs) |
| 8 × 2048 | 126 µs | 39 µs | 37 µs | 60% |
| 32 × 512 | 37 µs | 39 µs | 35 µs | 64% |
| 32 × 2048 | 283 µs | 117 µs | 111 µs | 80% |
| 128 × 512 | 107 µs | 118 µs | 99 µs | 88% |
| 128 × 2048 | 1043 µs | 477 µs | 464 µs | 79% |

Small working sets (batch 128 × ctx ≤ 256, ≤ 67 MB) are partly L2-resident (64 MB L2), so they
exceed 100% of DRAM bandwidth.

**The cost of paging is 3–10%.** With B = 16 the table walk is a dependent load every
16 tokens. It also scatters the 2 KB per-head chunks, so consecutive blocks no longer share
DRAM pages and the L2 prefetch stream breaks. The overhead shrinks with block size
(7.0% → 3.3% from B = 8 to B = 64 at batch 32 × 512; `kernel_blocksize.csv`).

**Why it's slower than vLLM:**
1. v1 allocates a full-context fp32 score buffer, which limits occupancy. vLLM's v1 also keeps
   logits in smem, but at 512-token granularity via v2 partitioning, and it uses 4–8 warps per
   CTA with a configurable partition count.
2. QK and PV each make one pass with fp32 FMAs. There is no half2 math and no tensor cores.
3. Batch-1 decode exposes too little parallelism: 4 KV heads × 4 partitions = 16 CTAs on 76
   SMs. Smaller partitions, or splitting the GQA group, would help.
4. Prefill uses the same row-per-token kernel. That is correct and simple but quadratic in
   memory traffic; a real system would use a FlashAttention-style prefill kernel.

Nsight Compute could not be used: the shared machine sets `RmProfilingAdminOnly=1`, so
hardware counters are admin-only. The achieved bandwidth above is computed from bytes read
divided by event-timed duration, and the register, smem and occupancy figures come from
`cudaFuncGetAttributes` and the occupancy API (`kvkernel` prints them). An Nsight Systems
timeline of a serving run is in `results/` (see §6).

## 4. Correctness (`kvcorrect`)

The reference is HF transformers in fp32 on CPU: greedy decoding with EOS ignored, 8 chat
prompts (63–70 tokens) that share a system prompt, 64 steps each.

| check | result |
|---|---|
| greedy tokens vs HF, each prompt alone, **scattered** block tables | 8/8 prompts, 64/64 tokens |
| teacher-forced top-1 agreement (prompt + HF continuation in one pass) | 512/512 (100%) |
| top-5 logit error vs fp32 HF | mean 0.0023, max 0.0148 |
| paged vs contiguous | 8/8 bit-identical |
| all 8 prompts in one batch | 8/8 identical to solo runs (no cross-contamination) |
| prefix caching on (7 of 8 prompts hit the shared system prompt) | 18 block hits, 288 prefill tokens saved, outputs identical to cache-off |

Unit tests: 22 CPU tests and 49 CUDA tests.
* The allocator leak/alias test runs 200k random operations.
* The prefix-cache tests cover hit/miss, LRU and tail-first eviction order, admission
  accounting and full-prompt hits.
* The kernel sweep covers block sizes {8, 16, 32, 64} × contexts {1, 15, 16, 17, 2047} × {v1, v2}
  against an fp32 CPU reference, with max error < 1e-2.
* The shuffled block layout is bit-identical to the sequential one.

## 5. Experiments

All figures and CSVs are in `results/`, regenerated by `scripts/run_experiments.sh` and `scripts/plot.gp`.

Workload: chat-like lognormal lengths, user prompt median 148 tokens (σ = 1.0) and output median
200 (σ = 0.9), clamped so that prompt + output ≤ 2048 = `max_seq_len`. Token contents are random,
except for the shared system prompt.

### 5.1 Fragmentation vs block size (`fig_fragmentation.png`, simulator, 2 GB pool)

| B | paged waste | paged block-table entries / seq | contiguous waste |
|---|---|---|---|
| 8 | 1.0% | 61 | 75.4% |
| 16 | 1.8% | 31 | 75.4% |
| 32 | 3.2% | 16 | 75.4% |
| 64 | 6.0% | 8 | 75.4% |

* **Paged.** All of the waste is the partial last block, about B/2 slots per sequence, so it
  grows linearly with B.
* **Contiguous.** Nearly all of its waste is *reserved-but-unused* slots (75%). Its internal
  tail waste is tiny; the reservation is the problem.
* **Block size and throughput.** On the GPU, block size barely affects throughput (7,314 →
  7,156 tok/s from B = 8 to 64, −2%). Smaller blocks cost table metadata and slightly more
  kernel indirection, but they pack better. **B = 16 is a good default**, and B = 8 would also
  be fine at this model size.

### 5.2 Capacity (`fig_capacity.png`, `fig_throughput_pool.png`)

| KV pool | max concurrent (paged / contiguous) | GPU output tok/s (paged / contiguous) |
|---|---|---|
| 0.5 GB | 97 / 11 (GPU) | 5,524 / 2,026 (**2.7×**) |
| 1 GB | 198 / 23 (GPU), 201 / 23 (sim) | 7,313 / 3,873 (**1.9×**) |
| 2 GB | 376 / 46 (GPU), 400 / 46 (sim) | 9,686 / 5,803 (**1.7×**) |
| 4 GB | 772 / 93 (sim) | — |
| 8 GB | 1,516 / 186 (sim) | — |

* **Concurrency.** Paged sustains **~8× more concurrent requests** at every pool size.
  Contiguous capacity is exactly `pool / (128 blocks × 16 × 22.5 KB)`.
* **Why throughput grows less than capacity.** A bigger batch makes each step slower: 5.5 ms
  at batch ~21 vs 8.7 ms at ~64 vs 11.4 ms at ~110. Decode at this size is bound by weight
  bandwidth plus attention. Preemption recompute also costs paged some prefill (118k vs 91k
  prompt tokens at 1 GB). The gain is largest when memory is scarcest.

### 5.3 Online serving vs arrival rate (`fig_rate.png`, 1 GB pool, 300 Poisson requests)

| rate (req/s) | tok/s paged / contiguous | TTFT p99 paged / contiguous | mean batch |
|---|---|---|---|
| 4–12 | identical (1,225 → 3,245) | 0.05–0.06 s / 0.05–0.48 s | ≤ 18 |
| 16 | 4,094 / 3,532 | 0.07 s / 4.05 s | 23 / 19 |
| 24 | 5,516 / 3,552 | 0.07 s / 9.66 s | 36 / 20 |
| 32 | 6,462 / 3,559 | 0.40 s / 12.6 s | 47 / 20 |

* **Below saturation the policies are indistinguishable.** That is the expected result:
  paging buys nothing until memory, not compute, limits the batch.
* **Contiguous hits a hard ceiling** of 23 sequences and ~3.55k tok/s. Beyond it, requests queue
  and TTFT explodes.
* **Paged keeps scaling** and only starts preempting at 32 req/s (10 preemptions).

### 5.4 Prefix caching (`fig_prefix.png`, `fig_prefix_saved.png`)

GPU, 1 GB pool, 300 requests at 12 req/s, with a 512-token system prompt in front of a fraction
of the requests:

| shared fraction | mean TTFT off → on | p99 TTFT off → on | prefill tokens off → on | block hit rate |
|---|---|---|---|---|
| 0 | 15.4 → 15.4 ms | 65 → 66 ms | 64.8k → 64.8k | 0% |
| 0.3 | 26.5 → 18.9 ms | 108 → 90 ms | 102.7k → 65.3k | 37% |
| 0.6 | 42.3 → 21.2 ms | 166 → 100 ms | 152.4k → 65.3k | 58% |
| 0.9 | 70.3 → 24.6 ms (**2.9×**) | 250 → 113 ms | 200.5k → 65.3k (**3.1×**) | 68% |

* **Prefill work becomes independent of sharing.** With caching on, prefill tokens stay flat at
  65.3k, which is exactly the unshared part of the workload. Every shared block after the first
  request is a hit.
* **Throughput moves only modestly** (3,427 → 3,616 tok/s at 0.9) because 12 req/s is below
  saturation. The benefit shows up as latency.
* **With no sharing, caching costs nothing measurable.** TTFT is identical. The 7k evictions are
  just LRU blocks of finished requests being recycled.
* **Simulator, saturated load** (2 GB pool, 2000 requests): prefill tokens grow from 478k to 1.38M
  as sharing goes from 0 to 0.9 with the cache off, and stay at about 478k with it on. Shared
  blocks also count once against the pool, so caching increases capacity as well.

### 5.5 Simulator fidelity

On the same 300-request burst at 1 GB, the simulator predicts 1,562 steps, 178 preemptions and
a max batch of 199. The GPU engine measured 1,567, 190 and 198. That justifies using the
simulator for the 4 and 8 GB capacity points that do not fit on the shared GPU.

## 6. Profiling

**Nsight Systems** (`nsys profile`, paged, 1 GB, 150-request burst; kernel summary in
`results/nsys_kern_cuda_gpu_kern_sum.csv`):

| GPU time share | kernels |
|---|---|
| 68.0% | cuBLAS GEMMs (QKV, O, gate/up, down, LM head; tensor-core HMMA kernels) |
| 22.7% | paged attention, main kernel (decode + row-per-token prefill) |
| 8.0% | small kernels: RMSNorm, RoPE, SiLU·mul, residual add, embedding, argmax |
| 0.9% | v2 partition merge |
| 0.4% | reshape_and_cache |

At this model size the engine is GEMM-bound. Attention is under a quarter of GPU time, so the
3–10% paging overhead costs **< 2.5% end to end**. The capacity win (§5.2) dominates it by two
orders of magnitude. The ~8% in small kernels points to fusion as the next optimisation:
RMSNorm into the following GEMM's prologue, RoPE with reshape_and_cache, and residual add into
the GEMM epilogue.

**Kernel resources** (`kvkernel` via `cudaFuncGetAttributes` and the occupancy API):
* Both kernels use 95 registers per thread. With 128-thread CTAs that caps occupancy at
  5 CTAs per SM (42%) even when shared memory is small.
* v1 at ctx 2048 needs 72 KB of smem, which leaves 1 CTA per SM (8% occupancy). That is the
  drop from 68% to 33% of bandwidth.
* v2 is fixed at 24 KB, which allows 3 CTAs per SM.
* At batch 1 the grid is only 0.01–0.07 waves, i.e. latency-bound by construction.

**Nsight Compute** returned `ERR_NVGPUCTRPERM`: performance counters are admin-only on this
shared machine (`RmProfilingAdminOnly=1`), so the roofline numbers above come from timing rather
than hardware counters.

## 7. Limitations

* **Greedy sampling only, no EOS handling** in benchmarks (fixed output lengths), fp16 weights
  without quantization, single GPU.
* **Prefill uses the row-per-token paged kernel**: O(n²) K/V reads, roughly 2× slower than a
  FlashAttention prefill. It is fine for evaluating memory behaviour, but it understates
  end-to-end throughput.
* **Preemption is by recomputation only** (no swap to host). Preempted sequences keep their
  generated tokens and re-prefill them. With prefix caching enabled, they can re-hit their own
  blocks if those have not been evicted.
* **The contiguous baseline uses first-fit over block extents.** With a single reservation
  size, external fragmentation never materializes. Variable per-request reservations (e.g.
  `prompt + max_new_tokens`) would be a stronger baseline; that is closer to what modern
  "static" servers do and would narrow, but not close, the gap.
* **Prefix-hash verification compares the stored parent hash and the block's tokens.** A
  collision earlier in a chain is not re-verified transitively; the probability is negligible at
  64 bits.
* **Random token IDs** in benchmarks: memory behaviour does not depend on content, but the
  numbers say nothing about output quality.
