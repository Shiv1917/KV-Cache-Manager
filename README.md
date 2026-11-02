# kvpage — a PagedAttention-style KV cache manager in C++/CUDA

A small LLM serving engine built to study one variable: **KV-cache allocation policy**.
It runs TinyLlama-1.1B (fp16, Llama architecture) on an RTX 4080. Two policies are compared
behind one `KVManager` interface, using the same engine, scheduler and attention kernel:

| policy | what it does |
|---|---|
| `PagedManager` | allocates fixed-size blocks on demand (vLLM-style), with optional hash-chained **prefix caching** |
| `ContiguousManager` | reserves `ceil(max_seq_len / B)` *consecutive* blocks per request at admission |

The headline numbers (details and plots are in [`docs/design.md`](docs/design.md)):

* **Correctness:** greedy output matches HuggingFace token for token (64/64 on all 8 reference
  prompts). Paged and contiguous are bit-identical. Prefix caching on and off are identical.
* **Memory waste** (2 GB pool, chat-like lengths): paged 1.8% vs contiguous 75%.
* **Capacity:** 8.2× more concurrent requests on a 2 GB pool (376 vs 46 on the GPU).
* **Throughput** (offline burst): 1.7–2.7× more output tokens/s, depending on pool size.
* **Attention kernel:** decode reaches 60–88% of DRAM bandwidth once the batch fills the GPU. Paged
  indirection costs 3–10% against the same kernel on a contiguous layout.

## Layout

```
tools/export_weights.py   the only Python: weights -> flat fp16 binary, HF reference outputs
src/core/                 block allocator, prefix cache, KV managers, scheduler (CPU only)
src/sim/                  workload generator + time-stepped simulator (CPU only)
src/cuda/                 KV cache, reshape_and_cache, paged attention v1/v2, small kernels
src/model/                weight loader, Llama forward pass (cuBLAS GEMMs)
src/engine/               engine (scheduler + KV manager + model), metrics
src/main_*.cpp            kvsim, kvbench, kvcorrect, kvkernel
tests/                    GoogleTest: allocator/managers/prefix cache/scheduler, kernels
scripts/                  run_experiments.sh, plot.gp (gnuplot)
results/                  CSVs and figures from the RTX 4080 runs
docs/design.md            design doc / writeup
```

## Build

Requirements: CMake ≥ 3.24 and a C++17 compiler. For the GPU targets you also need the CUDA
Toolkit (tested with 13.1) and an sm_89 GPU (RTX 40xx); otherwise set `CMAKE_CUDA_ARCHITECTURES`.
GoogleTest is fetched automatically.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release        # CUDA targets are skipped if nvcc is absent
cmake --build build -j
./build/test_core && ./build/test_cuda
```

Without CUDA (e.g. on a laptop), only `kvp_core`, `kvsim` and `test_core` are built.

**glibc ≥ 2.41 with CUDA 13.1:** glibc declares `rsqrt` as `noexcept` and CUDA's
`math_functions.h` doesn't, so nvcc fails with "exception specification is incompatible". On
the DoC machines, build with the shimmed toolkit:
`-DCMAKE_CUDA_COMPILER=$HOME/opt/cuda-shim/bin/nvcc -DCUDAToolkit_ROOT=$HOME/opt/cuda-shim`.

## Run

```bash
# 1. one-time export (needs torch + transformers; CPU is fine)
python tools/export_weights.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out data

# 2. correctness gate against HuggingFace
./build/kvcorrect --weights data/weights.bin --refs data/refs.bin

# 3. simulator (CPU only): fragmentation / capacity at any pool size
./build/kvsim --pool-gb 2 --block-size 16 --requests 2000            # runs both policies
./build/kvsim --policy paged --prefix --pool-gb 2 --shared-frac 0.5 --shared-len 512 --rate 4

# 4. serving benchmark on the GPU
./build/kvbench --policy paged --pool-gb 1 --requests 400 --rate 0   # offline burst
./build/kvbench --policy contiguous --pool-gb 1 --requests 300 --rate 16
./build/kvbench --policy paged --prefix --shared-frac 0.6 --shared-len 512 --rate 12

# 5. attention kernel microbenchmark
./build/kvkernel --batch 1,8,32,128 --ctx 128,512,2048

# 6. everything + plots
scripts/run_experiments.sh all && gnuplot -e "dir='results'" scripts/plot.gp
```

Common flags: `--block-size`, `--pool-gb` / `--num-blocks`, `--max-seq-len` (default 2048),
`--requests`, `--rate` (requests per step in `kvsim`, per second in `kvbench`; 0 = all at
once), `--prompt-mu/--prompt-sigma` and `--output-mu/--output-sigma` (lognormal lengths),
`--prompt-len/--output-len` (fixed lengths), `--lengths-csv` (rows of `prompt,output`),
`--shared-frac/--shared-len/--num-prefixes`, `--max-num-seqs`, `--max-batched-tokens`,
`--watermark` (default 1% of the pool), and `--attn v1|v2|auto`.

## Outputs

* `kvsim`: summary to stdout. `--summary` appends a CSV row; `--trace` writes per-step occupancy
  and fragmentation.
* `kvbench`: throughput, TTFT/E2E p50/p95/p99, TPOT, mean/max batch, preemptions, prefix hit
  rate, and peak KV and GPU memory. `--summary` appends to a CSV; `--step-log` writes a per-step
  trace.
* `kvkernel`: µs per call, achieved bandwidth, registers, smem, occupancy and waves.
