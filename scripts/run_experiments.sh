#!/usr/bin/env bash
# Runs every experiment from the plan and writes CSVs into $OUT.
#   BUILD=build DATA=data OUT=results scripts/run_experiments.sh [sim|kernel|gpu|all]
# GPU experiments assume ~4 GB of free device memory (pools up to 2 GB + 2 GB of weights).
set -euo pipefail
BUILD=${BUILD:-build}
DATA=${DATA:-data}
OUT=${OUT:-results}
WHAT=${1:-all}
W="$DATA/weights.bin"
mkdir -p "$OUT"

run_sim() {
  echo "== sim: fragmentation vs block size (2 GB pool, 2000-request burst)"
  rm -f "$OUT/sim_blocksize.csv"
  for b in 8 16 32 64; do
    "$BUILD/kvsim" --block-size $b --pool-gb 2 --requests 2000 --summary "$OUT/sim_blocksize.csv" >/dev/null
  done
  echo "== sim: capacity vs pool size"
  rm -f "$OUT/sim_capacity.csv"
  for g in 1 2 4 8; do
    "$BUILD/kvsim" --pool-gb $g --requests 6000 --max-num-seqs 4096 --max-batched-tokens 65536 \
      --summary "$OUT/sim_capacity.csv" >/dev/null
  done
  echo "== sim: prefix sharing sweep (512-token system prompt)"
  rm -f "$OUT/sim_prefix.csv"
  for f in 0 0.1 0.3 0.5 0.7 0.9; do
    for pc in 0 1; do
      flag=""; [ $pc = 1 ] && flag="--prefix"
      "$BUILD/kvsim" --policy paged $flag --pool-gb 2 --requests 2000 --rate 4 --shared-frac $f \
        --shared-len 512 --summary "$OUT/sim_prefix.csv" >/dev/null
    done
  done
  echo "== sim: per-step traces (1 GB pool, burst)"
  "$BUILD/kvsim" --pool-gb 1 --requests 1000 --trace "$OUT/sim_trace.csv" >/dev/null
}

run_kernel() {
  echo "== kernel: decode attention, paged vs contiguous layout"
  "$BUILD/kvkernel" --batch 1,8,32,128 --ctx 128,256,512,1024,2048 --block-size 16 \
    --out "$OUT/kernel_attn.csv" >/dev/null
  "$BUILD/kvkernel" --batch 32 --ctx 512,2048 --block-size 8,16,32,64 \
    --out "$OUT/kernel_blocksize.csv" >/dev/null
}

run_gpu() {
  echo "== gpu: capacity / throughput vs pool size (burst of 400)"
  rm -f "$OUT/bench_capacity.csv"
  for g in 0.5 1 2; do
    for p in paged contiguous; do
      "$BUILD/kvbench" --weights "$W" --policy $p --pool-gb $g --requests 400 --rate 0 \
        --max-num-seqs 512 --summary "$OUT/bench_capacity.csv" | tail -7
    done
  done
  echo "== gpu: block size (paged, 1 GB, burst of 400)"
  rm -f "$OUT/bench_blocksize.csv"
  for b in 8 16 32 64; do
    "$BUILD/kvbench" --weights "$W" --policy paged --block-size $b --pool-gb 1 --requests 400 \
      --rate 0 --summary "$OUT/bench_blocksize.csv" | tail -7
  done
  echo "== gpu: throughput / latency vs arrival rate (1 GB)"
  rm -f "$OUT/bench_rate.csv"
  for r in 4 8 12 16 24 32; do
    for p in paged contiguous; do
      "$BUILD/kvbench" --weights "$W" --policy $p --pool-gb 1 --requests 300 --rate $r \
        --summary "$OUT/bench_rate.csv" | tail -7
    done
  done
  echo "== gpu: prefix caching vs shared-prefix fraction (1 GB, 12 req/s, 512-token prefix)"
  rm -f "$OUT/bench_prefix.csv"
  for f in 0 0.3 0.6 0.9; do
    for pc in 0 1; do
      flag=""; [ $pc = 1 ] && flag="--prefix"
      "$BUILD/kvbench" --weights "$W" --policy paged $flag --pool-gb 1 --requests 300 --rate 12 \
        --shared-frac $f --shared-len 512 --summary "$OUT/bench_prefix.csv" | tail -7
    done
  done
  echo "== gpu: step traces (1 GB, burst of 300)"
  for p in paged contiguous; do
    "$BUILD/kvbench" --weights "$W" --policy $p --pool-gb 1 --requests 300 --rate 0 \
      --step-log "$OUT/bench_trace_$p.csv" | tail -7
  done
}

case "$WHAT" in
  sim) run_sim ;;
  kernel) run_kernel ;;
  gpu) run_gpu ;;
  all) run_sim; run_kernel; run_gpu ;;
  *) echo "usage: $0 [sim|kernel|gpu|all]"; exit 1 ;;
esac
echo "done -> $OUT"
