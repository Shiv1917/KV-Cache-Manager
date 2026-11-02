// kvbench: end-to-end serving benchmark on the GPU with real TinyLlama weights.
//
//   kvbench --weights data/weights.bin --policy paged --pool-gb 1 --requests 500 --rate 8 \
//           --summary results/bench_rate.csv
// Requests arrive by a Poisson process in wall-clock time (rate = req/s, 0 = all at once).
// Token contents are random: memory behaviour does not depend on them, except for the shared
// system-prompt prefix used by the prefix-caching experiment.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <thread>

#include "common/args.h"
#include "engine/engine.h"
#include "engine/metrics.h"
#include "sim/workload.h"

using namespace kvp;

int main(int argc, char** argv) {
  try {
    Args a(argc, argv);
    const std::string weights_path = a.str("weights", "data/weights.bin");
    EngineConfig ec;
    ec.kv.policy = a.str("policy", "paged");
    ec.kv.block_size = a.i32("block-size", 16);
    ec.kv.max_seq_len = a.i32("max-seq-len", 2048);
    ec.kv.prefix_caching = a.flag("prefix");
    ec.kv.num_blocks = a.i32("num-blocks", 0);
    ec.kv.watermark_blocks = a.i32("watermark", -1);
    ec.pool_gb = a.f64("pool-gb", 1.0);
    ec.sched.max_num_seqs = a.i32("max-num-seqs", 256);
    ec.sched.max_batched_tokens = a.i32("max-batched-tokens", 4096);
    const std::string attn = a.str("attn", "auto");
    ec.attn = attn == "v1" ? AttnVersion::V1 : attn == "v2" ? AttnVersion::V2 : AttnVersion::Auto;
    WorkloadConfig wl = WorkloadConfig::from_args(a);
    wl.max_seq_len = ec.kv.max_seq_len;
    const int warmup = a.i32("warmup", 8);
    const std::string summary_csv = a.str("summary", "");
    const std::string step_log = a.str("step-log", "");
    const double time_limit = a.f64("time-limit", 1e9);
    a.warn_unused();

    std::printf("loading %s ...\n", weights_path.c_str());
    LlamaWeights weights(weights_path);
    Engine eng(weights, ec);
    std::printf("weights %.2f GB, KV pool %d blocks x %zu B = %.2f GB, workspace %.2f GB\n",
                weights.bytes() / double(1ULL << 30), eng.num_blocks(),
                eng.bytes_per_block(), eng.cache().total_bytes() / double(1ULL << 30),
                eng.model().workspace_bytes() / double(1ULL << 30));

    // Warm-up: a few short requests (cuBLAS heuristics, kernel attributes, allocator paths).
    {
      WorkloadConfig w = wl;
      w.num_requests = warmup; w.arrival_rate = 0; w.shared_frac = 0; w.seed = wl.seed + 999;
      w.min_output = w.max_output = 16; w.output_sigma = 0; w.output_mu = std::log(16.0);
      for (const auto& r : generate_workload(w)) eng.add(make_sequence(r, w));
      while (!eng.idle()) eng.step();
      eng.take_finished();
      eng.scheduler().reset_counters();
    }
    const PrefixCacheStats ps0 = eng.kv().prefix_stats();

    std::vector<Request> reqs = generate_workload(wl);
    FILE* log = step_log.empty() ? nullptr : std::fopen(step_log.c_str(), "w");
    if (log) std::fprintf(log, "t,rows,seqs,prefill_tokens,decode_seqs,ms,running,waiting,blocks_in_use\n");

    MetricsCollector mc;
    BenchSummary s;
    size_t next = 0;
    double step_ms_sum = 0;
    eng.reset_clock();
    while (mc.count() + s.rejected < reqs.size()) {
      const double now = eng.now();
      if (now > time_limit) { std::printf("time limit reached\n"); break; }
      while (next < reqs.size() && reqs[next].arrival <= now)
        eng.add(make_sequence(reqs[next++], wl));
      if (eng.idle()) {
        if (next < reqs.size()) {
          const double wait = reqs[next].arrival - eng.now();
          if (wait > 0) std::this_thread::sleep_for(std::chrono::duration<double>(wait));
        }
        continue;
      }
      StepInfo si = eng.step();
      step_ms_sum += si.ms;
      s.peak_blocks = std::max(s.peak_blocks, eng.kv().blocks_in_use());
      if (log)
        std::fprintf(log, "%.4f,%d,%d,%d,%d,%.3f,%zu,%zu,%d\n", eng.now(), si.rows, si.seqs,
                     si.prefill_tokens, si.decode_seqs, si.ms, eng.scheduler().num_running(),
                     eng.scheduler().num_waiting(), eng.kv().blocks_in_use());
      for (auto& f : eng.take_finished()) {
        if (f->first_token < 0) ++s.rejected;
        mc.add_finished(*f);
      }
    }
    const double wall = eng.now();
    if (log) std::fclose(log);

    const auto& c = eng.scheduler().counters();
    s.policy = eng.kv().name();
    s.block_size = ec.kv.block_size;
    s.num_blocks = eng.num_blocks();
    s.pool_gb = eng.cache().total_bytes() / double(1ULL << 30);
    s.rate = wl.arrival_rate;
    s.shared_frac = wl.shared_frac;
    s.requests = static_cast<int>(reqs.size());
    mc.finalize(s, wall);
    s.mean_batch = c.steps ? double(c.batch_size_sum) / c.steps : 0;
    s.max_batch = c.max_running;
    s.preemptions = c.preemptions;
    s.steps = c.steps;
    s.mean_step_ms = c.steps ? step_ms_sum / c.steps : 0;
    s.prefill_tokens = c.prefill_tokens;
    s.prefill_saved = c.prefill_tokens_saved;
    const PrefixCacheStats ps = eng.kv().prefix_stats();
    const long long q = ps.queried_blocks - ps0.queried_blocks;
    s.prefix_hit_rate = q ? double(ps.hit_blocks - ps0.hit_blocks) / q : 0;
    s.evictions = ps.evictions - ps0.evictions;
    s.peak_kv_gb = s.peak_blocks * double(eng.bytes_per_block()) / double(1ULL << 30);
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    s.gpu_used_gb = (total_b - free_b) / double(1ULL << 30);
    print_summary(s);
    if (!summary_csv.empty()) append_bench_csv(summary_csv, s);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
