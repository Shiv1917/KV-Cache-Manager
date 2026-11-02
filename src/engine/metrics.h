// Benchmark metrics: throughput, latency percentiles, batch size, memory.
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "core/scheduler.h"
#include "core/sequence.h"

namespace kvp {

struct BenchSummary {
  std::string policy;
  int block_size = 0, num_blocks = 0;
  double pool_gb = 0;
  double rate = 0, shared_frac = 0;
  int requests = 0, completed = 0, rejected = 0;
  double wall_s = 0;
  long long output_tokens = 0, prompt_tokens = 0;
  double out_tok_per_s = 0, total_tok_per_s = 0, req_per_s = 0;
  double ttft_p50 = 0, ttft_p95 = 0, ttft_p99 = 0, ttft_mean = 0;
  double e2e_p50 = 0, e2e_p95 = 0, e2e_p99 = 0;
  double tpot_mean = 0;  // (e2e - ttft) / (output - 1), seconds per output token
  double mean_batch = 0;
  int max_batch = 0;
  long long preemptions = 0, steps = 0;
  long long prefill_tokens = 0, prefill_saved = 0;
  double prefix_hit_rate = 0;
  long long evictions = 0;
  int peak_blocks = 0;
  double peak_kv_gb = 0;       // peak blocks in use * bytes per block
  double gpu_used_gb = 0;      // cudaMemGetInfo (total - free) at the end of the run
  double mean_step_ms = 0;
};

class MetricsCollector {
 public:
  void add_finished(const Sequence& s);
  void finalize(BenchSummary& out, double wall_s) const;
  size_t count() const { return ttft_.size(); }

 private:
  std::vector<double> ttft_, e2e_, tpot_;
  long long out_tokens_ = 0, prompt_tokens_ = 0;
  int rejected_ = 0;
};

double percentile(std::vector<double> v, double p);
void print_summary(const BenchSummary& s);
void append_bench_csv(const std::string& path, const BenchSummary& s);

}  // namespace kvp
