#include "engine/metrics.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace kvp {

double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  const double idx = p * (v.size() - 1);
  const size_t lo = size_t(std::floor(idx)), hi = size_t(std::ceil(idx));
  return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

void MetricsCollector::add_finished(const Sequence& s) {
  if (s.first_token < 0) { ++rejected_; return; }
  ttft_.push_back(s.first_token - s.arrival);
  e2e_.push_back(s.finished - s.arrival);
  const int out = s.num_generated();
  if (out > 1) tpot_.push_back((s.finished - s.first_token) / (out - 1));
  out_tokens_ += out;
  prompt_tokens_ += s.prompt_len;
}

void MetricsCollector::finalize(BenchSummary& o, double wall_s) const {
  auto mean = [](const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return v.empty() ? 0.0 : s / v.size();
  };
  o.completed = static_cast<int>(ttft_.size());
  o.rejected = rejected_;
  o.wall_s = wall_s;
  o.output_tokens = out_tokens_;
  o.prompt_tokens = prompt_tokens_;
  o.out_tok_per_s = out_tokens_ / wall_s;
  o.total_tok_per_s = (out_tokens_ + prompt_tokens_) / wall_s;
  o.req_per_s = o.completed / wall_s;
  o.ttft_mean = mean(ttft_);
  o.ttft_p50 = percentile(ttft_, 0.50);
  o.ttft_p95 = percentile(ttft_, 0.95);
  o.ttft_p99 = percentile(ttft_, 0.99);
  o.e2e_p50 = percentile(e2e_, 0.50);
  o.e2e_p95 = percentile(e2e_, 0.95);
  o.e2e_p99 = percentile(e2e_, 0.99);
  o.tpot_mean = mean(tpot_);
}

void print_summary(const BenchSummary& s) {
  std::printf("[%s] B=%d blocks=%d (%.2f GB) rate=%g shared=%g\n", s.policy.c_str(), s.block_size,
              s.num_blocks, s.pool_gb, s.rate, s.shared_frac);
  std::printf("  completed %d/%d (rejected %d) in %.2fs, %lld steps (%.2f ms/step)\n", s.completed,
              s.requests, s.rejected, s.wall_s, s.steps, s.mean_step_ms);
  std::printf("  throughput: %.1f out tok/s, %.1f total tok/s, %.2f req/s\n", s.out_tok_per_s,
              s.total_tok_per_s, s.req_per_s);
  std::printf("  TTFT p50/p95/p99 = %.3f / %.3f / %.3f s   E2E p50/p95/p99 = %.3f / %.3f / %.3f s\n",
              s.ttft_p50, s.ttft_p95, s.ttft_p99, s.e2e_p50, s.e2e_p95, s.e2e_p99);
  std::printf("  TPOT mean %.2f ms   batch mean %.1f max %d   preemptions %lld\n",
              1e3 * s.tpot_mean, s.mean_batch, s.max_batch, s.preemptions);
  std::printf("  prefill tokens %lld (saved %lld, block hit rate %.1f%%, evictions %lld)\n",
              s.prefill_tokens, s.prefill_saved, 100 * s.prefix_hit_rate, s.evictions);
  std::printf("  peak KV blocks %d (%.3f GB), GPU memory used %.2f GB\n", s.peak_blocks,
              s.peak_kv_gb, s.gpu_used_gb);
}

void append_bench_csv(const std::string& path, const BenchSummary& s) {
  bool fresh;
  {
    std::ifstream in(path);
    fresh = !in.good() || in.peek() == std::ifstream::traits_type::eof();
  }
  FILE* f = std::fopen(path.c_str(), "a");
  if (!f) throw std::runtime_error("cannot open " + path);
  if (fresh)
    std::fprintf(f,
                 "policy,block_size,num_blocks,pool_gb,rate,shared_frac,requests,completed,"
                 "rejected,wall_s,output_tokens,prompt_tokens,out_tok_s,total_tok_s,req_s,"
                 "ttft_mean,ttft_p50,ttft_p95,ttft_p99,e2e_p50,e2e_p95,e2e_p99,tpot_mean_ms,"
                 "mean_batch,max_batch,preemptions,steps,mean_step_ms,prefill_tokens,"
                 "prefill_saved,prefix_hit_rate,evictions,peak_blocks,peak_kv_gb,gpu_used_gb\n");
  std::fprintf(f,
               "%s,%d,%d,%.3f,%g,%g,%d,%d,%d,%.3f,%lld,%lld,%.2f,%.2f,%.3f,%.4f,%.4f,%.4f,%.4f,"
               "%.4f,%.4f,%.4f,%.3f,%.2f,%d,%lld,%lld,%.3f,%lld,%lld,%.4f,%lld,%d,%.4f,%.3f\n",
               s.policy.c_str(), s.block_size, s.num_blocks, s.pool_gb, s.rate, s.shared_frac,
               s.requests, s.completed, s.rejected, s.wall_s, s.output_tokens, s.prompt_tokens,
               s.out_tok_per_s, s.total_tok_per_s, s.req_per_s, s.ttft_mean, s.ttft_p50,
               s.ttft_p95, s.ttft_p99, s.e2e_p50, s.e2e_p95, s.e2e_p99, 1e3 * s.tpot_mean,
               s.mean_batch, s.max_batch, s.preemptions, s.steps, s.mean_step_ms,
               s.prefill_tokens, s.prefill_saved, s.prefix_hit_rate, s.evictions, s.peak_blocks,
               s.peak_kv_gb, s.gpu_used_gb);
  std::fclose(f);
}

}  // namespace kvp
