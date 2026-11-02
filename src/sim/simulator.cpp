#include "sim/simulator.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>

namespace kvp {

static double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  const double idx = p * (v.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(idx)), hi = static_cast<size_t>(std::ceil(idx));
  return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

SimResult simulate(const SimConfig& cfg) {
  auto kv = make_kv_manager(cfg.kv);
  Scheduler sched(*kv, cfg.sched);
  std::vector<Request> reqs = generate_workload(cfg.wl);
  std::mt19937_64 rng(cfg.wl.seed ^ 0x5151);
  std::uniform_int_distribution<int32_t> tok(3, cfg.wl.vocab - 1);

  FILE* trace = nullptr;
  if (!cfg.trace_csv.empty()) {
    trace = std::fopen(cfg.trace_csv.c_str(), "w");
    if (!trace) throw std::runtime_error("cannot open " + cfg.trace_csv);
    std::fprintf(trace,
                 "step,running,waiting,blocks_in_use,blocks_cached_free,allocated_slots,"
                 "used_slots,frag_pct,tail_waste_slots,reserved_unused_slots,table_entries,"
                 "preemptions,completed\n");
  }

  SimResult r;
  size_t next = 0;
  double t = 0;
  long long frag_steps = 0;
  std::vector<double> ttft, e2e;
  std::vector<int32_t> sampled;

  while (r.steps < cfg.max_steps) {
    if (sched.idle()) {
      if (next >= reqs.size()) break;
      t = std::max(t, std::ceil(reqs[next].arrival));  // fast-forward over idle time
    }
    while (next < reqs.size() && reqs[next].arrival <= t) sched.add(make_sequence(reqs[next++], cfg.wl));

    const auto& plan = sched.schedule(t);
    sampled.resize(plan.size());
    for (auto& x : sampled) x = tok(rng);

    // Snapshot after allocation for this step (that is the peak memory of the step).
    KVStats st = kv->stats(sched.live());
    r.peak_blocks = std::max(r.peak_blocks, st.blocks_in_use);
    if (st.allocated_slots > 0) {
      r.mean_frag += st.fragmentation();
      r.mean_tail_waste += double(st.tail_waste_slots) / st.allocated_slots;
      r.mean_reserved_unused += double(st.reserved_unused_slots) / st.allocated_slots;
      r.mean_util += double(st.used_slots) / (double(st.num_blocks) * st.block_size);
      r.mean_table_entries += double(st.block_table_entries);
      ++frag_steps;
    }
    if (trace && r.steps % cfg.trace_every == 0) {
      std::fprintf(trace, "%lld,%zu,%zu,%d,%d,%lld,%lld,%.4f,%lld,%lld,%lld,%lld,%d\n", r.steps,
                   sched.num_running(), sched.num_waiting(), st.blocks_in_use,
                   st.blocks_cached_free, (long long)st.allocated_slots, (long long)st.used_slots,
                   100.0 * st.fragmentation(), (long long)st.tail_waste_slots,
                   (long long)st.reserved_unused_slots, (long long)st.block_table_entries,
                   (long long)sched.counters().preemptions, r.completed);
    }

    t += 1;
    sched.update(sampled, t);
    for (auto& s : sched.take_finished()) {
      if (s->first_token < 0) { ++r.rejected; continue; }
      ++r.completed;
      ttft.push_back(s->first_token - s->arrival);
      e2e.push_back(s->finished - s->arrival);
    }
    ++r.steps;
  }
  if (trace) std::fclose(trace);

  const auto& c = sched.counters();
  r.max_running = c.max_running;
  r.mean_running = c.steps ? double(c.batch_size_sum) / c.steps : 0;
  r.preemptions = c.preemptions;
  r.admission_blocked_steps = c.admission_blocked_steps;
  if (frag_steps) {
    r.mean_frag /= frag_steps;
    r.mean_tail_waste /= frag_steps;
    r.mean_reserved_unused /= frag_steps;
    r.mean_util /= frag_steps;
    r.mean_table_entries /= frag_steps;
  }
  auto ps = kv->prefix_stats();
  r.prefix_hit_rate = ps.queried_blocks ? double(ps.hit_blocks) / ps.queried_blocks : 0;
  r.evictions = ps.evictions;
  r.prefill_tokens = c.prefill_tokens;
  r.prefill_saved = c.prefill_tokens_saved;
  r.decode_tokens = c.decode_tokens;
  if (!ttft.empty()) {
    double s = 0;
    for (double x : ttft) s += x;
    r.mean_ttft = s / ttft.size();
    r.p99_ttft = percentile(ttft, 0.99);
    s = 0;
    for (double x : e2e) s += x;
    r.mean_e2e = s / e2e.size();
  }
  return r;
}

void print_result(const SimConfig& cfg, const SimResult& r) {
  std::printf("policy=%s%s block_size=%d pool_blocks=%d (%lld slots) max_seq_len=%d\n",
              cfg.kv.policy.c_str(), cfg.kv.prefix_caching ? "+prefix" : "", cfg.kv.block_size,
              cfg.kv.num_blocks, (long long)cfg.kv.num_blocks * cfg.kv.block_size,
              cfg.kv.max_seq_len);
  std::printf("workload: %s\n", cfg.wl.describe().c_str());
  std::printf("  steps=%lld completed=%d rejected=%d\n", r.steps, r.completed, r.rejected);
  std::printf("  max_running=%d mean_running=%.1f preemptions=%lld blocked_steps=%lld\n",
              r.max_running, r.mean_running, r.preemptions, r.admission_blocked_steps);
  std::printf("  frag=%.2f%% (tail %.2f%%, reserved-unused %.2f%%) pool_util=%.1f%% peak_blocks=%d\n",
              100 * r.mean_frag, 100 * r.mean_tail_waste, 100 * r.mean_reserved_unused,
              100 * r.mean_util, r.peak_blocks);
  std::printf("  mean_block_table_entries=%.0f\n", r.mean_table_entries);
  if (cfg.kv.prefix_caching)
    std::printf("  prefix: hit_rate=%.1f%% prefill_tokens=%lld saved=%lld evictions=%lld\n",
                100 * r.prefix_hit_rate, r.prefill_tokens, r.prefill_saved, r.evictions);
  std::printf("  ttft mean=%.1f p99=%.1f steps, e2e mean=%.1f steps\n", r.mean_ttft, r.p99_ttft,
              r.mean_e2e);
}

void append_summary_csv(const std::string& path, const SimConfig& c, const SimResult& r) {
  bool fresh;
  {
    std::ifstream in(path);
    fresh = !in.good() || in.peek() == std::ifstream::traits_type::eof();
  }
  FILE* f = std::fopen(path.c_str(), "a");
  if (!f) throw std::runtime_error("cannot open " + path);
  if (fresh)
    std::fprintf(f,
                 "policy,prefix,block_size,num_blocks,max_seq_len,requests,rate,shared_frac,"
                 "shared_len,steps,completed,rejected,max_running,mean_running,preemptions,"
                 "blocked_steps,frag_pct,tail_waste_pct,reserved_unused_pct,pool_util_pct,"
                 "peak_blocks,mean_table_entries,prefix_hit_pct,prefill_tokens,prefill_saved,"
                 "evictions,mean_ttft,p99_ttft,mean_e2e\n");
  std::fprintf(f,
               "%s,%d,%d,%d,%d,%d,%g,%g,%d,%lld,%d,%d,%d,%.2f,%lld,%lld,%.3f,%.3f,%.3f,%.3f,%d,"
               "%.1f,%.3f,%lld,%lld,%lld,%.2f,%.2f,%.2f\n",
               c.kv.policy.c_str(), c.kv.prefix_caching ? 1 : 0, c.kv.block_size, c.kv.num_blocks,
               c.kv.max_seq_len, c.wl.num_requests, c.wl.arrival_rate, c.wl.shared_frac,
               c.wl.shared_len, r.steps, r.completed, r.rejected, r.max_running, r.mean_running,
               r.preemptions, r.admission_blocked_steps, 100 * r.mean_frag,
               100 * r.mean_tail_waste, 100 * r.mean_reserved_unused, 100 * r.mean_util,
               r.peak_blocks, r.mean_table_entries, 100 * r.prefix_hit_rate, r.prefill_tokens,
               r.prefill_saved, r.evictions, r.mean_ttft, r.p99_ttft, r.mean_e2e);
  std::fclose(f);
}

}  // namespace kvp
