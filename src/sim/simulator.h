// CPU-only, time-stepped simulator: one step == one engine iteration (one forward pass).
#pragma once
#include <string>

#include "core/kv_manager.h"
#include "core/scheduler.h"
#include "sim/workload.h"

namespace kvp {

struct SimConfig {
  KVManagerConfig kv;
  SchedulerConfig sched;
  WorkloadConfig wl;
  std::string trace_csv;   // per-step rows ("" = none)
  int trace_every = 1;
  long long max_steps = 10'000'000;
};

struct SimResult {
  long long steps = 0;
  int completed = 0;
  int rejected = 0;
  int max_running = 0;
  double mean_running = 0;
  long long preemptions = 0;
  long long admission_blocked_steps = 0;
  double mean_frag = 0;         // mean over steps of (allocated - used) / allocated
  double mean_tail_waste = 0;   // share of allocated slots wasted in partial tail blocks
  double mean_reserved_unused = 0;  // share of allocated slots in empty reserved blocks
  double mean_util = 0;         // used_slots / pool slots
  double mean_table_entries = 0;
  int peak_blocks = 0;
  double prefix_hit_rate = 0;   // hit blocks / queried blocks
  long long prefill_tokens = 0, prefill_saved = 0, decode_tokens = 0;
  long long evictions = 0;
  double mean_ttft = 0, p99_ttft = 0, mean_e2e = 0;  // in steps
};

SimResult simulate(const SimConfig& cfg);
void print_result(const SimConfig& cfg, const SimResult& r);
// Appends one summary row (with header if the file is new/empty).
void append_summary_csv(const std::string& path, const SimConfig& cfg, const SimResult& r);

}  // namespace kvp
