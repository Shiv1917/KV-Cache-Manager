// Continuous-batching scheduler (FCFS, preemption by recomputation). CPU-only; shared by the
// simulator (which "executes" a step by doing nothing) and the GPU engine.
#pragma once
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#include "core/kv_manager.h"
#include "core/sequence.h"

namespace kvp {

struct SchedulerConfig {
  int max_num_seqs = 256;          // cap on the running batch
  int max_batched_tokens = 4096;   // cap on tokens computed per step (prefill + decode)
  // Admission stops at the first request that does not fit (strict FCFS, no reordering).
};

// One sequence's share of a forward step: compute tokens [start, start + num_tokens).
struct ScheduledSeq {
  Sequence* seq;
  int start;
  int num_tokens;
  bool is_prefill;
};

struct SchedulerCounters {
  int64_t steps = 0;
  int64_t preemptions = 0;
  int64_t admissions = 0;
  int64_t admission_blocked_steps = 0;  // steps where the queue head could not be admitted
  int64_t rejected = 0;                 // requests that can never fit (dropped)
  int64_t prefill_tokens = 0;
  int64_t decode_tokens = 0;
  int64_t prefill_tokens_saved = 0;
  int64_t batch_size_sum = 0;           // sum of running batch sizes (for mean)
  int max_running = 0;
};

class Scheduler {
 public:
  Scheduler(KVManager& kv, SchedulerConfig cfg) : kv_(kv), cfg_(cfg) {}

  void add(std::unique_ptr<Sequence> seq);
  // Plans one step. `now` is recorded as the admission time of newly admitted sequences.
  const std::vector<ScheduledSeq>& schedule(double now);
  // Applies sampled tokens (one per scheduled sequence, same order) and retires finished ones.
  void update(const std::vector<int32_t>& next_tokens, double now);
  std::vector<std::unique_ptr<Sequence>> take_finished();

  bool idle() const { return waiting_.empty() && running_.empty(); }
  size_t num_waiting() const { return waiting_.size(); }
  size_t num_running() const { return running_.size(); }
  const std::deque<std::unique_ptr<Sequence>>& running() const { return running_; }
  std::vector<const Sequence*> live() const;
  const SchedulerCounters& counters() const { return ctr_; }
  void reset_counters() { ctr_ = {}; }
  KVManager& kv() { return kv_; }

 private:
  void preempt_youngest();

  KVManager& kv_;
  SchedulerConfig cfg_;
  std::deque<std::unique_ptr<Sequence>> waiting_;
  std::deque<std::unique_ptr<Sequence>> running_;  // admission order: back = youngest
  std::vector<std::unique_ptr<Sequence>> finished_;
  std::vector<ScheduledSeq> plan_;
  SchedulerCounters ctr_;
};

}  // namespace kvp
