// Mini LLM serving engine: scheduler + KV manager (policy under test) + GPU model.
#pragma once
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "core/kv_manager.h"
#include "core/scheduler.h"
#include "cuda/kv_cache.h"
#include "model/llama.h"
#include "model/weights.h"

namespace kvp {

struct EngineConfig {
  KVManagerConfig kv;            // kv.num_blocks <= 0 => derived from pool_gb
  double pool_gb = 1.0;
  SchedulerConfig sched;
  AttnVersion attn = AttnVersion::Auto;
  int max_logit_rows = 0;        // 0 => max(sched.max_num_seqs, 512)
};

struct StepInfo {
  int rows = 0, seqs = 0, prefill_tokens = 0, decode_seqs = 0;
  double ms = 0;
};

class Engine {
 public:
  Engine(const LlamaWeights& w, const EngineConfig& cfg);

  void add(std::unique_ptr<Sequence> s) { sched_.add(std::move(s)); }
  StepInfo step();
  std::vector<std::unique_ptr<Sequence>> take_finished() { return sched_.take_finished(); }
  bool idle() const { return sched_.idle(); }

  // Teacher-forced scoring: fp32 logits for every position of `tokens` ([n, vocab]). Uses (and
  // releases) KV blocks from the same manager, so it exercises the real paged path.
  void score(const std::vector<int32_t>& tokens, std::vector<float>& logits);

  double now() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
  }
  void reset_clock() { t0_ = std::chrono::steady_clock::now(); }

  Scheduler& scheduler() { return sched_; }
  KVManager& kv() { return *kv_; }
  LlamaModel& model() { return *model_; }
  const KVCache& cache() const { return *cache_; }
  int num_blocks() const { return kvcfg_.num_blocks; }
  size_t bytes_per_block() const { return cache_->config().bytes_per_block(); }

 private:
  void build_batch(const std::vector<ScheduledSeq>& plan);

  const LlamaWeights& w_;
  KVManagerConfig kvcfg_;
  std::unique_ptr<KVManager> kv_;
  std::unique_ptr<KVCache> cache_;
  std::unique_ptr<LlamaModel> model_;
  Scheduler sched_;
  ForwardBatch batch_;
  std::vector<int32_t> next_;
  std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now();
};

}  // namespace kvp
