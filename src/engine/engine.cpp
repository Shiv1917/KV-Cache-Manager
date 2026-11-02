#include "engine/engine.h"

#include <algorithm>

namespace kvp {

namespace {

KVCacheConfig cache_config(const LlamaConfig& c, int num_blocks, int block_size) {
  KVCacheConfig k;
  k.num_layers = c.layers;
  k.num_blocks = num_blocks;
  k.block_size = block_size;
  k.num_kv_heads = c.kv_heads;
  k.head_dim = c.head_dim;
  return k;
}

KVManagerConfig resolve(const LlamaWeights& w, const EngineConfig& cfg) {
  KVManagerConfig k = cfg.kv;
  if (k.num_blocks <= 0) {
    const size_t bpb = cache_config(w.config(), 1, k.block_size).bytes_per_block();
    k.num_blocks = static_cast<int>(cfg.pool_gb * double(1ULL << 30) / double(bpb));
  }
  if (k.policy == "contiguous") k.prefix_caching = false;
  return k;
}

}  // namespace

Engine::Engine(const LlamaWeights& w, const EngineConfig& cfg)
    : w_(w),
      kvcfg_(resolve(w, cfg)),
      kv_(make_kv_manager(kvcfg_)),
      cache_(std::make_unique<KVCache>(cache_config(w.config(), kvcfg_.num_blocks,
                                                    kvcfg_.block_size))),
      sched_(*kv_, cfg.sched) {
  ModelLimits lim;
  lim.max_tokens = std::max(cfg.sched.max_batched_tokens + cfg.sched.max_num_seqs,
                            kvcfg_.max_seq_len);
  lim.max_seqs = cfg.sched.max_num_seqs;
  lim.max_logit_rows = cfg.max_logit_rows > 0 ? cfg.max_logit_rows
                                              : std::max(cfg.sched.max_num_seqs, 512);
  lim.max_blocks_per_seq = (kvcfg_.max_seq_len + kvcfg_.block_size - 1) / kvcfg_.block_size;
  lim.max_context_len = kvcfg_.max_seq_len;
  model_ = std::make_unique<LlamaModel>(w, *cache_, lim);
  model_->set_attention_version(cfg.attn);
}

void Engine::build_batch(const std::vector<ScheduledSeq>& plan) {
  batch_.clear();
  const int B = kvcfg_.block_size;
  batch_.max_blocks = (kvcfg_.max_seq_len + B - 1) / B;
  batch_.num_seqs = static_cast<int>(plan.size());
  batch_.block_tables.assign(size_t(batch_.num_seqs) * batch_.max_blocks, 0);
  for (int i = 0; i < batch_.num_seqs; ++i) {
    const Sequence& s = *plan[i].seq;
    const auto& table = s.block_table;
    std::copy(table.begin(), table.end(), batch_.block_tables.begin() + size_t(i) * batch_.max_blocks);
    for (int j = 0; j < plan[i].num_tokens; ++j) {
      const int pos = plan[i].start + j;
      batch_.tokens.push_back(s.tokens[pos]);
      batch_.positions.push_back(pos);
      batch_.slot_mapping.push_back(table[pos / B] * B + pos % B);
      batch_.row_seq.push_back(i);
      batch_.context_lens.push_back(pos + 1);
    }
    batch_.logit_rows.push_back(batch_.num_rows() - 1);
    batch_.max_context_len = std::max(batch_.max_context_len, plan[i].start + plan[i].num_tokens);
  }
}

StepInfo Engine::step() {
  StepInfo info;
  const double t_start = now();
  const auto& plan = sched_.schedule(t_start);
  if (plan.empty()) return info;
  build_batch(plan);
  for (const auto& p : plan) {
    if (p.is_prefill) info.prefill_tokens += p.num_tokens;
    else ++info.decode_seqs;
  }
  model_->forward(batch_, next_, nullptr);
  const double t_end = now();
  sched_.update(next_, t_end);
  info.rows = batch_.num_rows();
  info.seqs = batch_.num_seqs;
  info.ms = (t_end - t_start) * 1e3;
  return info;
}

void Engine::score(const std::vector<int32_t>& tokens, std::vector<float>& logits) {
  // Recomputing every position would rewrite shared (read-only) prefix blocks.
  if (kvcfg_.prefix_caching) throw std::runtime_error("score: disable prefix caching");
  Sequence s;
  s.id = -1;
  s.tokens = tokens;
  s.prompt_len = static_cast<int>(tokens.size());
  if (!kv_->can_admit(s)) throw std::runtime_error("score: sequence does not fit in KV pool");
  kv_->admit(s);
  // Recompute every position (ignore prefix hits) so all rows produce logits.
  ScheduledSeq ss{&s, 0, static_cast<int>(tokens.size()), true};
  build_batch({ss});
  batch_.logit_rows.clear();
  for (int r = 0; r < batch_.num_rows(); ++r) batch_.logit_rows.push_back(r);
  model_->forward(batch_, next_, &logits);
  kv_->release(s);
}

}  // namespace kvp
