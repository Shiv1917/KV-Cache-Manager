#include "core/scheduler.h"

#include <algorithm>

namespace kvp {

void Scheduler::add(std::unique_ptr<Sequence> seq) { waiting_.push_back(std::move(seq)); }

std::vector<const Sequence*> Scheduler::live() const {
  std::vector<const Sequence*> v;
  v.reserve(running_.size());
  for (const auto& s : running_) v.push_back(s.get());
  return v;
}

void Scheduler::preempt_youngest() {
  std::unique_ptr<Sequence> victim = std::move(running_.back());
  running_.pop_back();
  kv_.release(*victim);  // recompute later: KV dropped, tokens kept
  ++victim->num_preemptions;
  ++ctr_.preemptions;
  waiting_.push_front(std::move(victim));
}

const std::vector<ScheduledSeq>& Scheduler::schedule(double now) {
  plan_.clear();

  // 1) Grow every running sequence by one slot, oldest first. When the pool is exhausted, evict
  //    the youngest running sequence (possibly the one asking) and retry.
  for (size_t i = 0; i < running_.size();) {
    if (kv_.append_slot(*running_[i])) {
      ++i;
    } else if (i + 1 < running_.size()) {
      preempt_youngest();
    } else {
      preempt_youngest();  // the requester itself is the youngest
      break;
    }
  }
  int budget = cfg_.max_batched_tokens;
  for (auto& s : running_) {
    const int n = static_cast<int>(s->tokens.size()) - s->num_computed;
    plan_.push_back({s.get(), s->num_computed, n, false});
    budget -= n;
    ctr_.decode_tokens += n;
  }

  // 2) Admit waiting sequences in FCFS order while they fit.
  bool blocked = false;
  while (!waiting_.empty() && static_cast<int>(running_.size()) < cfg_.max_num_seqs) {
    Sequence& head = *waiting_.front();
    const int len = static_cast<int>(head.tokens.size());
    if (len > budget && !plan_.empty()) break;  // token budget for this step is used up
    if (!kv_.can_admit(head)) {
      if (running_.empty()) {
        // Can never fit (prompt larger than the whole pool / max_seq_len): reject it.
        head.finished = now;
        ++ctr_.rejected;
        finished_.push_back(std::move(waiting_.front()));
        waiting_.pop_front();
        continue;
      }
      blocked = true;
      break;
    }
    kv_.admit(head);
    if (head.admitted < 0) head.admitted = now;
    const int n = len - head.num_computed;
    plan_.push_back({&head, head.num_computed, n, true});
    budget -= n;
    ctr_.prefill_tokens += n;
    ctr_.prefill_tokens_saved += head.cached_tokens;
    head.prefill_tokens_computed += n;
    ++ctr_.admissions;
    running_.push_back(std::move(waiting_.front()));
    waiting_.pop_front();
  }
  if (blocked) ++ctr_.admission_blocked_steps;

  ++ctr_.steps;
  ctr_.batch_size_sum += static_cast<int64_t>(running_.size());
  ctr_.max_running = std::max(ctr_.max_running, static_cast<int>(running_.size()));
  return plan_;
}

void Scheduler::update(const std::vector<int32_t>& next_tokens, double now) {
  for (size_t i = 0; i < plan_.size(); ++i) {
    Sequence& s = *plan_[i].seq;
    s.num_computed = plan_[i].start + plan_[i].num_tokens;
    kv_.commit(s);
    s.tokens.push_back(next_tokens[i]);
    if (s.first_token < 0) s.first_token = now;
    if (s.done() || static_cast<int>(s.tokens.size()) >= kv_.max_seq_len()) s.finished = now;
  }
  for (auto it = running_.begin(); it != running_.end();) {
    if ((*it)->finished >= 0) {
      kv_.release(**it);
      finished_.push_back(std::move(*it));
      it = running_.erase(it);
    } else {
      ++it;
    }
  }
  plan_.clear();
}

std::vector<std::unique_ptr<Sequence>> Scheduler::take_finished() {
  std::vector<std::unique_ptr<Sequence>> out;
  out.swap(finished_);
  return out;
}

}  // namespace kvp
