#include <gtest/gtest.h>

#include "core/paged_manager.h"
#include "core/scheduler.h"
#include "sim/simulator.h"

using namespace kvp;

static std::unique_ptr<Sequence> mk(int id, int prompt, int out) {
  auto s = std::make_unique<Sequence>();
  s->id = id;
  for (int i = 0; i < prompt; ++i) s->tokens.push_back(10 + (id * 131 + i) % 1000);
  s->prompt_len = prompt;
  s->max_new_tokens = out;
  return s;
}

// Drive the scheduler like the engine would, with a fake model.
static int run_to_completion(Scheduler& sch, int max_steps = 100000) {
  int steps = 0, done = 0;
  std::vector<int32_t> toks;
  while (!sch.idle() && steps < max_steps) {
    const auto& plan = sch.schedule(steps);
    // Invariant: every scheduled sequence has slots for all tokens it computes.
    for (const auto& p : plan) {
      EXPECT_GE(static_cast<int>(p.seq->block_table.size()) * sch.kv().block_size(),
                p.start + p.num_tokens);
      EXPECT_EQ(p.start + p.num_tokens, static_cast<int>(p.seq->tokens.size()));
    }
    toks.assign(plan.size(), 42);
    sch.update(toks, steps + 1);
    for (auto& s : sch.take_finished()) {
      EXPECT_EQ(s->num_generated(), s->max_new_tokens);
      ++done;
    }
    ++steps;
  }
  return done;
}

TEST(Scheduler, CompletesAllWithoutPressure) {
  PagedManager kv(1000, 16, 1024, false);
  Scheduler sch(kv, {});
  for (int i = 0; i < 20; ++i) sch.add(mk(i, 30 + i, 10 + i));
  EXPECT_EQ(run_to_completion(sch), 20);
  EXPECT_EQ(sch.counters().preemptions, 0);
  EXPECT_EQ(kv.blocks_in_use(), 0);
}

TEST(Scheduler, PreemptsUnderPressureAndStillFinishes) {
  // 24 blocks of 16 = 384 slots; 8 sequences of 32 prompt + 64 output = 96 tokens each.
  PagedManager kv(24, 16, 1024, false);
  Scheduler sch(kv, {});
  for (int i = 0; i < 8; ++i) sch.add(mk(i, 32, 64));
  EXPECT_EQ(run_to_completion(sch), 8);
  EXPECT_GT(sch.counters().preemptions, 0);
  EXPECT_EQ(kv.blocks_in_use(), 0);
}

TEST(Scheduler, PrefixCachingSkipsPrefill) {
  PagedManager kv(1000, 16, 1024, true);
  Scheduler sch(kv, {});
  sch.add(mk(0, 100, 5));
  run_to_completion(sch);
  sch.add(mk(0, 100, 5));  // identical prompt
  const auto& plan = sch.schedule(0);
  ASSERT_EQ(plan.size(), 1u);
  EXPECT_EQ(plan[0].start, 96);     // 6 full blocks cached
  EXPECT_EQ(plan[0].num_tokens, 4);
}

TEST(Scheduler, RejectsImpossibleRequests) {
  PagedManager kv(4, 16, 1024, false);  // 64 slots total
  Scheduler sch(kv, {});
  sch.add(mk(0, 200, 5));
  sch.add(mk(1, 10, 5));
  int steps = 0;
  int finished = 0;
  std::vector<int32_t> toks;
  while (!sch.idle() && steps++ < 100) {
    toks.assign(sch.schedule(steps).size(), 1);
    sch.update(toks, steps);
    finished += static_cast<int>(sch.take_finished().size());
  }
  EXPECT_EQ(finished, 2);
  EXPECT_EQ(sch.counters().rejected, 1);
}

TEST(Simulator, PagedFitsMoreThanContiguous) {
  SimConfig c;
  c.kv.num_blocks = 2048;
  c.kv.block_size = 16;
  c.kv.max_seq_len = 2048;
  c.wl.num_requests = 400;
  c.wl.max_seq_len = 2048;
  c.kv.policy = "paged";
  SimResult p = simulate(c);
  c.kv.policy = "contiguous";
  SimResult q = simulate(c);
  EXPECT_EQ(p.completed, 400);
  EXPECT_EQ(q.completed, 400);
  EXPECT_EQ(q.max_running, 2048 / 128);  // 16 requests * 128 blocks
  EXPECT_GT(p.max_running, 4 * q.max_running);
  EXPECT_LT(p.mean_frag, 0.1);
  EXPECT_GT(q.mean_frag, 0.5);
}
