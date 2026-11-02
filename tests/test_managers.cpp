#include <gtest/gtest.h>

#include <map>
#include <random>
#include <set>

#include "core/contiguous_manager.h"
#include "core/paged_manager.h"

using namespace kvp;

static Sequence make_seq(int id, int len, int seed = 0) {
  Sequence s;
  s.id = id;
  std::mt19937 rng(seed * 1000 + id);
  for (int i = 0; i < len; ++i) s.tokens.push_back(3 + rng() % 30000);
  s.prompt_len = len;
  s.max_new_tokens = 1 << 20;
  return s;
}

TEST(PagedManager, AllocatesOnDemand) {
  PagedManager m(100, 16, 2048, false);
  Sequence s = make_seq(0, 17);
  ASSERT_TRUE(m.can_admit(s));
  m.admit(s);
  EXPECT_EQ(s.block_table.size(), 2u);
  EXPECT_EQ(s.num_computed, 0);
  EXPECT_EQ(m.blocks_in_use(), 2);
  s.num_computed = 17;
  for (int i = 0; i < 15; ++i) {  // fill block 2 up to 32 tokens
    s.tokens.push_back(5);
    ASSERT_TRUE(m.append_slot(s));
    s.num_computed++;
  }
  EXPECT_EQ(s.block_table.size(), 2u);
  s.tokens.push_back(5);
  ASSERT_TRUE(m.append_slot(s));
  EXPECT_EQ(s.block_table.size(), 3u);
  m.release(s);
  EXPECT_EQ(m.blocks_in_use(), 0);
}

TEST(PagedManager, AppendFailsWhenFull) {
  PagedManager m(2, 4, 2048, false);
  Sequence s = make_seq(0, 8);
  m.admit(s);
  s.tokens.push_back(1);
  EXPECT_FALSE(m.append_slot(s));
  Sequence t = make_seq(1, 1);
  EXPECT_FALSE(m.can_admit(t));
}

TEST(ContiguousManager, ReservesMaxSeqLen) {
  ContiguousManager m(64, 16, 256);  // 16 blocks per request
  Sequence a = make_seq(0, 5), b = make_seq(1, 5), c = make_seq(2, 5), d = make_seq(3, 5),
           e = make_seq(4, 5);
  for (Sequence* s : {&a, &b, &c, &d}) {
    ASSERT_TRUE(m.can_admit(*s));
    m.admit(*s);
    ASSERT_EQ(s->block_table.size(), 16u);
    for (size_t i = 1; i < s->block_table.size(); ++i)
      ASSERT_EQ(s->block_table[i], s->block_table[i - 1] + 1);
  }
  EXPECT_FALSE(m.can_admit(e));  // only 4 fit, even though each uses 1 block's worth of tokens
  m.release(b);
  EXPECT_TRUE(m.can_admit(e));
  m.release(a);
  m.release(c);
  m.release(d);
  EXPECT_EQ(m.blocks_in_use(), 0);
  EXPECT_EQ(m.num_free_extents(), 1);  // fully coalesced
}

TEST(ContiguousManager, ExternalFragmentation) {
  ContiguousManager m(48, 16, 256);  // 3 slots of 16 blocks
  Sequence a = make_seq(0, 5), b = make_seq(1, 5), c = make_seq(2, 5);
  m.admit(a); m.admit(b); m.admit(c);
  m.release(a);
  m.release(c);
  EXPECT_EQ(m.num_free_extents(), 2);
  m.release(b);
  EXPECT_EQ(m.num_free_extents(), 1);
}

// Random admit/grow/release under both policies: tables never alias, nothing leaks.
template <class M>
static void random_churn(M& m, int iters, uint32_t seed) {
  std::mt19937 rng(seed);
  std::map<int, Sequence> live;
  int next_id = 0;
  for (int it = 0; it < iters; ++it) {
    int op = rng() % 4;
    if (op == 0) {
      Sequence s = make_seq(next_id++, 1 + rng() % 300, seed);
      if (m.can_admit(s)) {
        m.admit(s);
        s.num_computed = static_cast<int>(s.tokens.size());
        live.emplace(s.id, std::move(s));
      }
    } else if (op <= 2 && !live.empty()) {
      auto itr = std::next(live.begin(), rng() % live.size());
      Sequence& s = itr->second;
      if (static_cast<int>(s.tokens.size()) < m.max_seq_len()) {
        s.tokens.push_back(7);
        if (m.append_slot(s)) s.num_computed++;
        else s.tokens.pop_back();
      }
    } else if (!live.empty()) {
      auto itr = std::next(live.begin(), rng() % live.size());
      m.release(itr->second);
      live.erase(itr);
    }
    std::set<int> owned;
    int total = 0;
    for (auto& [id, s] : live) {
      ASSERT_GE(static_cast<int>(s.block_table.size()) * m.block_size(),
                static_cast<int>(s.tokens.size()));
      for (int b : s.block_table) {
        ASSERT_TRUE(owned.insert(b).second) << "block " << b << " aliased";
        ++total;
      }
    }
    ASSERT_EQ(m.blocks_in_use(), total);
  }
  for (auto& [id, s] : live) m.release(s);
  ASSERT_EQ(m.blocks_in_use(), 0);
}

TEST(PagedManager, RandomChurnNoAliasNoLeak) {
  PagedManager m(512, 16, 1024, false);
  random_churn(m, 20000, 1);
}

TEST(ContiguousManager, RandomChurnNoAliasNoLeak) {
  ContiguousManager m(512, 16, 1024);
  random_churn(m, 20000, 2);
}

TEST(KVStats, FragmentationAccounting) {
  PagedManager p(100, 16, 256, false);
  ContiguousManager c(100, 16, 256);
  Sequence a = make_seq(0, 20), b = make_seq(1, 20);
  p.admit(a); a.num_computed = 20;
  c.admit(b); b.num_computed = 20;
  KVStats sp = p.stats({&a});
  EXPECT_EQ(sp.allocated_slots, 32);
  EXPECT_EQ(sp.used_slots, 20);
  EXPECT_EQ(sp.tail_waste_slots, 12);
  EXPECT_EQ(sp.reserved_unused_slots, 0);
  KVStats sc = c.stats({&b});
  EXPECT_EQ(sc.allocated_slots, 256);
  EXPECT_EQ(sc.used_slots, 20);
  EXPECT_EQ(sc.tail_waste_slots, 12);
  EXPECT_EQ(sc.reserved_unused_slots, 256 - 32);
}
