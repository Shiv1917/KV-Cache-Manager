#include <gtest/gtest.h>

#include "core/hash.h"
#include "core/paged_manager.h"

using namespace kvp;

static Sequence seq_from(int id, std::vector<int32_t> toks) {
  Sequence s;
  s.id = id;
  s.tokens = std::move(toks);
  s.prompt_len = static_cast<int>(s.tokens.size());
  s.max_new_tokens = 100;
  return s;
}
static std::vector<int32_t> iota_toks(int n, int start) {
  std::vector<int32_t> v(n);
  for (int i = 0; i < n; ++i) v[i] = start + i;
  return v;
}
// Pretend the forward pass ran: everything is computed, register full blocks.
static void compute(PagedManager& m, Sequence& s) {
  s.num_computed = static_cast<int>(s.tokens.size());
  m.commit(s);
}

TEST(Hash, Xxh64KnownVectors) {
  EXPECT_EQ(xxh64("", 0, 0), 0xEF46DB3751D8E999ULL);
  EXPECT_EQ(xxh64("a", 1, 0), 0xD24EC4F1A98C6E5BULL);
  EXPECT_EQ(xxh64("abc", 3, 0), 0x44BC2CF5AD770999ULL);
  const char* s = "Nobody inspects the spammish repetition";
  EXPECT_EQ(xxh64(s, 39, 0), 0xFBCEA83C8A378BF1ULL);
}

TEST(PrefixCache, HitAndMiss) {
  PagedManager m(32, 4, 256, true);
  Sequence a = seq_from(0, iota_toks(13, 100));
  m.admit(a);
  EXPECT_EQ(a.num_computed, 0);
  compute(m, a);
  EXPECT_EQ(m.cache().num_hashed(), 3);

  Sequence b = seq_from(1, iota_toks(13, 100));
  ASSERT_TRUE(m.can_admit(b));
  m.admit(b);
  EXPECT_EQ(b.cached_tokens, 12);
  EXPECT_EQ(b.num_computed, 12);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(b.block_table[i], a.block_table[i]);
    EXPECT_EQ(m.allocator().refcount(a.block_table[i]), 2);
  }
  EXPECT_NE(b.block_table[3], a.block_table[3]);  // tail stays private

  // Shares only the first block, then diverges.
  auto t = iota_toks(13, 100);
  t[5] = 9999;
  Sequence c = seq_from(2, t);
  m.admit(c);
  EXPECT_EQ(c.cached_tokens, 4);
  EXPECT_EQ(c.block_table[0], a.block_table[0]);

  // Different first block: total miss even though later blocks match.
  t = iota_toks(13, 100);
  t[0] = 1;
  Sequence d = seq_from(3, t);
  m.admit(d);
  EXPECT_EQ(d.cached_tokens, 0);

  const auto& st = m.prefix_stats();
  EXPECT_EQ(st.hit_blocks, 3 + 1);
  EXPECT_EQ(st.queried_blocks, 4 * 3);
}

TEST(PrefixCache, FullHitStillRecomputesLastToken) {
  PagedManager m(32, 4, 256, true);
  Sequence a = seq_from(0, iota_toks(8, 0));  // exactly 2 full blocks
  m.admit(a);
  compute(m, a);
  EXPECT_EQ(m.cache().num_hashed(), 2);
  Sequence b = seq_from(1, iota_toks(8, 0));
  m.admit(b);
  // Only block 0 is matchable; block 1 holds the last token which must be recomputed.
  EXPECT_EQ(b.cached_tokens, 4);
  EXPECT_EQ(b.num_computed, 4);
  EXPECT_NE(b.block_table[1], a.block_table[1]);
}

TEST(PrefixCache, ReleasedBlocksStayFindable) {
  PagedManager m(16, 4, 256, true);
  Sequence a = seq_from(0, iota_toks(13, 0));
  m.admit(a);
  compute(m, a);
  auto blocks = a.block_table;
  m.release(a);
  EXPECT_EQ(m.blocks_in_use(), 0);
  EXPECT_EQ(m.blocks_cached_free(), 3);  // 3 full hashed blocks; the partial tail was freed
  Sequence b = seq_from(1, iota_toks(13, 0));
  ASSERT_TRUE(m.can_admit(b));
  m.admit(b);
  EXPECT_EQ(b.cached_tokens, 12);
  for (int i = 0; i < 3; ++i) EXPECT_EQ(b.block_table[i], blocks[i]);
  EXPECT_EQ(m.blocks_cached_free(), 0);
}

TEST(PrefixCache, EvictionIsLruAndTailFirst) {
  PagedManager m(8, 4, 256, true);
  Sequence x = seq_from(0, iota_toks(9, 0));    // blocks: 2 full (hashed) + 1 partial
  Sequence y = seq_from(1, iota_toks(9, 500));
  m.admit(x); compute(m, x);
  m.admit(y); compute(m, y);
  auto xb = x.block_table, yb = y.block_table;
  m.release(x);  // x released first => older in LRU
  m.release(y);
  // Reverse release order puts each sequence's tail before its head.
  std::vector<int> expect = {xb[1], xb[0], yb[1], yb[0]};
  EXPECT_EQ(m.cache().lru_order(), expect);
  EXPECT_EQ(m.allocator().num_free(), 4);

  // A 5-block sequence needs 1 eviction: x's tail block goes first.
  Sequence z = seq_from(2, iota_toks(20, 9000));
  ASSERT_TRUE(m.can_admit(z));
  m.admit(z);
  EXPECT_EQ(m.prefix_stats().evictions, 1);
  expect = {xb[0], yb[1], yb[0]};
  EXPECT_EQ(m.cache().lru_order(), expect);

  // x's head block is still cached; its second block is gone.
  Sequence x2 = seq_from(3, iota_toks(9, 0));
  int lru_hits = 0;
  EXPECT_EQ(m.cache().count_match(x2.tokens.data(), 2, &lru_hits), 1);
  EXPECT_EQ(lru_hits, 1);
}

TEST(PrefixCache, CanAdmitAccountsForLruHits) {
  // 4 blocks total. A cached 2-block prefix sits on the LRU; a new sequence that hits it needs
  // those 2 plus 2 fresh blocks: it fits exactly. One more fresh block would not.
  PagedManager m(4, 4, 256, true);
  Sequence a = seq_from(0, iota_toks(9, 0));
  m.admit(a); compute(m, a); m.release(a);
  Sequence b = seq_from(1, iota_toks(16, 0));  // 2 hits + 2 fresh
  EXPECT_TRUE(m.can_admit(b));
  Sequence c = seq_from(2, iota_toks(17, 0));  // 2 hits + 3 fresh
  EXPECT_FALSE(m.can_admit(c));
}

TEST(PrefixCache, DisabledNeverCaches) {
  PagedManager m(16, 4, 256, false);
  Sequence a = seq_from(0, iota_toks(13, 0));
  m.admit(a); compute(m, a);
  Sequence b = seq_from(1, iota_toks(13, 0));
  m.admit(b);
  EXPECT_EQ(b.cached_tokens, 0);
  m.release(a); m.release(b);
  EXPECT_EQ(m.allocator().num_free(), 16);
}
