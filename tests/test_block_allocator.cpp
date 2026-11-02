#include <gtest/gtest.h>

#include <random>
#include <set>

#include "core/block_allocator.h"

using namespace kvp;

TEST(BlockAllocator, AllocateAllThenFree) {
  BlockAllocator a(64);
  std::set<int> got;
  for (int i = 0; i < 64; ++i) {
    int id = a.allocate();
    ASSERT_GE(id, 0);
    ASSERT_TRUE(got.insert(id).second) << "duplicate block " << id;
  }
  EXPECT_EQ(a.allocate(), -1);
  EXPECT_EQ(a.num_free(), 0);
  for (int id : got) a.free(id);
  EXPECT_EQ(a.num_free(), 64);
}

TEST(BlockAllocator, RefcountKeepsBlockAlive) {
  BlockAllocator a(4);
  int id = a.allocate();
  a.incref(id);
  EXPECT_EQ(a.refcount(id), 2);
  a.free(id);
  EXPECT_EQ(a.num_free(), 3);
  a.free(id);
  EXPECT_EQ(a.num_free(), 4);
  EXPECT_THROW(a.free(id), std::logic_error);
}

// N random alloc/free/incref cycles: no leaks, and no block is ever handed out twice while live.
TEST(BlockAllocator, RandomCyclesNoLeakNoAlias) {
  constexpr int kBlocks = 257;
  BlockAllocator a(kBlocks);
  std::mt19937 rng(42);
  std::vector<int> refs(kBlocks, 0);   // shadow refcounts
  std::vector<int> live;               // one entry per outstanding reference
  for (int iter = 0; iter < 200000; ++iter) {
    int op = rng() % 3;
    if (op == 0) {
      int id = a.allocate();
      if (id < 0) {
        ASSERT_EQ(live.size() > 0, true);
        continue;
      }
      ASSERT_EQ(refs[id], 0) << "allocator returned a live block";
      refs[id] = 1;
      live.push_back(id);
    } else if (op == 1 && !live.empty()) {
      int id = live[rng() % live.size()];
      a.incref(id);
      ++refs[id];
      live.push_back(id);
    } else if (!live.empty()) {
      size_t k = rng() % live.size();
      int id = live[k];
      live[k] = live.back();
      live.pop_back();
      a.free(id);
      --refs[id];
    }
    if (iter % 997 == 0) {
      int expect_free = 0;
      for (int r : refs) expect_free += r == 0;
      ASSERT_EQ(a.num_free(), expect_free);
      for (int i = 0; i < kBlocks; ++i) ASSERT_EQ(a.refcount(i), refs[i]);
    }
  }
  for (int id : live) a.free(id);
  EXPECT_EQ(a.num_free(), kBlocks);
}
