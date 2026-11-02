// Hash-chained prefix cache over full KV blocks (vLLM "automatic prefix caching" style).
//
//  - h_i = xxh64(seed = h_{i-1}, tokens of block i); only full blocks are cacheable.
//  - A block with a registered hash stays findable after its refcount drops to zero: it is parked
//    on an LRU list instead of the free list, and only evicted when the free list runs dry.
//  - On a hit we verify the stored parent hash and tokens, so a 64-bit collision cannot alias.
#pragma once
#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>

#include "core/block_allocator.h"

namespace kvp {

struct PrefixCacheStats {
  int64_t queried_blocks = 0;  // full prompt blocks eligible for lookup
  int64_t hit_blocks = 0;
  int64_t inserted_blocks = 0;
  int64_t evictions = 0;
  int64_t duplicate_inserts = 0;  // same content computed twice concurrently
};

class PrefixCache {
 public:
  PrefixCache(BlockAllocator& alloc, int block_size, bool enabled);

  bool enabled() const { return enabled_; }

  // Gets a writable block: free list first, then evicts the least-recently-used cached block.
  int allocate();
  // Drops one reference. At zero, hashed blocks go to the LRU list, unhashed to the free list.
  void release(int id);

  // Walks `num_blocks` full blocks of `tokens` and acquires every leading hit (incref, or revive
  // from LRU). Appends block ids / hashes to the outputs. Stops at the first miss.
  int match_and_acquire(const int32_t* tokens, int num_blocks, std::vector<int32_t>& blocks,
                        std::vector<uint64_t>& hashes);
  // Same walk without side effects. `lru_hits` = how many hits currently sit on the LRU list
  // (acquiring them would shrink the pool of evictable blocks).
  int count_match(const int32_t* tokens, int num_blocks, int* lru_hits) const;

  // Registers a computed, full block. Returns false if the hash is already present (the block
  // then stays private and is freed normally).
  bool insert(int block, uint64_t parent_hash, uint64_t hash, const int32_t* tokens);

  // Blocks that allocate() can still hand out.
  int num_available() const { return alloc_.num_free() + static_cast<int>(lru_.size()); }
  int num_cached_free() const { return static_cast<int>(lru_.size()); }
  int num_hashed() const { return static_cast<int>(map_.size()); }
  bool is_hashed(int id) const { return has_hash_[id] != 0; }
  const PrefixCacheStats& stats() const { return stats_; }
  // For tests: block ids on the LRU list, oldest first.
  std::vector<int> lru_order() const { return {lru_.begin(), lru_.end()}; }

 private:
  int lookup(uint64_t parent, uint64_t h, const int32_t* tokens) const;
  void evict_lru_front();

  BlockAllocator& alloc_;
  int block_size_;
  bool enabled_;
  std::unordered_map<uint64_t, int32_t> map_;
  std::vector<uint64_t> hash_, parent_;
  std::vector<uint8_t> has_hash_;
  std::vector<int32_t> tokens_;  // [num_blocks * block_size], for verification
  std::list<int32_t> lru_;
  std::vector<std::list<int32_t>::iterator> lru_pos_;
  std::vector<uint8_t> in_lru_;
  PrefixCacheStats stats_;
};

}  // namespace kvp
