#include "core/prefix_cache.h"

#include <algorithm>
#include <cstring>

#include "core/hash.h"

namespace kvp {

PrefixCache::PrefixCache(BlockAllocator& alloc, int block_size, bool enabled)
    : alloc_(alloc), block_size_(block_size), enabled_(enabled) {
  const int n = alloc.num_blocks();
  if (enabled_) {
    hash_.assign(n, 0);
    parent_.assign(n, 0);
    tokens_.assign(static_cast<size_t>(n) * block_size, 0);
  }
  has_hash_.assign(n, 0);
  in_lru_.assign(n, 0);
  lru_pos_.resize(n);
}

void PrefixCache::evict_lru_front() {
  int id = lru_.front();
  lru_.pop_front();
  in_lru_[id] = 0;
  map_.erase(hash_[id]);
  has_hash_[id] = 0;
  ++stats_.evictions;
  alloc_.push_free(id);
}

int PrefixCache::allocate() {
  if (alloc_.num_free() == 0 && !lru_.empty()) evict_lru_front();
  return alloc_.allocate();
}

void PrefixCache::release(int id) {
  if (!alloc_.decref(id)) return;
  if (has_hash_[id]) {
    lru_.push_back(id);
    lru_pos_[id] = std::prev(lru_.end());
    in_lru_[id] = 1;
  } else {
    alloc_.push_free(id);
  }
}

int PrefixCache::lookup(uint64_t parent, uint64_t h, const int32_t* tokens) const {
  auto it = map_.find(h);
  if (it == map_.end()) return -1;
  int id = it->second;
  if (parent_[id] != parent) return -1;
  if (std::memcmp(&tokens_[static_cast<size_t>(id) * block_size_], tokens,
                  sizeof(int32_t) * block_size_) != 0)
    return -1;
  return id;
}

int PrefixCache::count_match(const int32_t* tokens, int num_blocks, int* lru_hits) const {
  int hits = 0, lru = 0;
  if (enabled_) {
    uint64_t parent = 0;
    for (int b = 0; b < num_blocks; ++b) {
      const int32_t* t = tokens + static_cast<size_t>(b) * block_size_;
      uint64_t h = block_hash(parent, t, block_size_);
      int id = lookup(parent, h, t);
      if (id < 0) break;
      ++hits;
      if (in_lru_[id]) ++lru;
      parent = h;
    }
  }
  if (lru_hits) *lru_hits = lru;
  return hits;
}

int PrefixCache::match_and_acquire(const int32_t* tokens, int num_blocks,
                                   std::vector<int32_t>& blocks, std::vector<uint64_t>& hashes) {
  if (!enabled_) return 0;
  stats_.queried_blocks += num_blocks;
  uint64_t parent = 0;
  int hits = 0;
  for (int b = 0; b < num_blocks; ++b) {
    const int32_t* t = tokens + static_cast<size_t>(b) * block_size_;
    uint64_t h = block_hash(parent, t, block_size_);
    int id = lookup(parent, h, t);
    if (id < 0) break;
    if (in_lru_[id]) {
      lru_.erase(lru_pos_[id]);
      in_lru_[id] = 0;
      alloc_.revive(id);
    } else {
      alloc_.incref(id);
    }
    blocks.push_back(id);
    hashes.push_back(h);
    parent = h;
    ++hits;
  }
  stats_.hit_blocks += hits;
  return hits;
}

bool PrefixCache::insert(int block, uint64_t parent_hash, uint64_t hash, const int32_t* tokens) {
  if (!enabled_ || has_hash_[block]) return false;
  if (map_.count(hash)) {
    ++stats_.duplicate_inserts;
    return false;
  }
  map_.emplace(hash, block);
  hash_[block] = hash;
  parent_[block] = parent_hash;
  has_hash_[block] = 1;
  std::memcpy(&tokens_[static_cast<size_t>(block) * block_size_], tokens,
              sizeof(int32_t) * block_size_);
  ++stats_.inserted_blocks;
  return true;
}

}  // namespace kvp
