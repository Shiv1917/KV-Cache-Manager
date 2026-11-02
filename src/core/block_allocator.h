// Free list of physical KV block IDs with per-block reference counts.
#pragma once
#include <cstdint>
#include <vector>

namespace kvp {

class BlockAllocator {
 public:
  explicit BlockAllocator(int num_blocks);

  // Pops a free block and sets its refcount to 1. Returns -1 if none free.
  int allocate();
  // Decrements the refcount; when it reaches zero the block returns to the free list.
  void free(int id);
  void incref(int id);
  // Decrements the refcount without touching the free list. Returns true when it hit zero.
  // Used by the prefix cache, which parks zero-ref hashed blocks on an LRU list instead.
  bool decref(int id);
  // Returns a zero-refcount block to the free list.
  void push_free(int id);
  // Takes a specific zero-refcount block that is NOT on the free list (e.g. revived from LRU).
  void revive(int id);

  int refcount(int id) const { return refcount_[id]; }
  int num_free() const { return static_cast<int>(free_list_.size()); }
  int num_blocks() const { return static_cast<int>(refcount_.size()); }

 private:
  std::vector<int32_t> free_list_;
  std::vector<int32_t> refcount_;
  std::vector<uint8_t> on_free_list_;
};

}  // namespace kvp
