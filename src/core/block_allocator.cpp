#include "core/block_allocator.h"

#include <stdexcept>

namespace kvp {

BlockAllocator::BlockAllocator(int num_blocks)
    : refcount_(num_blocks, 0), on_free_list_(num_blocks, 1) {
  free_list_.reserve(num_blocks);
  // Push in reverse so that allocate() hands out 0, 1, 2, ... initially.
  for (int i = num_blocks - 1; i >= 0; --i) free_list_.push_back(i);
}

int BlockAllocator::allocate() {
  if (free_list_.empty()) return -1;
  int id = free_list_.back();
  free_list_.pop_back();
  on_free_list_[id] = 0;
  refcount_[id] = 1;
  return id;
}

void BlockAllocator::free(int id) {
  if (decref(id)) push_free(id);
}

void BlockAllocator::incref(int id) {
  if (on_free_list_[id]) throw std::logic_error("incref on free block");
  ++refcount_[id];
}

bool BlockAllocator::decref(int id) {
  if (refcount_[id] <= 0) throw std::logic_error("decref on block with refcount 0");
  return --refcount_[id] == 0;
}

void BlockAllocator::push_free(int id) {
  if (refcount_[id] != 0 || on_free_list_[id]) throw std::logic_error("push_free on live block");
  on_free_list_[id] = 1;
  free_list_.push_back(id);
}

void BlockAllocator::revive(int id) {
  if (refcount_[id] != 0 || on_free_list_[id]) throw std::logic_error("revive on bad block");
  refcount_[id] = 1;
}

}  // namespace kvp
