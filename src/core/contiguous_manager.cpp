#include "core/contiguous_manager.h"

#include <stdexcept>

namespace kvp {

ContiguousManager::ContiguousManager(int num_blocks, int block_size, int max_seq_len)
    : KVManager(num_blocks, block_size, max_seq_len),
      reserve_((max_seq_len + block_size - 1) / block_size),
      free_blocks_(num_blocks) {
  if (num_blocks > 0) extents_[0] = num_blocks;
}

int ContiguousManager::find_extent() const {
  for (const auto& [start, len] : extents_)
    if (len >= reserve_) return start;
  return -1;
}

bool ContiguousManager::can_admit(const Sequence& seq) const {
  if (static_cast<int>(seq.tokens.size()) > max_seq_len_) return false;
  return find_extent() >= 0;
}

void ContiguousManager::admit(Sequence& seq) {
  if (!seq.block_table.empty()) throw std::logic_error("admit: sequence already holds blocks");
  const int start = find_extent();
  if (start < 0) throw std::runtime_error("admit: no contiguous extent (call can_admit first)");
  auto it = extents_.find(start);
  const int len = it->second;
  extents_.erase(it);
  if (len > reserve_) extents_[start + reserve_] = len - reserve_;
  free_blocks_ -= reserve_;
  seq.block_table.resize(reserve_);
  for (int i = 0; i < reserve_; ++i) seq.block_table[i] = start + i;
  seq.cached_tokens = 0;
  seq.num_computed = 0;
}

bool ContiguousManager::append_slot(Sequence& seq) {
  return static_cast<int>(seq.tokens.size()) <= reserve_ * block_size_;
}

void ContiguousManager::release(Sequence& seq) {
  if (seq.block_table.empty()) return;
  int start = seq.block_table.front();
  int len = static_cast<int>(seq.block_table.size());
  free_blocks_ += len;
  // Coalesce with neighbours.
  auto next = extents_.lower_bound(start);
  if (next != extents_.end() && next->first == start + len) {
    len += next->second;
    next = extents_.erase(next);
  }
  if (next != extents_.begin()) {
    auto prev = std::prev(next);
    if (prev->first + prev->second == start) {
      start = prev->first;
      len += prev->second;
      extents_.erase(prev);
    }
  }
  extents_[start] = len;
  seq.block_table.clear();
  seq.num_computed = 0;
}

}  // namespace kvp
