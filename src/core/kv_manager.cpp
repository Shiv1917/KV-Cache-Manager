#include "core/kv_manager.h"

#include <algorithm>
#include <stdexcept>

#include "core/contiguous_manager.h"
#include "core/paged_manager.h"

namespace kvp {

KVStats KVManager::stats(const std::vector<const Sequence*>& live) const {
  KVStats s;
  s.num_blocks = num_blocks_;
  s.block_size = block_size_;
  s.blocks_in_use = blocks_in_use();
  s.blocks_cached_free = blocks_cached_free();
  s.allocated_slots = static_cast<int64_t>(s.blocks_in_use) * block_size_;
  s.num_seqs = static_cast<int>(live.size());
  s.prefix = prefix_stats();

  // Fill level of each physical block, taking the max over sequences sharing it.
  thread_local std::vector<int> fill;
  fill.assign(num_blocks_, -1);
  for (const Sequence* q : live) {
    s.block_table_entries += static_cast<int64_t>(q->block_table.size());
    for (size_t b = 0; b < q->block_table.size(); ++b) {
      const int f = std::clamp(q->num_computed - static_cast<int>(b) * block_size_, 0, block_size_);
      int& slot = fill[q->block_table[b]];
      slot = std::max(slot, f);
    }
  }
  for (int f : fill) {
    if (f < 0) continue;
    s.used_slots += f;
    if (f == 0) s.reserved_unused_slots += block_size_;
    else s.tail_waste_slots += block_size_ - f;
  }
  return s;
}

std::unique_ptr<KVManager> make_kv_manager(const KVManagerConfig& c) {
  if (c.policy == "paged") {
    const int wm = c.watermark_blocks >= 0 ? c.watermark_blocks
                                           : static_cast<int>(c.watermark_frac * c.num_blocks);
    return std::make_unique<PagedManager>(c.num_blocks, c.block_size, c.max_seq_len,
                                          c.prefix_caching, wm);
  }
  if (c.policy == "contiguous")
    return std::make_unique<ContiguousManager>(c.num_blocks, c.block_size, c.max_seq_len);
  throw std::invalid_argument("unknown KV policy: " + c.policy);
}

}  // namespace kvp
