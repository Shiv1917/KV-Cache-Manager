#pragma once
#include "core/block_allocator.h"
#include "core/kv_manager.h"
#include "core/prefix_cache.h"

namespace kvp {

// On-demand block allocation (PagedAttention). A sequence holds ceil(len / B) blocks, so the only
// waste is the unfilled tail of its last block. Optional hash-based prefix caching.
class PagedManager : public KVManager {
 public:
  PagedManager(int num_blocks, int block_size, int max_seq_len, bool prefix_caching,
               int watermark_blocks = 0);
  std::string name() const override { return cache_.enabled() ? "paged+prefix" : "paged"; }

  bool can_admit(const Sequence& seq) const override;
  void admit(Sequence& seq) override;
  bool append_slot(Sequence& seq) override;
  void commit(Sequence& seq) override;
  void release(Sequence& seq) override;

  int blocks_in_use() const override { return num_blocks_ - cache_.num_available(); }
  int blocks_cached_free() const override { return cache_.num_cached_free(); }
  PrefixCacheStats prefix_stats() const override { return cache_.stats(); }

  const BlockAllocator& allocator() const { return alloc_; }
  const PrefixCache& cache() const { return cache_; }

 private:
  int blocks_for(int ntok) const { return (ntok + block_size_ - 1) / block_size_; }
  // Leading full blocks eligible for a prefix hit. The last token is always recomputed (we need
  // its logits), so a block containing it is never shared.
  int matchable_blocks(const Sequence& s) const {
    return (static_cast<int>(s.tokens.size()) - 1) / block_size_;
  }

  BlockAllocator alloc_;
  PrefixCache cache_;
  int watermark_;
};

}  // namespace kvp
