#include "core/paged_manager.h"

#include <stdexcept>

#include "core/hash.h"

namespace kvp {

PagedManager::PagedManager(int num_blocks, int block_size, int max_seq_len, bool prefix_caching,
                           int watermark_blocks)
    : KVManager(num_blocks, block_size, max_seq_len),
      alloc_(num_blocks),
      cache_(alloc_, block_size, prefix_caching),
      watermark_(watermark_blocks) {}

bool PagedManager::can_admit(const Sequence& seq) const {
  const int n = static_cast<int>(seq.tokens.size());
  if (n > max_seq_len_) return false;
  int lru_hits = 0;
  const int hits = cache_.count_match(seq.tokens.data(), matchable_blocks(seq), &lru_hits);
  const int need = blocks_for(n) - hits;
  const int available = cache_.num_available() - lru_hits;
  return need <= available - watermark_;
}

void PagedManager::admit(Sequence& seq) {
  if (!seq.block_table.empty()) throw std::logic_error("admit: sequence already holds blocks");
  seq.block_hashes.clear();
  const int hits =
      cache_.match_and_acquire(seq.tokens.data(), matchable_blocks(seq), seq.block_table,
                               seq.block_hashes);
  seq.cached_tokens = hits * block_size_;
  seq.num_computed = seq.cached_tokens;
  seq.prefill_tokens_saved += seq.cached_tokens;
  if (!append_slot(seq)) throw std::runtime_error("admit: out of blocks (call can_admit first)");
}

bool PagedManager::append_slot(Sequence& seq) {
  const int need = blocks_for(static_cast<int>(seq.tokens.size()));
  while (static_cast<int>(seq.block_table.size()) < need) {
    int id = cache_.allocate();
    if (id < 0) return false;
    seq.block_table.push_back(id);
  }
  return true;
}

void PagedManager::commit(Sequence& seq) {
  if (!cache_.enabled()) return;
  const int full = seq.num_computed / block_size_;
  for (int b = static_cast<int>(seq.block_hashes.size()); b < full; ++b) {
    const uint64_t parent = b ? seq.block_hashes[b - 1] : 0;
    const int32_t* t = seq.tokens.data() + static_cast<size_t>(b) * block_size_;
    const uint64_t h = block_hash(parent, t, block_size_);
    cache_.insert(seq.block_table[b], parent, h, t);
    seq.block_hashes.push_back(h);
  }
}

void PagedManager::release(Sequence& seq) {
  // Release in reverse so the tail of a prefix is evicted before its head (LRU order).
  for (auto it = seq.block_table.rbegin(); it != seq.block_table.rend(); ++it) cache_.release(*it);
  seq.block_table.clear();
  seq.block_hashes.clear();
  seq.num_computed = 0;
}

}  // namespace kvp
