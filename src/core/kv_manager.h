// Allocation-policy interface. The engine, kernels and scheduler are identical for every policy;
// only the KVManager implementation changes, which isolates allocation policy as the variable.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/prefix_cache.h"
#include "core/sequence.h"

namespace kvp {

struct KVStats {
  int num_blocks = 0;
  int block_size = 0;
  int blocks_in_use = 0;        // refcount > 0 (reserved counts as in use)
  int blocks_cached_free = 0;   // refcount 0 but kept for prefix hits
  int64_t allocated_slots = 0;  // blocks_in_use * block_size
  int64_t used_slots = 0;       // unique physical slots holding a token
  int64_t tail_waste_slots = 0; // empty slots inside partially-filled blocks
  int64_t reserved_unused_slots = 0;  // slots in fully-empty reserved blocks (contiguous only)
  int64_t block_table_entries = 0;    // sum of block-table lengths (metadata overhead)
  int num_seqs = 0;
  PrefixCacheStats prefix;

  double fragmentation() const {
    return allocated_slots ? double(allocated_slots - used_slots) / double(allocated_slots) : 0.0;
  }
};

class KVManager {
 public:
  virtual ~KVManager() = default;
  virtual std::string name() const = 0;
  int block_size() const { return block_size_; }
  int num_blocks() const { return num_blocks_; }
  int max_seq_len() const { return max_seq_len_; }

  // Can the sequence (all tokens currently in seq.tokens) be admitted right now?
  virtual bool can_admit(const Sequence& seq) const = 0;
  // Fills block_table for all of seq.tokens, applying prefix hits; sets num_computed to the
  // number of cached tokens (the scheduler then only prefills the suffix).
  virtual void admit(Sequence& seq) = 0;
  // Makes room for every token in seq.tokens (i.e. the next slot). false => out of blocks.
  virtual bool append_slot(Sequence& seq) = 0;
  // Called after a forward step: newly completed full blocks become cacheable.
  virtual void commit(Sequence& /*seq*/) {}
  virtual void release(Sequence& seq) = 0;

  virtual int blocks_in_use() const = 0;
  virtual int blocks_cached_free() const { return 0; }
  virtual PrefixCacheStats prefix_stats() const { return {}; }

  // Fragmentation / occupancy snapshot over the live sequences.
  KVStats stats(const std::vector<const Sequence*>& live) const;

 protected:
  KVManager(int num_blocks, int block_size, int max_seq_len)
      : num_blocks_(num_blocks), block_size_(block_size), max_seq_len_(max_seq_len) {}
  int num_blocks_, block_size_, max_seq_len_;
};

struct KVManagerConfig {
  std::string policy = "paged";  // "paged" | "contiguous"
  int num_blocks = 1024;
  int block_size = 16;
  int max_seq_len = 2048;
  bool prefix_caching = false;
  int watermark_blocks = -1;     // paged: blocks kept spare at admission (-1 => watermark_frac)
  double watermark_frac = 0.01;  // vLLM-style default: 1% of the pool
};

std::unique_ptr<KVManager> make_kv_manager(const KVManagerConfig& cfg);

}  // namespace kvp
