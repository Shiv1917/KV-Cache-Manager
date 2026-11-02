#pragma once
#include <map>

#include "core/kv_manager.h"

namespace kvp {

// Baseline: every request reserves ceil(max_seq_len / B) *consecutive* blocks at admission, as a
// framework that pre-allocates a max-length KV buffer per request would. Same block-table format,
// same kernel; only the policy differs. Suffers internal waste (reserved-but-unused slots) and
// external fragmentation (first-fit over free extents).
class ContiguousManager : public KVManager {
 public:
  ContiguousManager(int num_blocks, int block_size, int max_seq_len);
  std::string name() const override { return "contiguous"; }

  bool can_admit(const Sequence& seq) const override;
  void admit(Sequence& seq) override;
  bool append_slot(Sequence& seq) override;
  void release(Sequence& seq) override;
  int blocks_in_use() const override { return num_blocks_ - free_blocks_; }

  int reserve_blocks() const { return reserve_; }
  int num_free_extents() const { return static_cast<int>(extents_.size()); }

 private:
  int find_extent() const;  // start of first-fit extent, or -1
  int reserve_;
  int free_blocks_;
  std::map<int, int> extents_;  // free extents: start -> length
};

}  // namespace kvp
