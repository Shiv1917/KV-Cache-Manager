#pragma once
#include <cstdint>
#include <vector>

namespace kvp {

// One request. `tokens` holds prompt + generated tokens. Tokens [0, num_computed) have their
// K/V in the cache; each forward step computes tokens [num_computed, tokens.size()) and samples
// the next token from the last row. A decode step is just the case num_computed == size - 1.
struct Sequence {
  int id = 0;
  std::vector<int32_t> tokens;
  int prompt_len = 0;       // original prompt length (generated = size - prompt_len)
  int max_new_tokens = 0;
  int num_computed = 0;
  std::vector<int32_t> block_table;  // physical block IDs; logical token t -> table[t / B]

  // Prefix-cache bookkeeping: hashes of the leading full blocks that are registered/hit.
  std::vector<uint64_t> block_hashes;
  int cached_tokens = 0;    // tokens skipped at the last admission thanks to prefix hits

  // Metrics (seconds for the engine, steps for the simulator).
  double arrival = 0, admitted = -1, first_token = -1, finished = -1;
  int num_preemptions = 0;
  int64_t prefill_tokens_computed = 0;
  int64_t prefill_tokens_saved = 0;

  int num_generated() const { return static_cast<int>(tokens.size()) - prompt_len; }
  bool done() const { return num_generated() >= max_new_tokens; }
};

}  // namespace kvp
