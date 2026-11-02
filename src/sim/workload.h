// Synthetic request generator: Poisson arrivals, lognormal prompt/output lengths (chat-like by
// default, roughly ShareGPT-shaped), optional shared system prompt, optional length CSV.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/args.h"
#include "core/sequence.h"

namespace kvp {

struct WorkloadConfig {
  int num_requests = 1000;
  double arrival_rate = 0;       // requests per time unit; <= 0 => all arrive at t = 0
  double prompt_mu = 5.0, prompt_sigma = 1.0;   // lognormal of user prompt length (median ~148)
  double output_mu = 5.3, output_sigma = 0.9;   // lognormal of output length (median ~200)
  int min_prompt = 4, max_prompt = 1024;
  int min_output = 4, max_output = 1024;
  int max_seq_len = 2048;
  double shared_frac = 0.0;      // fraction of requests that start with a shared system prompt
  int shared_len = 0;            // shared system prompt length (tokens)
  int num_shared_prefixes = 1;   // distinct system prompts (round-robin)
  std::string lengths_csv;       // optional "prompt_len,output_len" rows, sampled with replacement
  int vocab = 32000;
  uint64_t seed = 1;

  static WorkloadConfig from_args(const Args& a);
  std::string describe() const;
};

struct Request {
  int id;
  double arrival;
  int prompt_len;   // total, including the shared prefix
  int output_len;
  int prefix_id;    // -1 if no shared prefix
};

std::vector<Request> generate_workload(const WorkloadConfig& cfg);
// Deterministic prompt tokens for a request (shared prefix tokens depend only on prefix_id).
std::vector<int32_t> make_prompt_tokens(const Request& r, const WorkloadConfig& cfg);
std::unique_ptr<Sequence> make_sequence(const Request& r, const WorkloadConfig& cfg);

}  // namespace kvp
