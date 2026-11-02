#include "sim/workload.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace kvp {

WorkloadConfig WorkloadConfig::from_args(const Args& a) {
  WorkloadConfig c;
  c.num_requests = a.i32("requests", c.num_requests);
  c.arrival_rate = a.f64("rate", c.arrival_rate);
  c.prompt_mu = a.f64("prompt-mu", c.prompt_mu);
  c.prompt_sigma = a.f64("prompt-sigma", c.prompt_sigma);
  c.output_mu = a.f64("output-mu", c.output_mu);
  c.output_sigma = a.f64("output-sigma", c.output_sigma);
  c.min_prompt = a.i32("min-prompt", c.min_prompt);
  c.max_prompt = a.i32("max-prompt", c.max_prompt);
  c.min_output = a.i32("min-output", c.min_output);
  c.max_output = a.i32("max-output", c.max_output);
  c.max_seq_len = a.i32("max-seq-len", c.max_seq_len);
  c.shared_frac = a.f64("shared-frac", c.shared_frac);
  c.shared_len = a.i32("shared-len", c.shared_len);
  c.num_shared_prefixes = a.i32("num-prefixes", c.num_shared_prefixes);
  c.lengths_csv = a.str("lengths-csv", c.lengths_csv);
  c.seed = static_cast<uint64_t>(a.i64("seed", static_cast<long long>(c.seed)));
  // Fixed-length shortcut: --prompt-len / --output-len override the distributions.
  if (a.has("prompt-len")) {
    c.prompt_sigma = 0; c.prompt_mu = std::log(a.f64("prompt-len", 0));
    c.min_prompt = c.max_prompt = a.i32("prompt-len", 0);
  }
  if (a.has("output-len")) {
    c.output_sigma = 0; c.output_mu = std::log(a.f64("output-len", 0));
    c.min_output = c.max_output = a.i32("output-len", 0);
  }
  return c;
}

std::string WorkloadConfig::describe() const {
  std::ostringstream o;
  o << "requests=" << num_requests << " rate=" << arrival_rate << " prompt~LN(" << prompt_mu
    << "," << prompt_sigma << ") output~LN(" << output_mu << "," << output_sigma
    << ") shared_frac=" << shared_frac << " shared_len=" << shared_len
    << " max_seq_len=" << max_seq_len << " seed=" << seed;
  return o.str();
}

static std::vector<std::pair<int, int>> load_lengths(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open lengths csv: " + path);
  std::vector<std::pair<int, int>> v;
  std::string line;
  while (std::getline(f, line)) {
    int p = 0, o = 0;
    if (std::sscanf(line.c_str(), "%d,%d", &p, &o) == 2 && p > 0 && o > 0) v.push_back({p, o});
  }
  if (v.empty()) throw std::runtime_error("no rows in lengths csv: " + path);
  return v;
}

std::vector<Request> generate_workload(const WorkloadConfig& c) {
  std::mt19937_64 rng(c.seed);
  std::lognormal_distribution<double> pd(c.prompt_mu, std::max(c.prompt_sigma, 1e-9));
  std::lognormal_distribution<double> od(c.output_mu, std::max(c.output_sigma, 1e-9));
  std::exponential_distribution<double> gap(c.arrival_rate > 0 ? c.arrival_rate : 1.0);
  std::uniform_real_distribution<double> u01(0, 1);
  std::vector<std::pair<int, int>> table;
  if (!c.lengths_csv.empty()) table = load_lengths(c.lengths_csv);
  std::uniform_int_distribution<size_t> pick(0, table.empty() ? 0 : table.size() - 1);

  std::vector<Request> out;
  out.reserve(c.num_requests);
  double t = 0;
  int next_prefix = 0;
  for (int i = 0; i < c.num_requests; ++i) {
    if (c.arrival_rate > 0) t += gap(rng);
    int p, o;
    if (!table.empty()) {
      auto [tp, to] = table[pick(rng)];
      p = tp; o = to;
    } else {
      p = static_cast<int>(std::lround(pd(rng)));
      o = static_cast<int>(std::lround(od(rng)));
    }
    p = std::clamp(p, c.min_prompt, c.max_prompt);
    o = std::clamp(o, c.min_output, c.max_output);
    int prefix = -1;
    if (c.shared_len > 0 && u01(rng) < c.shared_frac) {
      prefix = next_prefix++ % std::max(1, c.num_shared_prefixes);
      p += c.shared_len;
    }
    p = std::min(p, c.max_seq_len - 1);
    o = std::max(1, std::min(o, c.max_seq_len - p));
    out.push_back({i, t, p, o, prefix});
  }
  return out;
}

std::vector<int32_t> make_prompt_tokens(const Request& r, const WorkloadConfig& c) {
  std::vector<int32_t> toks;
  toks.reserve(r.prompt_len);
  // Avoid ids 0..2 (unk/bos/eos in Llama tokenizers).
  if (r.prefix_id >= 0) {
    std::mt19937_64 prng(c.seed * 0x9E3779B97F4A7C15ULL + 0xABCDEF + r.prefix_id);
    std::uniform_int_distribution<int32_t> tok(3, c.vocab - 1);
    for (int i = 0; i < c.shared_len && i < r.prompt_len; ++i) toks.push_back(tok(prng));
  }
  std::mt19937_64 rng(c.seed * 0xD1B54A32D192ED03ULL + 0x1234567 + r.id);
  std::uniform_int_distribution<int32_t> tok(3, c.vocab - 1);
  while (static_cast<int>(toks.size()) < r.prompt_len) toks.push_back(tok(rng));
  return toks;
}

std::unique_ptr<Sequence> make_sequence(const Request& r, const WorkloadConfig& c) {
  auto s = std::make_unique<Sequence>();
  s->id = r.id;
  s->tokens = make_prompt_tokens(r, c);
  s->prompt_len = r.prompt_len;
  s->max_new_tokens = r.output_len;
  s->arrival = r.arrival;
  return s;
}

}  // namespace kvp
