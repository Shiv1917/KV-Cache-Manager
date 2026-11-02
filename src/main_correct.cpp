// kvcorrect: end-to-end correctness gate against HuggingFace reference outputs.
//
//  1. Greedy output matches HF token-for-token (per prompt, run alone) for --steps tokens.
//     A divergence is tolerated only at a near-tie (HF top-1/top-2 logit gap < --tie-margin),
//     where fp16 vs fp32 rounding can legitimately flip the argmax; it is reported either way.
//  2. Teacher-forced top-1 agreement with HF over every reference step, plus logit error.
//  3. Paged and contiguous managers produce bit-identical tokens (scattered physical blocks).
//  4. All prompts in one batch: no cross-contamination (matches HF under the same rule as 1).
//  5. Prefix caching on: hits happen and outputs still match HF.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>

#include "common/args.h"
#include "engine/engine.h"

using namespace kvp;

struct RefPrompt {
  std::vector<int32_t> prompt, greedy, top_ids;
  std::vector<float> top_logits;
};
struct Refs {
  int steps = 0, topk = 0;
  std::vector<RefPrompt> prompts;
};

static Refs load_refs(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[4];
  int32_t hdr[3];
  f.read(magic, 4);
  f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
  if (std::memcmp(magic, "KVPR", 4)) throw std::runtime_error("bad refs magic");
  Refs r;
  r.steps = hdr[1];
  r.topk = hdr[2];
  for (int i = 0; i < hdr[0]; ++i) {
    RefPrompt p;
    int32_t len;
    f.read(reinterpret_cast<char*>(&len), 4);
    p.prompt.resize(len);
    p.greedy.resize(r.steps);
    p.top_ids.resize(size_t(r.steps) * r.topk);
    p.top_logits.resize(size_t(r.steps) * r.topk);
    f.read(reinterpret_cast<char*>(p.prompt.data()), len * 4);
    f.read(reinterpret_cast<char*>(p.greedy.data()), r.steps * 4);
    f.read(reinterpret_cast<char*>(p.top_ids.data()), p.top_ids.size() * 4);
    f.read(reinterpret_cast<char*>(p.top_logits.data()), p.top_logits.size() * 4);
    r.prompts.push_back(std::move(p));
  }
  if (!f) throw std::runtime_error("refs file truncated");
  return r;
}

// Runs the given prompts through the engine together (all submitted at once), returns outputs.
static std::vector<std::vector<int32_t>> generate(Engine& eng, const Refs& refs,
                                                  const std::vector<int>& which, int steps) {
  for (int i : which) {
    auto s = std::make_unique<Sequence>();
    s->id = i;
    s->tokens = refs.prompts[i].prompt;
    s->prompt_len = static_cast<int>(s->tokens.size());
    s->max_new_tokens = steps;
    eng.add(std::move(s));
  }
  std::map<int, std::vector<int32_t>> out;
  while (!eng.idle()) {
    eng.step();
    for (auto& s : eng.take_finished())
      out[s->id] = std::vector<int32_t>(s->tokens.begin() + s->prompt_len, s->tokens.end());
  }
  std::vector<std::vector<int32_t>> v;
  for (int i : which) v.push_back(out[i]);
  return v;
}

struct Verdict {
  int first_mismatch;  // -1 if identical over all steps
  float margin;        // HF top1 - top2 at the mismatch
  bool ok;
};

static Verdict compare_hf(const RefPrompt& r, const std::vector<int32_t>& got, int steps,
                          int topk, float tie_margin) {
  for (int k = 0; k < steps; ++k) {
    if (k >= static_cast<int>(got.size()) || got[k] != r.greedy[k]) {
      const float m = r.top_logits[size_t(k) * topk] - r.top_logits[size_t(k) * topk + 1];
      return {k, m, m < tie_margin};
    }
  }
  return {-1, 0, true};
}

// Holds scattered blocks so that the paged manager hands out non-contiguous tables.
static std::vector<Sequence> fragment_pool(KVManager& kv, int n) {
  std::vector<Sequence> held, all;
  for (int i = 0; i < n; ++i) {
    Sequence s;
    s.id = -100 - i;
    s.tokens.assign(1 + (i * 37) % 50, 7);
    s.prompt_len = static_cast<int>(s.tokens.size());
    if (!kv.can_admit(s)) break;
    kv.admit(s);
    all.push_back(std::move(s));
  }
  for (size_t i = 0; i < all.size(); ++i) {
    if (i % 2) kv.release(all[i]);
    else held.push_back(std::move(all[i]));
  }
  return held;
}

int main(int argc, char** argv) {
  try {
    Args a(argc, argv);
    const std::string weights_path = a.str("weights", "data/weights.bin");
    const std::string refs_path = a.str("refs", "data/refs.bin");
    const float tie_margin = static_cast<float>(a.f64("tie-margin", 0.5));
    const int block_size = a.i32("block-size", 16);
    const std::string attn = a.str("attn", "auto");
    int steps = a.i32("steps", 0);
    a.warn_unused();

    Refs refs = load_refs(refs_path);
    if (steps <= 0 || steps > refs.steps) steps = refs.steps;
    const int N = static_cast<int>(refs.prompts.size());
    std::printf("loaded %d reference prompts, %d steps, top-%d\n", N, refs.steps, refs.topk);
    LlamaWeights weights(weights_path);

    EngineConfig base;
    base.kv.block_size = block_size;
    base.kv.max_seq_len = 512;
    base.pool_gb = 0.25;
    base.sched.max_num_seqs = 16;
    base.sched.max_batched_tokens = 2048;
    base.attn = attn == "v1" ? AttnVersion::V1 : attn == "v2" ? AttnVersion::V2 : AttnVersion::Auto;

    bool all_ok = true;
    auto check = [&](bool cond, const char* what) {
      std::printf("%s  %s\n", cond ? "[PASS]" : "[FAIL]", what);
      all_ok &= cond;
    };

    std::vector<std::vector<int32_t>> solo[2];
    const char* policies[2] = {"paged", "contiguous"};
    for (int pi = 0; pi < 2; ++pi) {
      EngineConfig ec = base;
      ec.kv.policy = policies[pi];
      Engine eng(weights, ec);
      std::vector<Sequence> held;
      if (pi == 0) held = fragment_pool(eng.kv(), 200);

      std::printf("\n== %s: each prompt alone (%s blocks)\n", policies[pi],
                  pi == 0 ? "scattered" : "contiguous");
      bool ok = true;
      int exact = 0;
      for (int i = 0; i < N; ++i) {
        auto out = generate(eng, refs, {i}, steps)[0];
        Verdict v = compare_hf(refs.prompts[i], out, steps, refs.topk, tie_margin);
        if (v.first_mismatch < 0) {
          ++exact;
          std::printf("  prompt %d (%zu tok): %d/%d tokens match HF\n", i,
                      refs.prompts[i].prompt.size(), steps, steps);
        } else {
          std::printf("  prompt %d (%zu tok): first divergence at step %d (HF top1-top2 margin "
                      "%.3f) -> %s\n",
                      i, refs.prompts[i].prompt.size(), v.first_mismatch, v.margin,
                      v.ok ? "near-tie, tolerated" : "REAL MISMATCH");
        }
        if (pi == 0 && i == 0) {
          std::printf("  table of prompt 0 is scattered: ");
          // (tables are released at finish; demonstrate via a re-admission)
          Sequence s;
          s.tokens = refs.prompts[0].prompt;
          s.prompt_len = static_cast<int>(s.tokens.size());
          eng.kv().admit(s);
          for (size_t b = 0; b < std::min<size_t>(s.block_table.size(), 8); ++b)
            std::printf("%d ", s.block_table[b]);
          std::printf("...\n");
          eng.kv().release(s);
        }
        ok &= v.ok;
        solo[pi].push_back(out);
      }
      std::printf("  %d/%d prompts exact over %d steps\n", exact, N, steps);
      check(ok, (std::string(policies[pi]) + ": greedy matches HF (alone)").c_str());

      if (pi == 0) {
        // Teacher-forced: score prompt + HF continuation in one pass, check per-step top-1.
        std::printf("\n== paged: teacher-forced logits vs HF\n");
        int agree = 0, total = 0;
        double max_err = 0, sum_err = 0;
        long n_err = 0;
        std::vector<float> logits;
        const int V = weights.config().vocab;
        for (int i = 0; i < N; ++i) {
          const auto& r = refs.prompts[i];
          std::vector<int32_t> toks = r.prompt;
          toks.insert(toks.end(), r.greedy.begin(), r.greedy.begin() + steps - 1);
          eng.score(toks, logits);
          for (int k = 0; k < steps; ++k) {
            const float* row = logits.data() + size_t(r.prompt.size() - 1 + k) * V;
            const int am = static_cast<int>(std::max_element(row, row + V) - row);
            agree += am == r.top_ids[size_t(k) * refs.topk];
            ++total;
            for (int j = 0; j < refs.topk; ++j) {
              const double e = std::fabs(row[r.top_ids[size_t(k) * refs.topk + j]] -
                                         r.top_logits[size_t(k) * refs.topk + j]);
              max_err = std::max(max_err, e);
              sum_err += e;
              ++n_err;
            }
          }
        }
        std::printf("  top-1 agreement %d/%d (%.2f%%), top-%d logit |err| mean %.4f max %.4f\n",
                    agree, total, 100.0 * agree / total, refs.topk, sum_err / n_err, max_err);
        check(agree >= 0.98 * total, "teacher-forced top-1 agreement >= 98%");

        // Batched: all prompts together.
        std::printf("\n== paged: all %d prompts in one batch\n", N);
        std::vector<int> all(N);
        for (int i = 0; i < N; ++i) all[i] = i;
        auto outs = generate(eng, refs, all, steps);
        bool bok = true;
        int same_as_solo = 0;
        for (int i = 0; i < N; ++i) {
          Verdict v = compare_hf(refs.prompts[i], outs[i], steps, refs.topk, tie_margin);
          same_as_solo += outs[i] == solo[0][i];
          if (v.first_mismatch >= 0)
            std::printf("  prompt %d: diverges from HF at %d (margin %.3f) %s\n", i,
                        v.first_mismatch, v.margin, v.ok ? "near-tie" : "REAL MISMATCH");
          bok &= v.ok;
        }
        std::printf("  %d/%d batched outputs identical to solo runs\n", same_as_solo, N);
        check(bok, "batched greedy matches HF (no cross-contamination)");
        for (auto& s : held) eng.kv().release(s);
      }
    }

    std::printf("\n== paged vs contiguous\n");
    int same = 0;
    for (int i = 0; i < N; ++i) same += solo[0][i] == solo[1][i];
    std::printf("  %d/%d prompts bit-identical\n", same, N);
    check(same == N, "paged and contiguous produce identical tokens");

    {
      std::printf("\n== paged + prefix caching (shared system prompt)\n");
      EngineConfig ec = base;
      ec.kv.policy = "paged";
      ec.kv.prefix_caching = true;
      Engine eng(weights, ec);
      bool ok = true;
      int same_solo = 0;
      for (int i = 0; i < N; ++i) {
        auto out = generate(eng, refs, {i}, steps)[0];
        Verdict v = compare_hf(refs.prompts[i], out, steps, refs.topk, tie_margin);
        same_solo += out == solo[0][i];
        if (v.first_mismatch >= 0)
          std::printf("  prompt %d: diverges from HF at %d (margin %.3f) %s\n", i,
                      v.first_mismatch, v.margin, v.ok ? "near-tie" : "REAL MISMATCH");
        ok &= v.ok;
      }
      // Re-run prompt 0 entirely: its whole prompt (minus last block) should now hit.
      auto again = generate(eng, refs, {0}, steps)[0];
      const auto st = eng.kv().prefix_stats();
      const auto& c = eng.scheduler().counters();
      std::printf("  block hits %lld / %lld queried, prefill tokens computed %lld, saved %lld\n",
                  (long long)st.hit_blocks, (long long)st.queried_blocks,
                  (long long)c.prefill_tokens, (long long)c.prefill_tokens_saved);
      std::printf("  %d/%d outputs identical to prefix-off runs; re-run of prompt 0 %s\n",
                  same_solo, N, again == solo[0][0] ? "identical" : "differs");
      check(st.hit_blocks > 0 && c.prefill_tokens_saved > 0, "prefix cache produces hits");
      check(ok && compare_hf(refs.prompts[0], again, steps, refs.topk, tie_margin).ok,
            "prefix-cached greedy matches HF");
    }

    std::printf("\n%s\n", all_ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return all_ok ? 0 : 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
