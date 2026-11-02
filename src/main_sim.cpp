// kvsim: CPU-only allocator / scheduler simulator.
//
//   kvsim --policy paged --block-size 16 --pool-gb 2 --requests 2000 --rate 0 \
//         --trace results/sim_trace.csv --summary results/sim_summary.csv
#include <cstdio>
#include <string>

#include "common/args.h"
#include "sim/simulator.h"

using namespace kvp;

// TinyLlama-1.1B fp16: 2 (K,V) * 22 layers * 4 kv heads * 64 dim * 2 bytes.
static constexpr long long kBytesPerToken = 2LL * 22 * 4 * 64 * 2;

int main(int argc, char** argv) {
  try {
    Args a(argc, argv);
    if (a.has("help")) {
      std::printf(
          "kvsim flags:\n"
          "  --policy paged|contiguous|both   --prefix (enable prefix caching)\n"
          "  --block-size N  --num-blocks N | --pool-gb G  --max-seq-len N\n"
          "  --bytes-per-token N (default %lld, TinyLlama fp16)\n"
          "  --requests N --rate R (req/step, 0=burst) --prompt-mu/-sigma --output-mu/-sigma\n"
          "  --prompt-len N --output-len N (fixed lengths) --lengths-csv FILE\n"
          "  --shared-frac F --shared-len N --num-prefixes K --seed S\n"
          "  --max-num-seqs N --max-batched-tokens N --watermark N\n"
          "  --trace FILE.csv --trace-every N --summary FILE.csv\n",
          kBytesPerToken);
      return 0;
    }
    SimConfig cfg;
    cfg.kv.block_size = a.i32("block-size", 16);
    cfg.kv.max_seq_len = a.i32("max-seq-len", 2048);
    const long long bpt = a.i64("bytes-per-token", kBytesPerToken);
    if (a.has("pool-gb")) {
      const double bytes = a.f64("pool-gb", 1) * (1LL << 30);
      cfg.kv.num_blocks = static_cast<int>(bytes / (double(bpt) * cfg.kv.block_size));
    } else {
      cfg.kv.num_blocks = a.i32("num-blocks", 4096);
    }
    cfg.kv.prefix_caching = a.flag("prefix");
    cfg.kv.watermark_blocks = a.i32("watermark", -1);
    cfg.sched.max_num_seqs = a.i32("max-num-seqs", 1024);
    cfg.sched.max_batched_tokens = a.i32("max-batched-tokens", 8192);
    cfg.wl = WorkloadConfig::from_args(a);
    cfg.wl.max_seq_len = cfg.kv.max_seq_len;
    cfg.trace_csv = a.str("trace", "");
    cfg.trace_every = a.i32("trace-every", 1);
    const std::string summary = a.str("summary", "");
    const std::string policy = a.str("policy", "both");
    a.warn_unused();

    for (std::string p : {std::string("paged"), std::string("contiguous")}) {
      if (policy != "both" && policy != p) continue;
      SimConfig c = cfg;
      c.kv.policy = p;
      if (p == "contiguous") c.kv.prefix_caching = false;
      if (policy == "both" && !c.trace_csv.empty()) {
        auto dot = c.trace_csv.rfind('.');
        c.trace_csv = c.trace_csv.substr(0, dot) + "_" + p + c.trace_csv.substr(dot);
      }
      SimResult r = simulate(c);
      print_result(c, r);
      if (!summary.empty()) append_summary_csv(summary, c, r);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
