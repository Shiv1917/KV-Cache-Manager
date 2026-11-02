// kvkernel: decode-attention microbenchmark (one layer, TinyLlama head shape).
//
// Compares, at equal work:
//   paged      - shuffled physical blocks, per-block table lookup (what the engine runs)
//   contiguous - each sequence's blocks consecutive, kernel skips the table (layout baseline)
// for v1 (one CTA per (kv_head, row)) and v2 (context split into 512-token partitions).
// Reports microseconds per call and achieved DRAM bandwidth for the K/V reads.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include "common/args.h"
#include "cuda/cuda_utils.h"
#include "cuda/paged_attention.h"

using namespace kvp;

static std::vector<int> parse_list(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string x;
  while (std::getline(ss, x, ',')) v.push_back(std::stoi(x));
  return v;
}

int main(int argc, char** argv) {
  try {
    Args a(argc, argv);
    const auto ctxs = parse_list(a.str("ctx", "128,256,512,1024,2048"));
    const auto batches = parse_list(a.str("batch", "1,8,32,128"));
    const auto blocks = parse_list(a.str("block-size", "16"));
    const int heads = a.i32("heads", 32), kvh = a.i32("kv-heads", 4), hd = a.i32("head-dim", 64);
    const int iters = a.i32("iters", 100);
    const std::string out_csv = a.str("out", "");
    a.warn_unused();

    int dev = 0, clock_khz = 0, bus_bits = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
    double peak_gbs = 2.0 * clock_khz * 1e3 * (bus_bits / 8) / 1e9;
    if (peak_gbs <= 0) peak_gbs = 716.8;  // RTX 4080 spec
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("device %s, peak DRAM bandwidth %.1f GB/s\n", prop.name, peak_gbs);

    FILE* f = out_csv.empty() ? nullptr : std::fopen(out_csv.c_str(), "w");
    if (f) std::fprintf(f, "block_size,batch,ctx,variant,us,gbps,pct_peak,regs,smem_bytes,ctas_per_sm,occupancy_pct,ctas,waves\n");
    int max_threads_sm = 0;
    cudaDeviceGetAttribute(&max_threads_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    std::printf("%5s %6s %6s %-16s %10s %9s %7s %5s %7s %6s %6s %7s\n", "B", "batch", "ctx",
                "variant", "us/call", "GB/s", "%peak", "regs", "smem", "cta/SM", "occ%",
                "waves");

    std::mt19937 rng(0);
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    for (int B : blocks)
      for (int batch : batches)
        for (int ctx : ctxs) {
          const int mb = (ctx + B - 1) / B;
          const int nb = batch * mb;
          const size_t cache_elems = size_t(nb) * kvh * B * hd;
          DeviceBuffer<half> K(cache_elems), V(cache_elems), Q(size_t(batch) * heads * hd),
              O(size_t(batch) * heads * hd);
          CUDA_CHECK(cudaMemset(K.get(), 0, K.bytes()));
          CUDA_CHECK(cudaMemset(V.get(), 0, V.bytes()));
          CUDA_CHECK(cudaMemset(Q.get(), 0, Q.bytes()));
          std::vector<int> seq_tab(size_t(batch) * mb), shuf_tab(seq_tab.size());
          std::iota(seq_tab.begin(), seq_tab.end(), 0);
          shuf_tab = seq_tab;
          std::shuffle(shuf_tab.begin(), shuf_tab.end(), rng);
          std::vector<int> rows(batch), lens(batch, ctx);
          std::iota(rows.begin(), rows.end(), 0);
          DeviceBuffer<int> dseq(seq_tab.size()), dshuf(seq_tab.size()), drows(batch), dlens(batch);
          CUDA_CHECK(cudaMemcpy(dseq.get(), seq_tab.data(), seq_tab.size() * 4, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(dshuf.get(), shuf_tab.data(), shuf_tab.size() * 4, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(drows.get(), rows.data(), batch * 4, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(dlens.get(), lens.data(), batch * 4, cudaMemcpyHostToDevice));
          DeviceBuffer<float> ws(paged_attention_v2_workspace(batch, heads, hd, ctx));

          PagedAttnParams p;
          p.q = Q.get(); p.q_stride = heads * hd; p.out = O.get(); p.out_stride = heads * hd;
          p.k_cache = K.get(); p.v_cache = V.get(); p.max_blocks_per_seq = mb;
          p.row_seq = drows.get(); p.context_lens = dlens.get(); p.num_rows = batch;
          p.num_heads = heads; p.num_kv_heads = kvh; p.head_dim = hd; p.block_size = B;
          p.scale = 0.125f; p.max_context_len = ctx;

          struct Variant { const char* name; const int* table; bool contig; AttnVersion v; };
          const Variant vars[] = {
              {"paged-v1", dshuf.get(), false, AttnVersion::V1},
              {"contiguous-v1", dseq.get(), true, AttnVersion::V1},
              {"paged-v2", dshuf.get(), false, AttnVersion::V2},
              {"contiguous-v2", dseq.get(), true, AttnVersion::V2},
          };
          const double bytes = 2.0 * batch * double(ctx) * kvh * hd * sizeof(half);
          for (const auto& var : vars) {
            p.block_tables = var.table;
            p.assume_contiguous = var.contig;
            PagedAttnWorkspace w{ws.get(), ws.size()};
            for (int i = 0; i < 5; ++i) paged_attention(p, var.v, w, nullptr);
            CUDA_CHECK(cudaEventRecord(e0));
            for (int i = 0; i < iters; ++i) paged_attention(p, var.v, w, nullptr);
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            const double us = 1e3 * ms / iters;
            const double gbs = bytes / (us * 1e-6) / 1e9;
            const PagedAttnKernelInfo ki = paged_attention_kernel_info(p, var.v);
            const int smem = ki.static_smem + ki.dyn_smem;
            const double occ = 100.0 * ki.blocks_per_sm * ki.threads / max_threads_sm;
            const double waves = double(ki.ctas) / (ki.blocks_per_sm * prop.multiProcessorCount);
            std::printf("%5d %6d %6d %-16s %10.2f %9.1f %6.1f%% %5d %6dB %6d %5.0f%% %7.2f\n", B,
                        batch, ctx, var.name, us, gbs, 100 * gbs / peak_gbs, ki.regs, smem,
                        ki.blocks_per_sm, occ, waves);
            if (f)
              std::fprintf(f, "%d,%d,%d,%s,%.3f,%.2f,%.2f,%d,%d,%d,%.1f,%d,%.3f\n", B, batch, ctx,
                           var.name, us, gbs, 100 * gbs / peak_gbs, ki.regs, smem,
                           ki.blocks_per_sm, occ, ki.ctas, waves);
          }
        }
    if (f) std::fclose(f);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
