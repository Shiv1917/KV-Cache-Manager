#include "model/weights.h"

#include <cstdio>
#include <cstring>
#include <memory>

namespace kvp {

namespace {
constexpr size_t kHeaderBytes = 64;

struct FileCloser {
  void operator()(FILE* f) const { if (f) std::fclose(f); }
};
using FilePtr = std::unique_ptr<FILE, FileCloser>;

FilePtr open_or_throw(const std::string& path) {
  FilePtr f(std::fopen(path.c_str(), "rb"));
  if (!f) throw std::runtime_error("cannot open weights file: " + path);
  return f;
}
}  // namespace

LlamaConfig LlamaWeights::read_config(const std::string& path) {
  FilePtr f = open_or_throw(path);
  unsigned char hdr[kHeaderBytes];
  if (std::fread(hdr, 1, kHeaderBytes, f.get()) != kHeaderBytes)
    throw std::runtime_error("weights file too short");
  if (std::memcmp(hdr, "KVPW", 4) != 0) throw std::runtime_error("bad weights magic");
  int32_t ints[9];
  float floats[2];
  std::memcpy(ints, hdr + 4, sizeof(ints));
  std::memcpy(floats, hdr + 4 + sizeof(ints), sizeof(floats));
  if (ints[0] != 1) throw std::runtime_error("unsupported weights version");
  LlamaConfig c;
  c.vocab = ints[1]; c.hidden = ints[2]; c.inter = ints[3]; c.layers = ints[4];
  c.heads = ints[5]; c.kv_heads = ints[6]; c.head_dim = ints[7]; c.max_pos = ints[8];
  c.rope_theta = floats[0]; c.rms_eps = floats[1];
  return c;
}

LlamaWeights::LlamaWeights(const std::string& path) : cfg_(read_config(path)) {
  const LlamaConfig& c = cfg_;
  const size_t H = c.hidden, I = c.inter, V = c.vocab;
  const size_t per_layer = H + size_t(c.qkv_dim()) * H + H * H + H + 2 * I * H + H * I;
  const size_t total = V * H + c.layers * per_layer + H + V * H;
  buf_.resize(total);

  // Stream the file to the GPU in chunks through a pinned staging buffer.
  FilePtr f = open_or_throw(path);
  std::fseek(f.get(), kHeaderBytes, SEEK_SET);
  constexpr size_t kChunk = 64 << 20;
  void* staging = nullptr;
  CUDA_CHECK(cudaMallocHost(&staging, kChunk));
  size_t done = 0;
  const size_t total_bytes = total * sizeof(half);
  while (done < total_bytes) {
    const size_t n = std::min(kChunk, total_bytes - done);
    if (std::fread(staging, 1, n, f.get()) != n) {
      cudaFreeHost(staging);
      throw std::runtime_error("weights file truncated");
    }
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<char*>(buf_.get()) + done, staging, n,
                          cudaMemcpyHostToDevice));
    done += n;
  }
  cudaFreeHost(staging);

  const half* p = buf_.get();
  auto take = [&](size_t n) { const half* r = p; p += n; return r; };
  embed_ = take(V * H);
  layers_.resize(c.layers);
  for (auto& L : layers_) {
    L.attn_norm = take(H);
    L.wqkv = take(size_t(c.qkv_dim()) * H);
    L.wo = take(H * H);
    L.ffn_norm = take(H);
    L.w_gate_up = take(2 * I * H);
    L.w_down = take(H * I);
  }
  final_norm_ = take(H);
  lm_head_ = take(V * H);
}

}  // namespace kvp
