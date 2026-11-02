// Small-kernel tests against straightforward CPU implementations.
#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "cuda/cuda_utils.h"
#include "cuda/kernels.h"

using namespace kvp;

template <class T>
static DeviceBuffer<T> upload(const std::vector<T>& v) {
  DeviceBuffer<T> d(v.size());
  CUDA_CHECK(cudaMemcpy(d.get(), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
template <class T>
static std::vector<T> download(const DeviceBuffer<T>& d) {
  std::vector<T> v(d.size());
  CUDA_CHECK(cudaMemcpy(v.data(), d.get(), d.bytes(), cudaMemcpyDeviceToHost));
  return v;
}

TEST(Kernels, RmsNormWithRowGather) {
  const int R = 5, H = 2048;
  std::mt19937 rng(1);
  std::normal_distribution<float> nd;
  std::vector<float> x(size_t(R) * H);
  std::vector<half> w(H);
  for (auto& v : x) v = nd(rng) * 3;
  for (auto& v : w) v = __float2half(1 + 0.1f * nd(rng));
  std::vector<int> rows = {4, 0, 2};
  auto dx = upload(x);
  auto dw = upload(w);
  auto dr = upload(rows);
  DeviceBuffer<half> out(rows.size() * H);
  rmsnorm(dx.get(), dr.get(), dw.get(), out.get(), 3, H, 1e-5f, nullptr);
  auto o = download(out);
  for (int i = 0; i < 3; ++i) {
    double ss = 0;
    for (int j = 0; j < H; ++j) ss += double(x[rows[i] * H + j]) * x[rows[i] * H + j];
    const double inv = 1.0 / std::sqrt(ss / H + 1e-5);
    for (int j = 0; j < H; ++j) {
      const float ref = __half2float(w[j]) * float(x[rows[i] * H + j] * inv);
      ASSERT_NEAR(__half2float(o[i * H + j]), ref, 2e-2 * std::max(1.f, std::fabs(ref)));
    }
  }
}

TEST(Kernels, RopeMatchesHfRotateHalf) {
  const int T = 3, NH = 4, NKV = 2, HD = 64, stride = (NH + 2 * NKV) * HD;
  std::mt19937 rng(2);
  std::normal_distribution<float> nd;
  std::vector<half> qkv(size_t(T) * stride);
  for (auto& v : qkv) v = __float2half(nd(rng));
  std::vector<int> pos = {0, 7, 1999};
  auto d = upload(qkv);
  auto dp = upload(pos);
  rope_inplace(d.get(), stride, dp.get(), T, NH, NKV, HD, 10000.f, nullptr);
  auto o = download(d);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < NH + 2 * NKV; ++h)
      for (int j = 0; j < HD; ++j) {
        const size_t base = size_t(t) * stride + h * HD;
        float ref;
        if (h >= NH + NKV) {
          ref = __half2float(qkv[base + j]);  // v heads untouched
        } else {
          const int i = j % (HD / 2);
          const double ang = pos[t] / std::pow(10000.0, 2.0 * i / HD);
          const float a = __half2float(qkv[base + i]), b = __half2float(qkv[base + i + HD / 2]);
          ref = j < HD / 2 ? float(a * std::cos(ang) - b * std::sin(ang))
                           : float(b * std::cos(ang) + a * std::sin(ang));
        }
        ASSERT_NEAR(__half2float(o[base + j]), ref, 1e-2) << t << " " << h << " " << j;
      }
}

TEST(Kernels, SiluMulAndArgmax) {
  const int T = 2, I = 300;
  std::mt19937 rng(3);
  std::normal_distribution<float> nd;
  std::vector<half> gu(size_t(T) * 2 * I);
  for (auto& v : gu) v = __float2half(nd(rng));
  auto d = upload(gu);
  DeviceBuffer<half> out(size_t(T) * I);
  silu_mul(d.get(), out.get(), T, I, nullptr);
  auto o = download(out);
  for (int t = 0; t < T; ++t)
    for (int i = 0; i < I; ++i) {
      const float g = __half2float(gu[t * 2 * I + i]), u = __half2float(gu[t * 2 * I + I + i]);
      ASSERT_NEAR(__half2float(o[t * I + i]), g / (1 + std::exp(-g)) * u, 1e-2);
    }

  const int R = 3, V = 32000;
  std::vector<float> logits(size_t(R) * V);
  for (auto& v : logits) v = nd(rng);
  logits[0 * V + 31999] = 50;
  logits[1 * V + 7] = 50;
  logits[1 * V + 9] = 50;  // tie -> lowest index
  logits[2 * V + 12345] = 50;
  auto dl = upload(logits);
  DeviceBuffer<int> am(R);
  argmax_rows(dl.get(), am.get(), R, V, nullptr);
  auto a = download(am);
  EXPECT_EQ(a[0], 31999);
  EXPECT_EQ(a[1], 7);
  EXPECT_EQ(a[2], 12345);
}
