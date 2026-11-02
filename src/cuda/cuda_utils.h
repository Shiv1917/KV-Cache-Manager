#pragma once
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(expr)                                                                    \
  do {                                                                                      \
    cudaError_t _e = (expr);                                                                \
    if (_e != cudaSuccess)                                                                  \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_e) +       \
                               " at " + __FILE__ + ":" + std::to_string(__LINE__) + " (" + \
                               #expr ")");                                                  \
  } while (0)

#define CUBLAS_CHECK(expr)                                                                \
  do {                                                                                    \
    cublasStatus_t _s = (expr);                                                           \
    if (_s != CUBLAS_STATUS_SUCCESS)                                                      \
      throw std::runtime_error(std::string("cuBLAS error ") + std::to_string((int)_s) +   \
                               " at " + __FILE__ + ":" + std::to_string(__LINE__));       \
  } while (0)

namespace kvp {

// Minimal owning device buffer.
template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t n) { resize(n); }
  ~DeviceBuffer() { reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) { reset(); p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0; }
    return *this;
  }
  void resize(size_t n) {
    reset();
    if (n) CUDA_CHECK(cudaMalloc(&p_, n * sizeof(T)));
    n_ = n;
  }
  void reset() {
    if (p_) cudaFree(p_);
    p_ = nullptr;
    n_ = 0;
  }
  T* get() const { return p_; }
  size_t size() const { return n_; }
  size_t bytes() const { return n_ * sizeof(T); }

 private:
  T* p_ = nullptr;
  size_t n_ = 0;
};

}  // namespace kvp
