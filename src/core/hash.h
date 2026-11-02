// XXH64 (reference algorithm, scalar) used for prefix-cache block hashing.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace kvp {

namespace detail {
constexpr uint64_t P1 = 0x9E3779B185EBCA87ULL;
constexpr uint64_t P2 = 0xC2B2AE3D27D4EB4FULL;
constexpr uint64_t P3 = 0x165667B19E3779F9ULL;
constexpr uint64_t P4 = 0x85EBCA77C2B2AE63ULL;
constexpr uint64_t P5 = 0x27D4EB2F165667C5ULL;
inline uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
inline uint64_t round(uint64_t acc, uint64_t in) {
  acc += in * P2;
  acc = rotl(acc, 31);
  return acc * P1;
}
inline uint64_t merge(uint64_t acc, uint64_t v) {
  acc ^= round(0, v);
  return acc * P1 + P4;
}
inline uint64_t rd64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
inline uint32_t rd32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
}  // namespace detail

inline uint64_t xxh64(const void* data, size_t len, uint64_t seed) {
  using namespace detail;
  const uint8_t* p = static_cast<const uint8_t*>(data);
  const uint8_t* end = p + len;
  uint64_t h;
  if (len >= 32) {
    uint64_t v1 = seed + P1 + P2, v2 = seed + P2, v3 = seed, v4 = seed - P1;
    const uint8_t* limit = end - 32;
    do {
      v1 = round(v1, rd64(p)); p += 8;
      v2 = round(v2, rd64(p)); p += 8;
      v3 = round(v3, rd64(p)); p += 8;
      v4 = round(v4, rd64(p)); p += 8;
    } while (p <= limit);
    h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
    h = merge(h, v1); h = merge(h, v2); h = merge(h, v3); h = merge(h, v4);
  } else {
    h = seed + P5;
  }
  h += static_cast<uint64_t>(len);
  while (p + 8 <= end) {
    h ^= round(0, rd64(p));
    h = rotl(h, 27) * P1 + P4;
    p += 8;
  }
  if (p + 4 <= end) {
    h ^= static_cast<uint64_t>(rd32(p)) * P1;
    h = rotl(h, 23) * P2 + P3;
    p += 4;
  }
  while (p < end) {
    h ^= static_cast<uint64_t>(*p) * P5;
    h = rotl(h, 11) * P1;
    ++p;
  }
  h ^= h >> 33; h *= P2;
  h ^= h >> 29; h *= P3;
  h ^= h >> 32;
  return h;
}

// h_i = hash(h_{i-1}, tokens of block i)
inline uint64_t block_hash(uint64_t parent, const int32_t* tokens, int n) {
  return xxh64(tokens, sizeof(int32_t) * static_cast<size_t>(n), parent);
}

}  // namespace kvp
