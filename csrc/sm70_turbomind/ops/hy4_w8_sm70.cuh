// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace hy4_w8 {

// Each matrix stores [N/8, K/16, 8, 16] biased int8 bytes, followed
// by [N/8, ceil(K/group_size), 8] FP16 scales. Both regions pad N to eight.
__host__ __device__ constexpr int groups(int k, int group_size = 128) {
  return (k + group_size - 1) / group_size;
}
__host__ __device__ constexpr int padded_n(int n) { return (n + 7) / 8 * 8; }
__host__ __device__ constexpr int matrix_bytes(int n, int k, int group_size = 128) {
  return padded_n(n) * (k + 2 * groups(k, group_size));
}

__device__ __forceinline__ int acc_row(int i, int lane) {
  return (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
}
__device__ __forceinline__ int acc_col(int i, int lane) {
  return (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
}

__device__ __forceinline__ void mma(
    float (&c)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]),
        "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void dequant(
    uint32_t w, uint32_t scale, uint32_t& d0, uint32_t& d1) {
  const uint32_t a = __byte_perm(w, 0x64646464u, 0x4140);
  const uint32_t b = __byte_perm(w, 0x64646464u, 0x4342);
  asm("sub.f16x2 %0, %1, %2;" : "=r"(d0) : "r"(a), "r"(0x64806480));
  asm("sub.f16x2 %0, %1, %2;" : "=r"(d1) : "r"(b), "r"(0x64806480));
  asm("mul.f16x2 %0, %1, %2;" : "=r"(d0) : "r"(d0), "r"(scale));
  asm("mul.f16x2 %0, %1, %2;" : "=r"(d1) : "r"(d1), "r"(scale));
}

// Each of the four quadpairs can read a distinct 8-column tile, as in
// the NATIVE benchmark, or a distinct range of K for the same output tile.
template <int RB, bool CoherentX, bool WideActivation = true, int GroupSize = 128>
__device__ __forceinline__ void warp_project(
    const half* x, const uint8_t* packed, int m, int n, int k, int tile,
    int begin, int end, int steps, float (&acc)[RB][8]) {
  const int lane = threadIdx.x & 31;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const auto* weight = reinterpret_cast<const uint4*>(packed) + tile * (k / 2) + r;
  const auto* scales = reinterpret_cast<const uint16_t*>(packed + padded_n(n) * k)
      + tile * groups(k, GroupSize) * 8 + r;

  uint4 current = begin < end ? __ldcs(weight + begin * 8) : make_uint4(0, 0, 0, 0);
  uint32_t scale = 0;
#pragma unroll 1
  for (int step = 0; step < steps; ++step) {
    const int chunk = begin + step;
    const uint4 value = current;
    current = chunk + 1 < end ? __ldcs(weight + (chunk + 1) * 8) : make_uint4(0, 0, 0, 0);
    constexpr int group_chunks = GroupSize / 16;
    if (chunk < end && (step == 0 || chunk % group_chunks == 0)) {
      const uint32_t s = __ldg(scales + chunk / group_chunks * 8);
      scale = s | (s << 16);
    }
    uint2 activation[RB][4] = {};
    if constexpr (WideActivation) {
#pragma unroll
      for (int rb = 0; rb < RB; ++rb) {
        if (chunk < end && rb * 8 + r < m) {
          const auto* in = reinterpret_cast<const uint4*>(
              x + (rb * 8 + r) * k + chunk * 16);
          const uint4 lo = CoherentX ? __ldcg(in) : __ldg(in);
          const uint4 hi = CoherentX ? __ldcg(in + 1) : __ldg(in + 1);
          activation[rb][0] = make_uint2(lo.x, lo.y);
          activation[rb][1] = make_uint2(lo.z, lo.w);
          activation[rb][2] = make_uint2(hi.x, hi.y);
          activation[rb][3] = make_uint2(hi.z, hi.w);
        }
      }
    }
    const uint32_t words[4] = {value.x, value.y, value.z, value.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t d0, d1;
      dequant(words[j], scale, d0, d1);
#pragma unroll
      for (int rb = 0; rb < RB; ++rb) {
        uint2 a = make_uint2(0, 0);
        if constexpr (WideActivation) {
          a = activation[rb][j];
        } else if (chunk < end && rb * 8 + r < m) {
          const auto* in = reinterpret_cast<const uint2*>(
              x + (rb * 8 + r) * k + chunk * 16 + j * 4);
          a = CoherentX ? __ldcg(in) : __ldg(in);
        }
        mma(acc[rb], a.x, a.y, d0, d1);
      }
    }
  }
}

// Mirror the existing cooperative prep/merge epilogue layout.
template <int RB, bool CoherentX, int WarpsK, int TilesN = 2>
__device__ __forceinline__ void slice16(
    const half* x, const uint8_t* packed, int m, int n, int k, int col,
    int k_begin, int k_end, volatile float* red) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int quad = (lane >> 2) & 3;
  const int split = (warp / TilesN) * 4 + quad;
  const int count = (k_end - k_begin) / 16;
  const int begin = k_begin / 16 + split * count / (4 * WarpsK);
  const int end = k_begin / 16 + (split + 1) * count / (4 * WarpsK);
  // All quadpairs participate in each collective MMA. Give every
  // quadpair the same loop trip count, masking the final partial step.
  const int steps = (count + 4 * WarpsK - 1) / (4 * WarpsK);
  float acc[RB][8] = {};
  warp_project<RB, CoherentX>(x, packed, m, n, k, col / 8 + warp % TilesN,
                             begin, end, steps, acc);
  const int base_lane = lane & ~12;
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float v = __shfl_sync(0xffffffffu, acc[rb][i], base_lane);
#pragma unroll
      for (int q = 1; q < 4; ++q) {
        v += __shfl_sync(0xffffffffu, acc[rb][i], base_lane | (q << 2));
      }
      if (quad == 0) {
        const int row = rb * 8 + acc_row(i, lane);
        const int c = (warp % TilesN) * 8 + acc_col(i, lane);
        red[((warp / TilesN) * RB * 8 + row) * (TilesN * 8) + c] = v;
      }
    }
  }
  __syncthreads();
}

}  // namespace hy4_w8
