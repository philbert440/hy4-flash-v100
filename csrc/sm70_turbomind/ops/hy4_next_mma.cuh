// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace hy4_next {

#define HY4_NEXT_MMA(C, A0, A1, B0, B1)                               \
  asm volatile(                                                      \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "               \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "                \
      "{%0,%1,%2,%3,%4,%5,%6,%7};"                                    \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]),                 \
        "+f"(C[4]), "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                 \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// Four K warps and four independent quadpairs per warp. Use the same
// lane layout and reduction order as the F7 attention preparation tile.
template <int RowBlocks, int TileN, bool CoherentX = false>
__device__ __forceinline__ void project_tile(
    const half* x,
    const half* weight,
    int rows,
    int k_stride,
    int begin,
    int end,
    volatile float* red) {
  constexpr int kRows = RowBlocks * 8;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int wn = warp % (TileN / 8);
  const int wk = warp / (TileN / 8);
  const int qp = (lane >> 2) & 3;
  const int row_in_tile = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int steps = (end - begin) / 8;
  const int lo = begin + 8 * (steps * wk / 4);
  const int hi = begin + 8 * (steps * (wk + 1) / 4);
  const uint4 zero = make_uint4(0, 0, 0, 0);
  const half* w = weight + static_cast<int64_t>(row_in_tile) * k_stride;
  float acc[RowBlocks][8]{};
#pragma unroll 1
  for (int k = lo + qp * 8; k < hi + qp * 8; k += 32) {
    // The loop bound is warp-uniform. A masked quadpair still executes MMA.
    const bool live = k < hi;
    const uint4 b = live ? __ldcs(reinterpret_cast<const uint4*>(w + k)) : zero;
#pragma unroll
    for (int rb = 0; rb < RowBlocks; ++rb) {
      const int row = rb * 8 + row_in_tile;
      uint4 a = zero;
      if (live && row < rows) {
        const auto* addr = reinterpret_cast<const uint4*>(
            x + static_cast<int64_t>(row) * k_stride + k);
        // Grid synchronization does not invalidate SM70's per-SM L1.
        a = CoherentX ? __ldcg(addr) : __ldg(addr);
      }
      HY4_NEXT_MMA(acc[rb], a.x, a.y, b.x, b.y);
      HY4_NEXT_MMA(acc[rb], a.z, a.w, b.z, b.w);
    }
  }

  const int base = lane & ~12;
#pragma unroll
  for (int rb = 0; rb < RowBlocks; ++rb) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float value = __shfl_sync(0xffffffffu, acc[rb][i], base);
#pragma unroll
      for (int q = 1; q < 4; ++q) {
        value = __fadd_rn(value,
            __shfl_sync(0xffffffffu, acc[rb][i], base | q * 4));
      }
      if (qp == 0) {
        const int row = rb * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
        const int col = wn * 8 + (i & 1) + (lane & 2) + 4 * (i >> 2);
        red[(wk * kRows + row) * TileN + col] = value;
      }
    }
  }
  __syncthreads();
}

#undef HY4_NEXT_MMA

}  // namespace hy4_next
