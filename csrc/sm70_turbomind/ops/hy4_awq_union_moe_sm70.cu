// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Hy4-flash TP4 native-g32 AWQ grouped-union MoE kernel for Tesla V100 (sm_70).
//
// Adapted from Leonccaa/ReVolta awq_grouped_decode_sm70.cu (Apache-2.0) and
// awq_qpn_m1_sm70.cu, whose quadpair-N m8n8k4 dataflow is derived from
// dnv2003/v100-skinny (MIT, see adjacent LICENSE.v100-skinny).
//
// Shape contract (Hy4-flash TP4 per-rank routed experts):
//   E = 512 experts, top_k = 16, group_size = 32
//   W13: K = 1536, N = 768 (non-interleaved: [0..384) gate, [384..768) up)
//   W2:  K = 384,  N = 1536
//   SwiGLU with optional clamp (swiglu_limit = 10.0 for Hy4-flash)
//   M = 1..32 tokens (routes = M * 16 <= 512)
//
// Reads the existing prepared TurboMind banks (w13_tm_weight, w13_tm_scales,
// w2_tm_weight, w2_tm_scales) in place with zero repacking and zero second
// weight copy. Supports both 4-byte scale+bias metadata (int32, default) and
// compact 3-byte metadata (uint8).
//
// Four CUDA-graph-capturable launches per MoE layer, zero atomics, zero host
// synchronization, and bit-for-bit row invariance across M = 1..32 for fixed
// (split, w2_mode):
//   1. plan: atomic-free warp-0 __match_any_sync + 512-thread exclusive scan
//      forming the union of active experts and row packs.
//   2. W13 + SwiGLU (+ clamp): grid (12 gate+up tile pairs, max_packs=routes).
//      Inactive pack CTAs exit immediately on device. Each CTA reads one
//      expert's gate tile and up tile once for all rows in its pack, shares
//      activation loads across gate and up, accumulates in FP32 with Volta
//      mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32, reduces Split warps
//      in fixed order in shared memory, rounds gate/up to FP16, applies the
//      Hy4 SwiGLU clamp (gate <= limit, -limit <= up <= limit) and SiLU in
//      FP32 -> FP16, and writes intermediate[route, 384].
//   3. W2: grid (48/TilesPerCTA, max_packs=routes). Reads each active
//      expert's W2 weights once for all rows in its pack, accumulates in FP32
//      across all 12 g32 groups of K2=384, and writes per-route FP16 outputs
//      to route_out[route, 1536] (L2-resident).
//   4. weighted_reduce: grid (6, tokens), 64 threads/CTA (uint2 = 4 cols/thr).
//      Accumulates the 16 route outputs per token in fixed router slot order
//      0..15 with FP32 fmaf and rounds once to FP16. Invalid expert ids
//      (< 0 or >= 512) contribute 0 without zeroing route_out.

#include <torch/all.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <algorithm>
#include <cstdint>

#include "hy4_w8_sm70.cuh"

namespace {

constexpr int kK13 = 1536;          // routed_expert_hidden_size
constexpr int kInter = 384;         // moe_intermediate_size / TP4
constexpr int kN13 = kInter * 2;    // 768 (non-interleaved gate | up)
constexpr int kK2 = kInter;         // 384
constexpr int kN2 = kK13;           // 1536
constexpr int kExperts = 512;
constexpr int kTopK = 16;
constexpr int kMaxTokens = 32;
constexpr int kMaxRoutes = kMaxTokens * kTopK;       // 512
constexpr int kMaxRowBlocks = (kMaxTokens + 7) / 8;  // 4
constexpr int kRowCap = kMaxRowBlocks * 8;           // 32
constexpr int kTilesInter = kInter / 32;             // 12
constexpr int kTiles2 = kN2 / 32;                    // 48
static_assert(kMaxRoutes <= kExperts, "plan kernel loads one route per thread");

// Fixed-size int32 plan buffer layout (graph capturable):
//   [0]                    number of active packs
//   [kPlanExperts + p]     expert id of pack p
//   [kPlanCounts + p]      number of rows in pack p (1..cap)
//   [kPlanStarts + p]      first index of pack p in kPlanRows
//   [kPlanRows + i]        route index (token * 16 + slot)
constexpr int kPlanExperts = 8;
constexpr int kPlanCounts = kPlanExperts + kMaxRoutes;
constexpr int kPlanStarts = kPlanCounts + kMaxRoutes;
constexpr int kPlanRows = kPlanStarts + kMaxRoutes;
constexpr int kPlanInts = kPlanRows + kMaxRoutes;  // 2056

// Metadata readers (identical contract to awq_grouped_decode_sm70.cu):
//   Format 2: 4-byte scale+bias int32 [E, K/32, N] (Hy4 default)
//   Format 1: 3-byte compact uint8 [E, K/32, N, 3] warp-cooperative
//   Format 0: 3-byte compact uint8 [E, K/32, N, 3] scalar
template <int Format>
__device__ __forceinline__ uint32_t read_awq_stats(const uint8_t* stats,
                                                   int group, int tile, int col,
                                                   int n) {
  uint32_t bits = 0;
  if constexpr (Format == 2) {
    return __ldg(reinterpret_cast<const uint32_t*>(stats) +
                 static_cast<size_t>(group) * n + tile * 32 + col);
  } else if constexpr (Format == 1) {
    const int lane = threadIdx.x & 31;
    const auto* words = reinterpret_cast<const uint32_t*>(
        stats + (static_cast<size_t>(group) * n + tile * 32) * 3);
    const uint32_t word = lane < 24 ? __ldg(words + lane) : 0;
    const int byte = col * 3;
    const int shift = (byte & 3) * 8;
    const uint32_t lo = __shfl_sync(0xffffffffu, word, byte >> 2);
    const uint32_t hi =
        __shfl_sync(0xffffffffu, word, (byte >> 2) + (shift > 8));
    bits = lo >> shift;
    if (shift > 8) bits |= hi << (32 - shift);
  } else {
    const auto* record =
        stats + (static_cast<size_t>(group) * n + tile * 32 + col) * 3;
    bits = static_cast<uint32_t>(__ldg(record)) |
           (static_cast<uint32_t>(__ldg(record + 1)) << 8) |
           (static_cast<uint32_t>(__ldg(record + 2)) << 16);
  }
  const half scale = __ushort_as_half(static_cast<uint16_t>(bits));
  const half zero = __int2half_rn(static_cast<int>((bits >> 16) & 0xff));
  const half bias = __hmul(__hneg(zero), scale);
  return (bits & 0xffffu) |
         (static_cast<uint32_t>(__half_as_ushort(bias)) << 16);
}

// Exact TurboMind U4 -> FP16 dequantization with FP16 FMA bias boundary.
__device__ __forceinline__ void dequant_awq_u4x8(uint32_t packed,
                                                 uint32_t stats,
                                                 half2* decoded) {
  uint32_t h[4];
  const uint32_t upper = __byte_perm(packed, 0, 0x4321);
  constexpr uint32_t lut = (0xf0 & 0xcc) | 0xaa;
  constexpr uint32_t bottom_mask = 0x000f000f;
  constexpr uint32_t top_mask = 0x00f000f0;
  constexpr uint32_t magic0 = 0x64006400;
  constexpr uint32_t magic1 = 0x54005400;
  asm("lop3.b32 %0, %1, %2, %3, %4;"
      : "=r"(h[0])
      : "r"(packed), "n"(bottom_mask), "n"(magic0), "n"(lut));
  asm("lop3.b32 %0, %1, %2, %3, %4;"
      : "=r"(h[1])
      : "r"(packed), "n"(top_mask), "n"(magic1), "n"(lut));
  asm("lop3.b32 %0, %1, %2, %3, %4;"
      : "=r"(h[2])
      : "r"(upper), "n"(bottom_mask), "n"(magic0), "n"(lut));
  asm("lop3.b32 %0, %1, %2, %3, %4;"
      : "=r"(h[3])
      : "r"(upper), "n"(top_mask), "n"(magic1), "n"(lut));
  asm("sub.f16x2 %0, %1, %2;" : "=r"(h[0]) : "r"(h[0]), "r"(magic0));
  asm("sub.f16x2 %0, %1, %2;" : "=r"(h[1]) : "r"(h[1]), "r"(magic1));
  asm("sub.f16x2 %0, %1, %2;" : "=r"(h[2]) : "r"(h[2]), "r"(magic0));
  asm("sub.f16x2 %0, %1, %2;" : "=r"(h[3]) : "r"(h[3]), "r"(magic1));
  const half scale = __ushort_as_half(static_cast<uint16_t>(stats));
  const half bias = __ushort_as_half(static_cast<uint16_t>(stats >> 16));
  const half2 s2 = __halves2half2(scale, scale);
  const half2 b2 = __halves2half2(bias, bias);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    decoded[i] = __hfma2(*reinterpret_cast<const half2*>(&h[i]), s2, b2);
  }
}

#define HY4_MMA(C, A0, A1, B0, B1)                                  \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};"                                  \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// Single-tile K16 step across active row blocks (used by W2).
template <int RB>
__device__ __forceinline__ void mma_row_blocks(float (&accum)[RB][8],
                                               const half* const (&a)[RB],
                                               const bool (&on)[RB], int nrb,
                                               int k16, const half2* decoded) {
  const auto* b = reinterpret_cast<const uint32_t*>(decoded);
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    if (rb < nrb) {
      uint4 lo = make_uint4(0, 0, 0, 0);
      uint4 hi = make_uint4(0, 0, 0, 0);
      if (on[rb]) {
        lo = *reinterpret_cast<const uint4*>(a[rb] + k16 * 16);
        hi = *reinterpret_cast<const uint4*>(a[rb] + k16 * 16 + 8);
      }
      HY4_MMA(accum[rb], lo.x, lo.y, b[0], b[1]);
      HY4_MMA(accum[rb], lo.z, lo.w, b[2], b[3]);
      HY4_MMA(accum[rb], hi.x, hi.y, b[4], b[5]);
      HY4_MMA(accum[rb], hi.z, hi.w, b[6], b[7]);
    }
  }
}

// Dual-tile K16 step sharing each activation row load across gate and up (W13).
template <int RB>
__device__ __forceinline__ void mma_row_blocks_dual(
    float (&acc_g)[RB][8], float (&acc_u)[RB][8], const half* const (&a)[RB],
    const bool (&on)[RB], int nrb, int k16, const half2* dec_g,
    const half2* dec_u) {
  const auto* bg = reinterpret_cast<const uint32_t*>(dec_g);
  const auto* bu = reinterpret_cast<const uint32_t*>(dec_u);
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    if (rb < nrb) {
      uint4 lo = make_uint4(0, 0, 0, 0);
      uint4 hi = make_uint4(0, 0, 0, 0);
      if (on[rb]) {
        lo = *reinterpret_cast<const uint4*>(a[rb] + k16 * 16);
        hi = *reinterpret_cast<const uint4*>(a[rb] + k16 * 16 + 8);
      }
      HY4_MMA(acc_g[rb], lo.x, lo.y, bg[0], bg[1]);
      HY4_MMA(acc_g[rb], lo.z, lo.w, bg[2], bg[3]);
      HY4_MMA(acc_g[rb], hi.x, hi.y, bg[4], bg[5]);
      HY4_MMA(acc_g[rb], hi.z, hi.w, bg[6], bg[7]);
      HY4_MMA(acc_u[rb], lo.x, lo.y, bu[0], bu[1]);
      HY4_MMA(acc_u[rb], lo.z, lo.w, bu[2], bu[3]);
      HY4_MMA(acc_u[rb], hi.x, hi.y, bu[4], bu[5]);
      HY4_MMA(acc_u[rb], hi.z, hi.w, bu[6], bu[7]);
    }
  }
}

__device__ __forceinline__ int acc_row(int i, int lane) {
  return (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
}

__device__ __forceinline__ int acc_col(int i, int lane) {
  return (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
}

// Kernel 1: Deterministic, atomic-free expert-union plan builder.
__global__ void __launch_bounds__(kExperts)
    hy4_grouped_plan_kernel(const int32_t* __restrict__ ids, int routes,
                            int cap, int32_t* __restrict__ plan) {
  __shared__ int s_id[kMaxRoutes];
  __shared__ int s_ordinal[kMaxRoutes];
  __shared__ int s_count[kExperts];
  __shared__ int s_first_row[kExperts];
  __shared__ int s_warp_packs[kExperts / 32];
  __shared__ int s_warp_rows[kExperts / 32];
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int warp = t >> 5;
  s_count[t] = 0;
  if (t < kMaxRoutes) {
    int id = t < routes ? ids[t] : -1;
    if (id < 0 || id >= kExperts) id = -1;
    s_id[t] = id;
  }
  __syncthreads();
  if (warp == 0) {
    for (int base = 0; base < routes; base += 32) {
      const int r = base + lane;
      const int id = r < routes ? s_id[r] : -1;
      const unsigned peers = __match_any_sync(0xffffffffu, id);
      const unsigned lower = peers & ((1u << lane) - 1u);
      int ordinal = -1;
      if (id >= 0) ordinal = s_count[id] + __popc(lower);
      __syncwarp();
      if (id >= 0 && lower == 0) s_count[id] += __popc(peers);
      if (r < routes) s_ordinal[r] = ordinal;
      __syncwarp();
    }
  }
  __syncthreads();
  const int count = s_count[t];
  const int npacks = (count + cap - 1) / cap;
  int packs_incl = npacks;
  int rows_incl = count;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int packs = __shfl_up_sync(0xffffffffu, packs_incl, offset);
    const int rows = __shfl_up_sync(0xffffffffu, rows_incl, offset);
    if (lane >= offset) {
      packs_incl += packs;
      rows_incl += rows;
    }
  }
  if (lane == 31) {
    s_warp_packs[warp] = packs_incl;
    s_warp_rows[warp] = rows_incl;
  }
  __syncthreads();
  int packs_base = 0;
  int rows_base = 0;
  for (int w = 0; w < warp; ++w) {
    packs_base += s_warp_packs[w];
    rows_base += s_warp_rows[w];
  }
  const int first_pack = packs_base + packs_incl - npacks;
  const int first_row = rows_base + rows_incl - count;
  s_first_row[t] = first_row;
  for (int p = 0; p < npacks; ++p) {
    plan[kPlanExperts + first_pack + p] = t;
    plan[kPlanCounts + first_pack + p] = min(cap, count - p * cap);
    plan[kPlanStarts + first_pack + p] = first_row + p * cap;
  }
  if (t == kExperts - 1) plan[0] = packs_base + packs_incl;
  __syncthreads();
  if (t < routes) {
    const int id = s_id[t];
    if (id >= 0) plan[kPlanRows + s_first_row[id] + s_ordinal[t]] = t;
  }
}

// Kernel 2: Fused W13 (gate + up) + SwiGLU (+ optional clamp).
// Grid: (12, max_packs=routes). Each CTA owns 32 gate cols (tile) and 32 up
// cols (tile + 12) for one active expert pack.
template <int Format, int Split, int RB>
__global__ void __launch_bounds__(Split * 32)
    hy4_grouped_w13_swiglu_kernel(const half* __restrict__ x,
                                  const uint32_t* __restrict__ weights,
                                  const uint8_t* __restrict__ metadata,
                                  const int32_t* __restrict__ plan,
                                  half* __restrict__ intermediate,
                                  float swiglu_limit) {
  // Partial buffer for 1 row block (8 rows) x 64 cols (32 gate + 32 up):
  // Split * 8 * 64 * 4 bytes (8 KB at Split=4, 12 KB at 6, 16 KB at 8).
  extern __shared__ float partial[];
  constexpr int kSteps = (kK13 / 16) / Split;  // 96 / Split
  static_assert((kK13 / 16) % Split == 0, "Split must divide 96");
  static_assert(kSteps % 2 == 0, "warps must start on an even K16 step");
  static_assert(RB >= 1 && RB <= kMaxRowBlocks, "unsupported row blocks");

  const int pack = blockIdx.y;
  if (pack >= __ldg(plan)) return;

  const int expert = __ldg(plan + kPlanExperts + pack);
  const int count = __ldg(plan + kPlanCounts + pack);
  const int32_t* rows = plan + kPlanRows + __ldg(plan + kPlanStarts + pack);
  const int nrb = (count + 7) >> 3;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;  // 0..11
  const int quadpair = (lane >> 2) & 3;
  const int a_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int packed_col = quadpair * 8 + a_row;

  const half* a[RB];
  bool on[RB];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    const int r = rb * 8 + a_row;
    on[rb] = r < count;
    const int route = on[rb] ? __ldg(rows + r) : 0;
    a[rb] = x + static_cast<size_t>(route / kTopK) * kK13;
  }

  const uint32_t* w = weights + static_cast<size_t>(expert) * (kK13 * kN13 / 8);
  constexpr size_t stats_bytes =
      static_cast<size_t>(kK13 / 32) * kN13 * (Format == 2 ? 4 : 3);
  const uint8_t* st = metadata + static_cast<size_t>(expert) * stats_bytes;

  float acc_g[RB][8];
  float acc_u[RB][8];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc_g[rb][i] = 0.f;
      acc_u[rb][i] = 0.f;
    }
  }

  constexpr size_t tile_stride_w = static_cast<size_t>(kK13 / 8) * 32;
  const uint32_t* wg = w + static_cast<size_t>(tile) * tile_stride_w;
  const uint32_t* wu =
      w + static_cast<size_t>(tile + kTilesInter) * tile_stride_w;

#pragma unroll 2
  for (int s = 0; s < kSteps; s += 2) {
    const int group = warp * kSteps + s;  // even K16 step
    const size_t off = static_cast<size_t>(group * 2) * 32 + packed_col;

    const uint32_t pg0 = __ldcs(wg + off);
    const uint32_t pg1 = __ldcs(wg + off + 32);
    const uint32_t pg2 = __ldcs(wg + off + 64);
    const uint32_t pg3 = __ldcs(wg + off + 96);
    const uint32_t st_g =
        read_awq_stats<Format>(st, group >> 1, tile, packed_col, kN13);

    const uint32_t pu0 = __ldcs(wu + off);
    const uint32_t pu1 = __ldcs(wu + off + 32);
    const uint32_t pu2 = __ldcs(wu + off + 64);
    const uint32_t pu3 = __ldcs(wu + off + 96);
    const uint32_t st_u = read_awq_stats<Format>(
        st, group >> 1, tile + kTilesInter, packed_col, kN13);

    half2 dg0[8], dg1[8], du0[8], du1[8];
    dequant_awq_u4x8(pg0, st_g, dg0);
    dequant_awq_u4x8(pg1, st_g, dg0 + 4);
    dequant_awq_u4x8(pu0, st_u, du0);
    dequant_awq_u4x8(pu1, st_u, du0 + 4);
    mma_row_blocks_dual<RB>(acc_g, acc_u, a, on, nrb, group, dg0, du0);

    dequant_awq_u4x8(pg2, st_g, dg1);
    dequant_awq_u4x8(pg3, st_g, dg1 + 4);
    dequant_awq_u4x8(pu2, st_u, du1);
    dequant_awq_u4x8(pu3, st_u, du1 + 4);
    mma_row_blocks_dual<RB>(acc_g, acc_u, a, on, nrb, group + 1, dg1, du1);
  }

  const bool has_clamp = swiglu_limit > 0.f;
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    if (rb < nrb) {
      if (rb > 0) __syncthreads();
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int r = acc_row(i, lane);
        const int c = quadpair * 8 + acc_col(i, lane);
        const int base = (warp * 8 + r) * 64 + c;
        partial[base] = acc_g[rb][i];
        partial[base + 32] = acc_u[rb][i];
      }
      __syncthreads();
      const int row0 = rb * 8;
      const int rb_rows = min(count - row0, 8);
      for (int idx = threadIdx.x; idx < rb_rows * 32; idx += blockDim.x) {
        const int r = idx >> 5;
        const int c = idx & 31;
        float gate_sum = 0.f;
        float up_sum = 0.f;
#pragma unroll
        for (int s = 0; s < Split; ++s) {
          const float* p = partial + (s * 8 + r) * 64;
          gate_sum += p[c];
          up_sum += p[32 + c];
        }
        half gate_h = __float2half(gate_sum);
        half up_h = __float2half(up_sum);
        float g = __half2float(gate_h);
        float u = __half2float(up_h);
        if (has_clamp) {
          g = fminf(g, swiglu_limit);
          u = fmaxf(fminf(u, swiglu_limit), -swiglu_limit);
          gate_h = __float2half(g);
          up_h = __float2half(u);
          g = __half2float(gate_h);
        }
        const half silu_h = __float2half(g / (1.0f + expf(-g)));
        const int route = __ldg(rows + row0 + r);
        intermediate[static_cast<size_t>(route) * kInter + tile * 32 + c] =
            __hmul(silu_h, up_h);
      }
    }
  }
}

// Kernel 3a: W2 SplitK=1 (zero SMEM, zero barriers, TilesPerCTA warps per CTA).
// Grid: (48 / TilesPerCTA, max_packs=routes).
template <int Format, int RB, int TilesPerCTA>
__global__ void __launch_bounds__(TilesPerCTA * 32)
    hy4_grouped_w2_nosplit_kernel(const half* __restrict__ intermediate,
                                  const uint32_t* __restrict__ weights,
                                  const uint8_t* __restrict__ metadata,
                                  const int32_t* __restrict__ plan,
                                  half* __restrict__ route_out) {
  static_assert(kTiles2 % TilesPerCTA == 0, "TilesPerCTA must divide 48");
  const int pack = blockIdx.y;
  if (pack >= __ldg(plan)) return;

  const int expert = __ldg(plan + kPlanExperts + pack);
  const int count = __ldg(plan + kPlanCounts + pack);
  const int32_t* rows = plan + kPlanRows + __ldg(plan + kPlanStarts + pack);
  const int nrb = (count + 7) >> 3;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x * TilesPerCTA + warp;
  const int quadpair = (lane >> 2) & 3;
  const int a_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int packed_col = quadpair * 8 + a_row;

  const half* a[RB];
  bool on[RB];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    const int r = rb * 8 + a_row;
    on[rb] = r < count;
    const int route = on[rb] ? __ldg(rows + r) : 0;
    a[rb] = intermediate + static_cast<size_t>(route) * kK2;
  }

  const uint32_t* w = weights + static_cast<size_t>(expert) * (kK2 * kN2 / 8) +
                      static_cast<size_t>(tile) * (kK2 / 8) * 32;
  constexpr size_t stats_bytes =
      static_cast<size_t>(kK2 / 32) * kN2 * (Format == 2 ? 4 : 3);
  const uint8_t* st = metadata + static_cast<size_t>(expert) * stats_bytes;

  float accum[RB][8];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
#pragma unroll
    for (int i = 0; i < 8; ++i) accum[rb][i] = 0.f;
  }

#pragma unroll 2
  for (int group = 0; group < kK2 / 16; group += 2) {
    const size_t off = static_cast<size_t>(group * 2) * 32 + packed_col;
    const uint32_t p0 = __ldcs(w + off);
    const uint32_t p1 = __ldcs(w + off + 32);
    const uint32_t p2 = __ldcs(w + off + 64);
    const uint32_t p3 = __ldcs(w + off + 96);
    const uint32_t stats =
        read_awq_stats<Format>(st, group >> 1, tile, packed_col, kN2);
    half2 d0[8], d1[8];
    dequant_awq_u4x8(p0, stats, d0);
    dequant_awq_u4x8(p1, stats, d0 + 4);
    mma_row_blocks<RB>(accum, a, on, nrb, group, d0);
    dequant_awq_u4x8(p2, stats, d1);
    dequant_awq_u4x8(p3, stats, d1 + 4);
    mma_row_blocks<RB>(accum, a, on, nrb, group + 1, d1);
  }

  // Coalesced half2 stores directly to L2-resident route_out[route, 1536].
  const int r0_local = acc_row(0, lane);
  const int r1_local = acc_row(2, lane);
  const int c0 = quadpair * 8 + acc_col(0, lane);
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    if (rb < nrb) {
      const int r0 = rb * 8 + r0_local;
      const int r1 = rb * 8 + r1_local;
      if (r0 < count) {
        const int route0 = __ldg(rows + r0);
        half* dst0 = route_out + static_cast<size_t>(route0) * kN2 +
                     tile * 32 + c0;
        *reinterpret_cast<half2*>(dst0) =
            __floats2half2_rn(accum[rb][0], accum[rb][1]);
        *reinterpret_cast<half2*>(dst0 + 4) =
            __floats2half2_rn(accum[rb][4], accum[rb][5]);
      }
      if (r1 < count) {
        const int route1 = __ldg(rows + r1);
        half* dst1 = route_out + static_cast<size_t>(route1) * kN2 +
                     tile * 32 + c0;
        *reinterpret_cast<half2*>(dst1) =
            __floats2half2_rn(accum[rb][2], accum[rb][3]);
        *reinterpret_cast<half2*>(dst1 + 4) =
            __floats2half2_rn(accum[rb][6], accum[rb][7]);
      }
    }
  }
}

// Kernel 3b: W2 SplitK=2 (2 warps per tile x 2 tiles per CTA = 4 warps).
// Grid: (24, max_packs=routes).
template <int Format, int RB>
__global__ void __launch_bounds__(128)
    hy4_grouped_w2_split2_kernel(const half* __restrict__ intermediate,
                                 const uint32_t* __restrict__ weights,
                                 const uint8_t* __restrict__ metadata,
                                 const int32_t* __restrict__ plan,
                                 half* __restrict__ route_out) {
  // 2 tiles x 2 splits x 8 rows x 32 cols x 4 bytes = 4096 bytes.
  __shared__ float partial[2][2][8][32];
  const int pack = blockIdx.y;
  if (pack >= __ldg(plan)) return;

  const int expert = __ldg(plan + kPlanExperts + pack);
  const int count = __ldg(plan + kPlanCounts + pack);
  const int32_t* rows = plan + kPlanRows + __ldg(plan + kPlanStarts + pack);
  const int nrb = (count + 7) >> 3;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int local_tile = warp >> 1;  // 0..1
  const int k_split = warp & 1;      // 0..1
  const int tile = blockIdx.x * 2 + local_tile;
  const int quadpair = (lane >> 2) & 3;
  const int a_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int packed_col = quadpair * 8 + a_row;

  const half* a[RB];
  bool on[RB];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    const int r = rb * 8 + a_row;
    on[rb] = r < count;
    const int route = on[rb] ? __ldg(rows + r) : 0;
    a[rb] = intermediate + static_cast<size_t>(route) * kK2;
  }

  const uint32_t* w = weights + static_cast<size_t>(expert) * (kK2 * kN2 / 8) +
                      static_cast<size_t>(tile) * (kK2 / 8) * 32;
  constexpr size_t stats_bytes =
      static_cast<size_t>(kK2 / 32) * kN2 * (Format == 2 ? 4 : 3);
  const uint8_t* st = metadata + static_cast<size_t>(expert) * stats_bytes;

  float accum[RB][8];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
#pragma unroll
    for (int i = 0; i < 8; ++i) accum[rb][i] = 0.f;
  }

  const int g_start = k_split * 12;
#pragma unroll 2
  for (int s = 0; s < 12; s += 2) {
    const int group = g_start + s;
    const size_t off = static_cast<size_t>(group * 2) * 32 + packed_col;
    const uint32_t p0 = __ldcs(w + off);
    const uint32_t p1 = __ldcs(w + off + 32);
    const uint32_t p2 = __ldcs(w + off + 64);
    const uint32_t p3 = __ldcs(w + off + 96);
    const uint32_t stats =
        read_awq_stats<Format>(st, group >> 1, tile, packed_col, kN2);
    half2 d0[8], d1[8];
    dequant_awq_u4x8(p0, stats, d0);
    dequant_awq_u4x8(p1, stats, d0 + 4);
    mma_row_blocks<RB>(accum, a, on, nrb, group, d0);
    dequant_awq_u4x8(p2, stats, d1);
    dequant_awq_u4x8(p3, stats, d1 + 4);
    mma_row_blocks<RB>(accum, a, on, nrb, group + 1, d1);
  }

#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    if (rb < nrb) {
      if (rb > 0) __syncthreads();
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        partial[local_tile][k_split][acc_row(i, lane)]
               [quadpair * 8 + acc_col(i, lane)] = accum[rb][i];
      }
      __syncthreads();
      const int row0 = rb * 8;
      const int rb_rows = min(count - row0, 8);
      for (int idx = threadIdx.x; idx < rb_rows * 64; idx += 128) {
        const int r = idx >> 6;
        const int lt = (idx >> 5) & 1;
        const int c = idx & 31;
        const float sum = partial[lt][0][r][c] + partial[lt][1][r][c];
        const int route = __ldg(rows + row0 + r);
        route_out[static_cast<size_t>(route) * kN2 +
                  (blockIdx.x * 2 + lt) * 32 + c] = __float2half(sum);
      }
    }
  }
}

// Kernel 4: Weighted reduction in fixed router slot order 0..15 with FP32 fmaf.
// Grid: (6, tokens), Block: 64 threads (each thread handles uint2 = 4 halfs).
__global__ void __launch_bounds__(64)
    hy4_weighted_reduce_kernel(const half* __restrict__ route_out,
                               const int32_t* __restrict__ ids,
                               const float* __restrict__ topk,
                               half* __restrict__ out) {
  __shared__ float s_w[kTopK];
  __shared__ int s_valid[kTopK];
  const int token = blockIdx.y;
  const int tid = threadIdx.x;
  if (tid < kTopK) {
    const int id = __ldg(ids + token * kTopK + tid);
    s_valid[tid] = (id >= 0 && id < kExperts) ? 1 : 0;
    s_w[tid] = __ldg(topk + token * kTopK + tid);
  }
  __syncthreads();

  const int col4 = blockIdx.x * 64 + tid;  // 0..383 (4 halfs each = 1536 cols)
  const auto* src4 = reinterpret_cast<const uint2*>(route_out);
  auto* dst4 = reinterpret_cast<uint2*>(out);

  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  const size_t base_route = static_cast<size_t>(token) * kTopK;
#pragma unroll
  for (int slot = 0; slot < kTopK; ++slot) {
    if (s_valid[slot]) {
      const float w = s_w[slot];
      const uint2 packed =
          __ldg(src4 + (base_route + slot) * (kN2 / 4) + col4);
      const half2 h01 = *reinterpret_cast<const half2*>(&packed.x);
      const half2 h23 = *reinterpret_cast<const half2*>(&packed.y);
      const float2 f01 = __half22float2(h01);
      const float2 f23 = __half22float2(h23);
      acc0 = fmaf(w, f01.x, acc0);
      acc1 = fmaf(w, f01.y, acc1);
      acc2 = fmaf(w, f23.x, acc2);
      acc3 = fmaf(w, f23.y, acc3);
    }
  }
  uint2 res;
  const half2 r01 = __floats2half2_rn(acc0, acc1);
  const half2 r23 = __floats2half2_rn(acc2, acc3);
  res.x = *reinterpret_cast<const uint32_t*>(&r01);
  res.y = *reinterpret_cast<const uint32_t*>(&r23);
  dst4[static_cast<size_t>(token) * (kN2 / 4) + col4] = res;
}

template <int Format, int Split, int RB>
void launch_w13(const at::Tensor& input, const at::Tensor& w13,
                const at::Tensor& s13, const at::Tensor& plan,
                at::Tensor& intermediate, int routes, float swiglu_limit,
                cudaStream_t stream) {
  constexpr size_t smem = static_cast<size_t>(Split) * 8 * 64 * sizeof(float);
  static_assert(smem <= 48 * 1024, "W13 partials exceed default smem");
  hy4_grouped_w13_swiglu_kernel<Format, Split, RB>
      <<<dim3(kTilesInter, routes), Split * 32, smem, stream>>>(
          reinterpret_cast<const half*>(input.const_data_ptr()),
          reinterpret_cast<const uint32_t*>(w13.const_data_ptr()),
          reinterpret_cast<const uint8_t*>(s13.const_data_ptr()),
          plan.const_data_ptr<int32_t>(),
          reinterpret_cast<half*>(intermediate.mutable_data_ptr()),
          swiglu_limit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Format, int RB>
void launch_w2(const at::Tensor& intermediate, const at::Tensor& w2,
               const at::Tensor& s2, const at::Tensor& plan,
               at::Tensor& route_out, int routes, int w2_mode,
               cudaStream_t stream) {
  if (w2_mode == 1) {
    hy4_grouped_w2_nosplit_kernel<Format, RB, 2>
        <<<dim3(kTiles2 / 2, routes), 64, 0, stream>>>(
            reinterpret_cast<const half*>(intermediate.const_data_ptr()),
            reinterpret_cast<const uint32_t*>(w2.const_data_ptr()),
            reinterpret_cast<const uint8_t*>(s2.const_data_ptr()),
            plan.const_data_ptr<int32_t>(),
            reinterpret_cast<half*>(route_out.mutable_data_ptr()));
  } else if (w2_mode == 2) {
    hy4_grouped_w2_split2_kernel<Format, RB>
        <<<dim3(kTiles2 / 2, routes), 128, 0, stream>>>(
            reinterpret_cast<const half*>(intermediate.const_data_ptr()),
            reinterpret_cast<const uint32_t*>(w2.const_data_ptr()),
            reinterpret_cast<const uint8_t*>(s2.const_data_ptr()),
            plan.const_data_ptr<int32_t>(),
            reinterpret_cast<half*>(route_out.mutable_data_ptr()));
  } else {
    hy4_grouped_w2_nosplit_kernel<Format, RB, 4>
        <<<dim3(kTiles2 / 4, routes), 128, 0, stream>>>(
            reinterpret_cast<const half*>(intermediate.const_data_ptr()),
            reinterpret_cast<const uint32_t*>(w2.const_data_ptr()),
            reinterpret_cast<const uint8_t*>(s2.const_data_ptr()),
            plan.const_data_ptr<int32_t>(),
            reinterpret_cast<half*>(route_out.mutable_data_ptr()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int W13Format, int W2Format, int RB>
void launch_all(const at::Tensor& input, const at::Tensor& w13,
                const at::Tensor& s13, const at::Tensor& w2,
                const at::Tensor& s2, const at::Tensor& ids,
                const at::Tensor& topk, const at::Tensor& plan,
                at::Tensor& intermediate, at::Tensor& route_out,
                at::Tensor& out, int tokens, int split, int w2_mode,
                float swiglu_limit, cudaStream_t stream) {
  const int routes = tokens * kTopK;
  if (split == 4) {
    launch_w13<W13Format, 4, RB>(input, w13, s13, plan, intermediate, routes,
                                 swiglu_limit, stream);
  } else if (split == 6) {
    launch_w13<W13Format, 6, RB>(input, w13, s13, plan, intermediate, routes,
                                 swiglu_limit, stream);
  } else if (split == 12) {
    launch_w13<W13Format, 12, RB>(input, w13, s13, plan, intermediate, routes,
                                  swiglu_limit, stream);
  } else {
    launch_w13<W13Format, 8, RB>(input, w13, s13, plan, intermediate, routes,
                                 swiglu_limit, stream);
  }
  launch_w2<W2Format, RB>(intermediate, w2, s2, plan, route_out, routes,
                          w2_mode, stream);
  hy4_weighted_reduce_kernel<<<dim3(kN2 / 256, tokens), 64, 0, stream>>>(
      reinterpret_cast<const half*>(route_out.const_data_ptr()),
      ids.const_data_ptr<int32_t>(), topk.const_data_ptr<float>(),
      reinterpret_cast<half*>(out.mutable_data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int W13Format, int W2Format>
void dispatch_rb(int rb, const at::Tensor& input, const at::Tensor& w13,
                 const at::Tensor& s13, const at::Tensor& w2,
                 const at::Tensor& s2, const at::Tensor& ids,
                 const at::Tensor& topk, const at::Tensor& plan,
                 at::Tensor& intermediate, at::Tensor& route_out,
                 at::Tensor& out, int tokens, int split, int w2_mode,
                 float swiglu_limit, cudaStream_t stream) {
  switch (rb) {
    case 1:
      launch_all<W13Format, W2Format, 1>(input, w13, s13, w2, s2, ids, topk,
                                         plan, intermediate, route_out, out,
                                         tokens, split, w2_mode, swiglu_limit,
                                         stream);
      break;
    case 2:
      launch_all<W13Format, W2Format, 2>(input, w13, s13, w2, s2, ids, topk,
                                         plan, intermediate, route_out, out,
                                         tokens, split, w2_mode, swiglu_limit,
                                         stream);
      break;
    case 3:
      launch_all<W13Format, W2Format, 3>(input, w13, s13, w2, s2, ids, topk,
                                         plan, intermediate, route_out, out,
                                         tokens, split, w2_mode, swiglu_limit,
                                         stream);
      break;
    default:
      launch_all<W13Format, W2Format, 4>(input, w13, s13, w2, s2, ids, topk,
                                         plan, intermediate, route_out, out,
                                         tokens, split, w2_mode, swiglu_limit,
                                         stream);
      break;
  }
}

// ---------------------------------------------------------------------------
// Small-batch plan and reduce folding (VLLM_HY4_EXPERT_UNION_FOLD=1), 1..4
// tokens. The default W13 and W2 split-2 kernel bodies run unchanged (same
// grids, registers and load pipelining, so the aux-stream shared expert keeps
// its co-residency); only two launches go:
//   * No plan launch. Each CTA builds the expert union from <= 64 routing ids
//     in shared memory (packs ordered by first route, rows by token).
//   * No weighted-reduce launch. The last W2 CTA to finish a 64-column tile
//     (atomic ticket, zeroed by the preceding W13 launch) sums that tile's 16
//     router slots per token from L2 in the default slot order.
// Output bits equal awq_union_moe_out(split=0|4, w2_mode=2, pack_rows=0).
constexpr int kFusedMaxTokens = 4;
constexpr int kFusedMaxRoutes = kFusedMaxTokens * kTopK;  // 64
constexpr int kFusedTiles2 = kTiles2 / 2;                 // 24 W2 CTA tiles
struct FusedPlan {
  static constexpr int kRoutes = kFusedMaxRoutes;
  int ids[kFusedMaxRoutes];
  int experts[kFusedMaxRoutes];
  int first_count[2];
  int npacks;
  int count;
  int rows[kFusedMaxTokens];
  int ticket;
  int last;
};

// Every thread of a 128-thread CTA calls this. Builds s.experts[0..npacks).
// Requires distinct expert ids within each token, which grouped top-k
// guarantees; the default planner also accepts repeats.
__device__ __forceinline__ void fused_plan(const int32_t* __restrict__ ids,
                                           int tokens, FusedPlan& s) {
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int warp = t >> 5;
  const int routes = tokens * kTopK;
  int id = -1;
  if (t < kFusedMaxRoutes) {
    id = t < routes ? __ldg(ids + t) : -1;
    if (id < 0 || id >= kExperts) id = -1;
    s.ids[t] = id;
  }
  __syncthreads();
  // First occurrence: no earlier token routes to the same expert (ids are
  // distinct within a token).
  bool first = id >= 0;
  for (int r = 0; first && r < (t / kTopK) * kTopK; ++r) first = s.ids[r] != id;
  const unsigned ballot = __ballot_sync(0xffffffffu, first);
  if (warp < 2 && lane == 0) s.first_count[warp] = __popc(ballot);
  __syncthreads();
  if (first) {
    s.experts[(warp == 1 ? s.first_count[0] : 0) +
              __popc(ballot & ((1u << lane) - 1u))] = id;
  }
  if (t == 0) s.npacks = s.first_count[0] + s.first_count[1];
  __syncthreads();
}

// EXPERTS4B (HY4_EXPERTS4B_VERIFY): the inline plan for 5..8 tokens, the DFly
// verify rows. Its 128 routes fill the 128-thread CTA. A per-expert token mask
// in shared memory finds first occurrences with one atomic per route instead of
// scanning earlier tokens. Every pack still holds at most one row per token, so
// one 8-row block covers it and the W13/W2 bodies run unchanged.
constexpr int kWideMaxTokens = 8;
struct WidePlan {
  static constexpr int kRoutes = kWideMaxTokens * kTopK;  // 128
  int ids[kRoutes];
  int experts[kRoutes];
  unsigned seen[kExperts / 4];  // 8-bit token mask per expert
  int first_count[kRoutes / 32];
  int npacks;
  int count;
  int rows[kWideMaxTokens];
  int ticket;
  int last;
};
static_assert(WidePlan::kRoutes == 128 && kExperts / 4 == 128,
              "WidePlan: one route and one mask word per thread");

// Every thread of a 128-thread CTA calls this. Same packs as the narrow plan:
// first occurrences in route order, so packs are ordered by first route.
__device__ __forceinline__ void fused_plan(const int32_t* __restrict__ ids,
                                           int tokens, WidePlan& s) {
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int warp = t >> 5;
  const int token = t / kTopK;
  int id = t < tokens * kTopK ? __ldg(ids + t) : -1;
  if (id < 0 || id >= kExperts) id = -1;
  s.ids[t] = id;
  s.seen[t] = 0u;
  __syncthreads();
  const int shift = (id & 3) * 8;
  if (id >= 0) atomicOr(s.seen + (id >> 2), 1u << (shift + token));
  __syncthreads();
  // First occurrence: no earlier token routes to the same expert (ids are
  // distinct within a token).
  const bool first =
      id >= 0 && ((s.seen[id >> 2] >> shift) & ((1u << token) - 1u)) == 0u;
  const unsigned ballot = __ballot_sync(0xffffffffu, first);
  if (lane == 0) s.first_count[warp] = __popc(ballot);
  __syncthreads();
  if (first) {
    int base = 0;
    for (int w = 0; w < warp; ++w) base += s.first_count[w];
    s.experts[base + __popc(ballot & ((1u << lane) - 1u))] = id;
  }
  if (t == 0) {
    s.npacks = s.first_count[0] + s.first_count[1] + s.first_count[2] +
               s.first_count[3];
  }
  __syncthreads();
}

// Warp 0 gathers the rows (one per token at most) of pack's expert.
template <class Plan>
__device__ __forceinline__ void fused_rows(int tokens, int pack, Plan& s) {
  const int lane = threadIdx.x & 31;
  if (threadIdx.x < 32) {
    const int expert = s.experts[pack];
    int route = -1;
    if (lane < tokens) {
#pragma unroll
      for (int slot = 0; slot < kTopK; ++slot) {
        if (s.ids[lane * kTopK + slot] == expert) route = lane * kTopK + slot;
      }
    }
    const unsigned found = __ballot_sync(0xffffffffu, route >= 0);
    if (route >= 0) s.rows[__popc(found & ((1u << lane) - 1u))] = route;
    if (lane == 0) s.count = __popc(found);
  }
  __syncthreads();
}

// Writes zeros for one 64-column output tile (no valid expert at all).
__device__ __forceinline__ void fused_zero_tile(int tokens, int x2,
                                                half* __restrict__ out) {
  for (int idx = threadIdx.x; idx < tokens * 32; idx += blockDim.x) {
    reinterpret_cast<half2*>(out)[static_cast<size_t>(idx >> 5) * (kN2 / 2) +
                                  x2 * 32 + (idx & 31)] =
        __floats2half2_rn(0.f, 0.f);
  }
}



// Lever 2 only: the default W13/W2 kernel bodies with the plan built in each
// CTA (no plan launch) and the weighted reduction done by W2's last CTA per
// tile (no reduce launch). Same grids, registers and load pipelining as the
// default kernels. W13 CTA (0,0) zeroes W2's tile tickets.
template <class Plan>
__device__ __forceinline__ bool fold_pack(const int32_t* __restrict__ ids,
                                          int tokens, int pack,
                                          int32_t* __restrict__ counters,
                                          Plan& s) {
  if (counters != nullptr && blockIdx.x == 0 && blockIdx.y == 0 &&
      threadIdx.x < kFusedTiles2) {
    counters[threadIdx.x] = 0;
  }
  fused_plan(ids, tokens, s);
  if (pack >= s.npacks) return false;
  fused_rows(tokens, pack, s);
  return true;
}

template <int Format, int Split, int RB>
__global__ void __launch_bounds__(Split * 32)
    hy4_fold_w13_swiglu_kernel(const half* __restrict__ x,
                                  const uint32_t* __restrict__ weights,
                                  const uint8_t* __restrict__ metadata,
                                  const int32_t* __restrict__ ids, int tokens,
                                  int32_t* __restrict__ counters,
                                  half* __restrict__ intermediate,
                                  float swiglu_limit) {
  // Partial buffer for 1 row block (8 rows) x 64 cols (32 gate + 32 up):
  // Split * 8 * 64 * 4 bytes (8 KB at Split=4, 12 KB at 6, 16 KB at 8).
  extern __shared__ float partial[];
  __shared__ FusedPlan s;
  constexpr int kSteps = (kK13 / 16) / Split;  // 96 / Split
  static_assert((kK13 / 16) % Split == 0, "Split must divide 96");
  static_assert(kSteps % 2 == 0, "warps must start on an even K16 step");
  static_assert(RB >= 1 && RB <= kMaxRowBlocks, "unsupported row blocks");

  const int pack = blockIdx.y;
  if (!fold_pack(ids, tokens, pack, counters, s)) return;

  const int expert = s.experts[pack];
  const int count = s.count;
  const int* rows = s.rows;
  const int nrb = (count + 7) >> 3;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;  // 0..11
  const int quadpair = (lane >> 2) & 3;
  const int a_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int packed_col = quadpair * 8 + a_row;

  const half* a[RB];
  bool on[RB];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    const int r = rb * 8 + a_row;
    on[rb] = r < count;
    const int route = on[rb] ? rows[r] : 0;
    a[rb] = x + static_cast<size_t>(route / kTopK) * kK13;
  }

  const uint32_t* w = weights + static_cast<size_t>(expert) * (kK13 * kN13 / 8);
  constexpr size_t stats_bytes =
      static_cast<size_t>(kK13 / 32) * kN13 * (Format == 2 ? 4 : 3);
  const uint8_t* st = metadata + static_cast<size_t>(expert) * stats_bytes;

  float acc_g[RB][8];
  float acc_u[RB][8];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc_g[rb][i] = 0.f;
      acc_u[rb][i] = 0.f;
    }
  }

  constexpr size_t tile_stride_w = static_cast<size_t>(kK13 / 8) * 32;
  const uint32_t* wg = w + static_cast<size_t>(tile) * tile_stride_w;
  const uint32_t* wu =
      w + static_cast<size_t>(tile + kTilesInter) * tile_stride_w;

#pragma unroll 2
  for (int s = 0; s < kSteps; s += 2) {
    const int group = warp * kSteps + s;  // even K16 step
    const size_t off = static_cast<size_t>(group * 2) * 32 + packed_col;

    const uint32_t pg0 = __ldcs(wg + off);
    const uint32_t pg1 = __ldcs(wg + off + 32);
    const uint32_t pg2 = __ldcs(wg + off + 64);
    const uint32_t pg3 = __ldcs(wg + off + 96);
    const uint32_t st_g =
        read_awq_stats<Format>(st, group >> 1, tile, packed_col, kN13);

    const uint32_t pu0 = __ldcs(wu + off);
    const uint32_t pu1 = __ldcs(wu + off + 32);
    const uint32_t pu2 = __ldcs(wu + off + 64);
    const uint32_t pu3 = __ldcs(wu + off + 96);
    const uint32_t st_u = read_awq_stats<Format>(
        st, group >> 1, tile + kTilesInter, packed_col, kN13);

    half2 dg0[8], dg1[8], du0[8], du1[8];
    dequant_awq_u4x8(pg0, st_g, dg0);
    dequant_awq_u4x8(pg1, st_g, dg0 + 4);
    dequant_awq_u4x8(pu0, st_u, du0);
    dequant_awq_u4x8(pu1, st_u, du0 + 4);
    mma_row_blocks_dual<RB>(acc_g, acc_u, a, on, nrb, group, dg0, du0);

    dequant_awq_u4x8(pg2, st_g, dg1);
    dequant_awq_u4x8(pg3, st_g, dg1 + 4);
    dequant_awq_u4x8(pu2, st_u, du1);
    dequant_awq_u4x8(pu3, st_u, du1 + 4);
    mma_row_blocks_dual<RB>(acc_g, acc_u, a, on, nrb, group + 1, dg1, du1);
  }

  const bool has_clamp = swiglu_limit > 0.f;
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    if (rb < nrb) {
      if (rb > 0) __syncthreads();
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int r = acc_row(i, lane);
        const int c = quadpair * 8 + acc_col(i, lane);
        const int base = (warp * 8 + r) * 64 + c;
        partial[base] = acc_g[rb][i];
        partial[base + 32] = acc_u[rb][i];
      }
      __syncthreads();
      const int row0 = rb * 8;
      const int rb_rows = min(count - row0, 8);
      for (int idx = threadIdx.x; idx < rb_rows * 32; idx += blockDim.x) {
        const int r = idx >> 5;
        const int c = idx & 31;
        float gate_sum = 0.f;
        float up_sum = 0.f;
#pragma unroll
        for (int s = 0; s < Split; ++s) {
          const float* p = partial + (s * 8 + r) * 64;
          gate_sum += p[c];
          up_sum += p[32 + c];
        }
        half gate_h = __float2half(gate_sum);
        half up_h = __float2half(up_sum);
        float g = __half2float(gate_h);
        float u = __half2float(up_h);
        if (has_clamp) {
          g = fminf(g, swiglu_limit);
          u = fmaxf(fminf(u, swiglu_limit), -swiglu_limit);
          gate_h = __float2half(g);
          up_h = __float2half(u);
          g = __half2float(gate_h);
        }
        const half silu_h = __float2half(g / (1.0f + expf(-g)));
        const int route = rows[row0 + r];
        intermediate[static_cast<size_t>(route) * kInter + tile * 32 + c] =
            __hmul(silu_h, up_h);
      }
    }
  }
}


template <int Format, int RB>
__global__ void __launch_bounds__(128)
    hy4_fold_w2_split2_kernel(const half* __restrict__ intermediate,
                                 const uint32_t* __restrict__ weights,
                                 const uint8_t* __restrict__ metadata,
                                 const int32_t* __restrict__ ids,
                                 const float* __restrict__ topk, int tokens,
                                 int32_t* __restrict__ counters,
                                 half* __restrict__ route_out,
                                 half* __restrict__ out) {
  // 2 tiles x 2 splits x 8 rows x 32 cols x 4 bytes = 4096 bytes.
  __shared__ float partial[2][2][8][32];
  __shared__ FusedPlan s;
  __shared__ float s_w[kFusedMaxRoutes];
  const int pack = blockIdx.y;
  if (threadIdx.x < kFusedMaxRoutes) {
    s_w[threadIdx.x] =
        threadIdx.x < tokens * kTopK ? __ldg(topk + threadIdx.x) : 0.f;
  }
  if (!fold_pack(ids, tokens, pack, nullptr, s)) {
    if (s.npacks == 0 && pack == 0) fused_zero_tile(tokens, blockIdx.x, out);
    return;
  }

  const int expert = s.experts[pack];
  const int count = s.count;
  const int* rows = s.rows;
  const int nrb = (count + 7) >> 3;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int local_tile = warp >> 1;  // 0..1
  const int k_split = warp & 1;      // 0..1
  const int tile = blockIdx.x * 2 + local_tile;
  const int quadpair = (lane >> 2) & 3;
  const int a_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int packed_col = quadpair * 8 + a_row;

  const half* a[RB];
  bool on[RB];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    const int r = rb * 8 + a_row;
    on[rb] = r < count;
    const int route = on[rb] ? rows[r] : 0;
    a[rb] = intermediate + static_cast<size_t>(route) * kK2;
  }

  const uint32_t* w = weights + static_cast<size_t>(expert) * (kK2 * kN2 / 8) +
                      static_cast<size_t>(tile) * (kK2 / 8) * 32;
  constexpr size_t stats_bytes =
      static_cast<size_t>(kK2 / 32) * kN2 * (Format == 2 ? 4 : 3);
  const uint8_t* st = metadata + static_cast<size_t>(expert) * stats_bytes;

  float accum[RB][8];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
#pragma unroll
    for (int i = 0; i < 8; ++i) accum[rb][i] = 0.f;
  }

  const int g_start = k_split * 12;
#pragma unroll 2
  for (int s = 0; s < 12; s += 2) {
    const int group = g_start + s;
    const size_t off = static_cast<size_t>(group * 2) * 32 + packed_col;
    const uint32_t p0 = __ldcs(w + off);
    const uint32_t p1 = __ldcs(w + off + 32);
    const uint32_t p2 = __ldcs(w + off + 64);
    const uint32_t p3 = __ldcs(w + off + 96);
    const uint32_t stats =
        read_awq_stats<Format>(st, group >> 1, tile, packed_col, kN2);
    half2 d0[8], d1[8];
    dequant_awq_u4x8(p0, stats, d0);
    dequant_awq_u4x8(p1, stats, d0 + 4);
    mma_row_blocks<RB>(accum, a, on, nrb, group, d0);
    dequant_awq_u4x8(p2, stats, d1);
    dequant_awq_u4x8(p3, stats, d1 + 4);
    mma_row_blocks<RB>(accum, a, on, nrb, group + 1, d1);
  }

#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    if (rb < nrb) {
      if (rb > 0) __syncthreads();
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        partial[local_tile][k_split][acc_row(i, lane)]
               [quadpair * 8 + acc_col(i, lane)] = accum[rb][i];
      }
      __syncthreads();
      const int row0 = rb * 8;
      const int rb_rows = min(count - row0, 8);
      for (int idx = threadIdx.x; idx < rb_rows * 64; idx += 128) {
        const int r = idx >> 6;
        const int lt = (idx >> 5) & 1;
        const int c = idx & 31;
        const float sum = partial[lt][0][r][c] + partial[lt][1][r][c];
        const int route = rows[row0 + r];
        route_out[static_cast<size_t>(route) * kN2 +
                  (blockIdx.x * 2 + lt) * 32 + c] = __float2half(sum);
      }
    }
  }

  // Last CTA for this 64-column tile performs the weighted reduction in the
  // default router-slot order.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    s.last = atomicAdd(counters + blockIdx.x, 1) == s.npacks - 1;
  }
  __syncthreads();
  if (!s.last) return;
  __threadfence();
  for (int idx = threadIdx.x; idx < tokens * 32; idx += blockDim.x) {
    const int tok = idx >> 5;
    const int col2 = blockIdx.x * 32 + (idx & 31);
    float acc0 = 0.f, acc1 = 0.f;
#pragma unroll
    for (int slot = 0; slot < kTopK; ++slot) {
      const int route = tok * kTopK + slot;
      if (s.ids[route] >= 0) {
        const float wt = s_w[route];
        const half2 h = __ldcg(reinterpret_cast<const half2*>(route_out) +
                               static_cast<size_t>(route) * (kN2 / 2) + col2);
        const float2 f = __half22float2(h);
        acc0 = fmaf(wt, f.x, acc0);
        acc1 = fmaf(wt, f.y, acc1);
      }
    }
    reinterpret_cast<half2*>(out)[static_cast<size_t>(tok) * (kN2 / 2) + col2] =
        __floats2half2_rn(acc0, acc1);
  }
}

// EXPERTS3 ring kernels (HY4_EXPERTS3_RING): the fold W13/W2 with a rolled
// software pipeline. Each warp keeps Depth K32 pairs of weight and stats loads
// in flight in a register ring while it runs the mma for the oldest pair; the
// loop is unrolled only by Depth, so the body holds Depth pairs. Stats stay
// raw loaded words until use, so no load waits on a stats shuffle. Each CTA
// builds the plan inline (fold_pack), as the fold kernels do.
// Per-warp K order, mma sequence, partial-sum order and epilogue equal the
// fold kernels', so the output bits do too.
// Megakernel hosting: each body re-asserts warp convergence (__syncwarp)
// after the plan and before its K loop (W13WHY rule 2).
template <int Format>
struct RawStats {
  uint32_t v[Format == 0 ? 3 : 1];
};

template <int Format>
__device__ __forceinline__ RawStats<Format> load_awq_stats(const uint8_t* stats,
                                                           int group, int tile,
                                                           int col, int n) {
  RawStats<Format> r;
  if constexpr (Format == 2) {
    r.v[0] = __ldg(reinterpret_cast<const uint32_t*>(stats) +
                   static_cast<size_t>(group) * n + tile * 32 + col);
  } else if constexpr (Format == 1) {
    const int lane = threadIdx.x & 31;
    const auto* words = reinterpret_cast<const uint32_t*>(
        stats + (static_cast<size_t>(group) * n + tile * 32) * 3);
    r.v[0] = lane < 24 ? __ldg(words + lane) : 0;
  } else {
    const auto* record =
        stats + (static_cast<size_t>(group) * n + tile * 32 + col) * 3;
    r.v[0] = __ldg(record);
    r.v[1] = __ldg(record + 1);
    r.v[2] = __ldg(record + 2);
  }
  return r;
}

// Bit-identical to read_awq_stats<Format> on the same bytes.
template <int Format>
__device__ __forceinline__ uint32_t decode_awq_stats(const RawStats<Format>& r,
                                                     int col) {
  if constexpr (Format == 2) {
    return r.v[0];
  } else {
    uint32_t bits = 0;
    if constexpr (Format == 1) {
      const int byte = col * 3;
      const int shift = (byte & 3) * 8;
      const uint32_t lo = __shfl_sync(0xffffffffu, r.v[0], byte >> 2);
      const uint32_t hi =
          __shfl_sync(0xffffffffu, r.v[0], (byte >> 2) + (shift > 8));
      bits = lo >> shift;
      if (shift > 8) bits |= hi << (32 - shift);
    } else {
      bits = r.v[0] | (r.v[1] << 8) | (r.v[2] << 16);
    }
    const half scale = __ushort_as_half(static_cast<uint16_t>(bits));
    const half zero = __int2half_rn(static_cast<int>((bits >> 16) & 0xff));
    const half bias = __hmul(__hneg(zero), scale);
    return (bits & 0xffffu) |
           (static_cast<uint32_t>(__half_as_ushort(bias)) << 16);
  }
}

template <int Format>
struct W13Pair {
  uint32_t g[4], u[4];
  RawStats<Format> sg, su;
};

struct W13Src {
  const uint32_t* wg;
  const uint32_t* wu;
  const uint8_t* st;
};

template <int Format>
__device__ __forceinline__ W13Src w13_src(const uint32_t* weights,
                                          const uint8_t* metadata, int expert,
                                          int tile) {
  const uint32_t* w = weights + static_cast<size_t>(expert) * (kK13 * kN13 / 8);
  constexpr size_t stats_bytes =
      static_cast<size_t>(kK13 / 32) * kN13 * (Format == 2 ? 4 : 3);
  constexpr size_t tile_stride_w = static_cast<size_t>(kK13 / 8) * 32;
  return {w + static_cast<size_t>(tile) * tile_stride_w,
          w + static_cast<size_t>(tile + kTilesInter) * tile_stride_w,
          metadata + static_cast<size_t>(expert) * stats_bytes};
}

// group: even K16 step of the pair.
template <int Format>
__device__ __forceinline__ void w13_issue(W13Pair<Format>& p, const W13Src& src,
                                          int group, int tile, int col) {
  const size_t off = static_cast<size_t>(group * 2) * 32 + col;
  p.g[0] = __ldcs(src.wg + off);
  p.g[1] = __ldcs(src.wg + off + 32);
  p.g[2] = __ldcs(src.wg + off + 64);
  p.g[3] = __ldcs(src.wg + off + 96);
  p.sg = load_awq_stats<Format>(src.st, group >> 1, tile, col, kN13);
  p.u[0] = __ldcs(src.wu + off);
  p.u[1] = __ldcs(src.wu + off + 32);
  p.u[2] = __ldcs(src.wu + off + 64);
  p.u[3] = __ldcs(src.wu + off + 96);
  p.su = load_awq_stats<Format>(src.st, group >> 1, tile + kTilesInter, col,
                                kN13);
}

// W13 + SwiGLU for one (pack, 32 gate + 32 up column tile): 4 warps split K.
// partial: 4 warps * 8 rows * 64 cols of dynamic shared memory.
template <int Format, int Depth, class Plan>
__device__ __forceinline__ void ring_w13_cta(
    const half* __restrict__ x, const uint32_t* __restrict__ weights,
    const uint8_t* __restrict__ metadata, const int32_t* __restrict__ ids,
    int tokens, int32_t* __restrict__ counters,
    half* __restrict__ intermediate, float swiglu_limit, float* partial,
    Plan& s) {
  constexpr int kSplit = 4;
  constexpr int kSteps = (kK13 / 16) / kSplit;
  constexpr int kPairs = kSteps / 2;
  static_assert(kPairs % Depth == 0, "Depth must divide the warp's K pairs");
  const int pack = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int quadpair = (lane >> 2) & 3;
  const int a_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int packed_col = quadpair * 8 + a_row;
  const int step0 = warp * kSteps;

  if (!fold_pack(ids, tokens, pack, counters, s)) return;
  __syncwarp();
  const int expert = s.experts[pack];
  const int count = s.count;
  const W13Src src = w13_src<Format>(weights, metadata, expert, tile);

  float acc_g[1][8];
  float acc_u[1][8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc_g[0][i] = 0.f;
    acc_u[0][i] = 0.f;
  }
  W13Pair<Format> ring[Depth];
#pragma unroll
  for (int i = 0; i < Depth; ++i)
    w13_issue(ring[i], src, step0 + 2 * i, tile, packed_col);
  bool on[1];
  const half* a[1];
  on[0] = a_row < count;
  a[0] = x + static_cast<size_t>((on[0] ? s.rows[a_row] : 0) / kTopK) * kK13;
  __syncwarp();
#pragma unroll 1
  for (int i0 = 0; i0 < kPairs; i0 += Depth) {
#pragma unroll
    for (int j = 0; j < Depth; ++j) {
      const int i = i0 + j;
      const W13Pair<Format> cur = ring[j];
      if (i + Depth < kPairs)
        w13_issue(ring[j], src, step0 + 2 * (i + Depth), tile, packed_col);
      const int group = step0 + 2 * i;
      const uint32_t st_g = decode_awq_stats<Format>(cur.sg, packed_col);
      const uint32_t st_u = decode_awq_stats<Format>(cur.su, packed_col);
      half2 dg0[8], dg1[8], du0[8], du1[8];
      dequant_awq_u4x8(cur.g[0], st_g, dg0);
      dequant_awq_u4x8(cur.g[1], st_g, dg0 + 4);
      dequant_awq_u4x8(cur.u[0], st_u, du0);
      dequant_awq_u4x8(cur.u[1], st_u, du0 + 4);
      mma_row_blocks_dual<1>(acc_g, acc_u, a, on, 1, group, dg0, du0);
      dequant_awq_u4x8(cur.g[2], st_g, dg1);
      dequant_awq_u4x8(cur.g[3], st_g, dg1 + 4);
      dequant_awq_u4x8(cur.u[2], st_u, du1);
      dequant_awq_u4x8(cur.u[3], st_u, du1 + 4);
      mma_row_blocks_dual<1>(acc_g, acc_u, a, on, 1, group + 1, dg1, du1);
    }
  }

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int r = acc_row(i, lane);
    const int c = quadpair * 8 + acc_col(i, lane);
    const int base = (warp * 8 + r) * 64 + c;
    partial[base] = acc_g[0][i];
    partial[base + 32] = acc_u[0][i];
  }
  __syncthreads();
  const bool has_clamp = swiglu_limit > 0.f;
  const int rows = min(count, 8);
  for (int idx = threadIdx.x; idx < rows * 32; idx += blockDim.x) {
    const int r = idx >> 5;
    const int c = idx & 31;
    float gate_sum = 0.f;
    float up_sum = 0.f;
#pragma unroll
    for (int sp = 0; sp < kSplit; ++sp) {
      const float* p = partial + (sp * 8 + r) * 64;
      gate_sum += p[c];
      up_sum += p[32 + c];
    }
    half gate_h = __float2half(gate_sum);
    half up_h = __float2half(up_sum);
    float g = __half2float(gate_h);
    float u = __half2float(up_h);
    if (has_clamp) {
      g = fminf(g, swiglu_limit);
      u = fmaxf(fminf(u, swiglu_limit), -swiglu_limit);
      gate_h = __float2half(g);
      up_h = __float2half(u);
      g = __half2float(gate_h);
    }
    const half silu_h = __float2half(g / (1.0f + expf(-g)));
    intermediate[static_cast<size_t>(s.rows[r]) * kInter + tile * 32 + c] =
        __hmul(silu_h, up_h);
  }
}

template <int Format, int Depth>
__global__ void __launch_bounds__(128, 1)
    hy4_ring_w13_kernel(const half* __restrict__ x,
                        const uint32_t* __restrict__ weights,
                        const uint8_t* __restrict__ metadata,
                        const int32_t* __restrict__ ids, int tokens,
                        int32_t* __restrict__ counters,
                        half* __restrict__ intermediate, float swiglu_limit) {
  extern __shared__ float partial[];  // 4 warps * 8 rows * 64 cols
  __shared__ FusedPlan s;
  ring_w13_cta<Format, Depth>(x, weights, metadata, ids, tokens, counters,
                              intermediate, swiglu_limit, partial, s);
}

template <int Format>
struct W2Pair {
  uint32_t w[4];
  RawStats<Format> st;
};

template <int Format>
__device__ __forceinline__ void w2_issue(W2Pair<Format>& p, const uint32_t* w,
                                         const uint8_t* st, int group, int tile,
                                         int col) {
  const size_t off = static_cast<size_t>(group * 2) * 32 + col;
  p.w[0] = __ldcs(w + off);
  p.w[1] = __ldcs(w + off + 32);
  p.w[2] = __ldcs(w + off + 64);
  p.w[3] = __ldcs(w + off + 96);
  p.st = load_awq_stats<Format>(st, group >> 1, tile, col, kN2);
}

// W2 with 64-column CTA tiles, two K halves per tile (fold W2 layout).
template <int Format, int Depth, class Plan>
__device__ __forceinline__ void ring_w2_cta(
    const half* __restrict__ intermediate, const uint32_t* __restrict__ weights,
    const uint8_t* __restrict__ metadata, const int32_t* __restrict__ ids,
    const float* __restrict__ topk, int tokens, int32_t* __restrict__ counters,
    half* __restrict__ route_out, half* __restrict__ out,
    float (&partial)[2][2][8][32], Plan& s, float* s_w) {
  constexpr int kPairs = 6;
  static_assert(kPairs % Depth == 0, "Depth must divide 6");
  constexpr size_t stats_bytes =
      static_cast<size_t>(kK2 / 32) * kN2 * (Format == 2 ? 4 : 3);
  const int pack = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int local_tile = warp >> 1;
  const int k_split = warp & 1;
  const int tile = blockIdx.x * 2 + local_tile;
  const int quadpair = (lane >> 2) & 3;
  const int a_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int packed_col = quadpair * 8 + a_row;
  const int g_start = k_split * 12;

  if (threadIdx.x < Plan::kRoutes) {
    s_w[threadIdx.x] =
        threadIdx.x < tokens * kTopK ? __ldg(topk + threadIdx.x) : 0.f;
  }
  if (!fold_pack(ids, tokens, pack, nullptr, s)) {
    if (s.npacks == 0 && pack == 0) fused_zero_tile(tokens, blockIdx.x, out);
    return;
  }
  __syncwarp();
  const int expert = s.experts[pack];
  const int count = s.count;
  const uint32_t* w = weights + static_cast<size_t>(expert) * (kK2 * kN2 / 8) +
                      static_cast<size_t>(tile) * (kK2 / 8) * 32;
  const uint8_t* st = metadata + static_cast<size_t>(expert) * stats_bytes;
  W2Pair<Format> ring[Depth];
#pragma unroll
  for (int i = 0; i < Depth; ++i)
    w2_issue(ring[i], w, st, g_start + 2 * i, tile, packed_col);
  bool on[1];
  const half* a[1];
  on[0] = a_row < count;
  a[0] = intermediate + static_cast<size_t>(on[0] ? s.rows[a_row] : 0) * kK2;
  float accum[1][8];
#pragma unroll
  for (int i = 0; i < 8; ++i) accum[0][i] = 0.f;
  __syncwarp();
#pragma unroll 1
  for (int i0 = 0; i0 < kPairs; i0 += Depth) {
#pragma unroll
    for (int j = 0; j < Depth; ++j) {
      const int i = i0 + j;
      const W2Pair<Format> cur = ring[j];
      if (i + Depth < kPairs)
        w2_issue(ring[j], w, st, g_start + 2 * (i + Depth), tile, packed_col);
      const int group = g_start + 2 * i;
      const uint32_t stats = decode_awq_stats<Format>(cur.st, packed_col);
      half2 d0[8], d1[8];
      dequant_awq_u4x8(cur.w[0], stats, d0);
      dequant_awq_u4x8(cur.w[1], stats, d0 + 4);
      mma_row_blocks<1>(accum, a, on, 1, group, d0);
      dequant_awq_u4x8(cur.w[2], stats, d1);
      dequant_awq_u4x8(cur.w[3], stats, d1 + 4);
      mma_row_blocks<1>(accum, a, on, 1, group + 1, d1);
    }
  }

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    partial[local_tile][k_split][acc_row(i, lane)]
           [quadpair * 8 + acc_col(i, lane)] = accum[0][i];
  }
  __syncthreads();
  const int rows = min(count, 8);
  for (int idx = threadIdx.x; idx < rows * 64; idx += 128) {
    const int r = idx >> 6;
    const int lt = (idx >> 5) & 1;
    const int c = idx & 31;
    const float sum = partial[lt][0][r][c] + partial[lt][1][r][c];
    route_out[static_cast<size_t>(s.rows[r]) * kN2 + (blockIdx.x * 2 + lt) * 32 +
              c] = __float2half(sum);
  }

  // Last CTA for this 64-column tile: weighted reduction in router-slot order.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    s.last = atomicAdd(counters + blockIdx.x, 1) == s.npacks - 1;
  }
  __syncthreads();
  if (!s.last) return;
  __threadfence();
  for (int idx = threadIdx.x; idx < tokens * 32; idx += blockDim.x) {
    const int tok = idx >> 5;
    const int col2 = blockIdx.x * 32 + (idx & 31);
    float acc0 = 0.f, acc1 = 0.f;
#pragma unroll
    for (int slot = 0; slot < kTopK; ++slot) {
      const int route = tok * kTopK + slot;
      if (s.ids[route] >= 0) {
        const float wt = s_w[route];
        const half2 h = __ldcg(reinterpret_cast<const half2*>(route_out) +
                               static_cast<size_t>(route) * (kN2 / 2) + col2);
        const float2 f = __half22float2(h);
        acc0 = fmaf(wt, f.x, acc0);
        acc1 = fmaf(wt, f.y, acc1);
      }
    }
    reinterpret_cast<half2*>(out)[static_cast<size_t>(tok) * (kN2 / 2) + col2] =
        __floats2half2_rn(acc0, acc1);
  }
}

template <int Format, int Depth>
__global__ void __launch_bounds__(128)
    hy4_ring_w2_kernel(const half* __restrict__ intermediate,
                       const uint32_t* __restrict__ weights,
                       const uint8_t* __restrict__ metadata,
                       const int32_t* __restrict__ ids,
                       const float* __restrict__ topk, int tokens,
                       int32_t* __restrict__ counters,
                       half* __restrict__ route_out, half* __restrict__ out) {
  __shared__ float partial[2][2][8][32];
  __shared__ FusedPlan s;
  __shared__ float s_w[kFusedMaxRoutes];
  ring_w2_cta<Format, Depth>(intermediate, weights, metadata, ids, topk, tokens,
                             counters, route_out, out, partial, s, s_w);
}

// EXPERTS4 (HY4_EXPERTS4_SHARED): the shared expert rides inside the ring
// W13 and W2 launches as extra CTA rows (blockIdx.y >= routes), so it needs no
// aux stream, fork or join, and its CTAs fill the routed grid's empty slots.
// The role is per CTA, so neither codec sits behind a lane-divergent branch.
// Each shared CTA runs w8_project's body for one CTA of the grid that
// w8_projection_grouped launches for that shape (same warp_project call, K
// split, shared-memory sum order and rounding), so the bits match the aux-stream
// chain gate_up (split-K 8) -> finish -> silu_and_mul -> down.
//   W13 launch: 192 CTAs = gate_up grid (24 tiles, 8 splits). The last of the
//     16 CTAs feeding one 32-column gate/up pair (unsigned ticket counted
//     modulo 16, so it never needs a reset) runs finish and SiLU-and-mul.
//   W2 launch: 96 CTAs = down grid (96 tiles, no split).
constexpr int kSharedK = 3072;                     // hidden size
constexpr int kSharedInter = 384;                  // shared intermediate / TP4
constexpr int kSharedN13 = 2 * kSharedInter;       // gate | up
constexpr int kSharedGroup = 32;
constexpr int kSharedSplits = 8;                   // w8_projection_grouped, K=3072
constexpr int kSharedTiles13 = kSharedN13 / 32;    // 24
constexpr int kSharedPairs = kSharedInter / 32;    // 12 tickets
constexpr int kSharedTiles2 = kSharedK / 32;       // 96
constexpr int kSharedRows13 = kSharedTiles13 * kSharedSplits / kTilesInter;  // 16
constexpr int kSharedRows2 = kSharedTiles2 / (kTiles2 / 2);                    // 4
static_assert(kSharedTiles13 * kSharedSplits % kTilesInter == 0, "W13 rows");
static_assert(kSharedTiles2 % (kTiles2 / 2) == 0, "W2 rows");

struct SharedExpert {
  const half* x;          // [m, 3072]
  const uint8_t* w13;     // W8 g32 gate_up [768, 3072 + 192]
  const uint8_t* w2;      // W8 g32 down [3072, 384 + 24]
  float* partial;         // [8, m, 768] split-K partial sums
  half* act;              // [m, 384]
  unsigned* tickets;      // [12], persistent, zero at allocation
  half* out;              // [m, 3072]
  int m;
};

// w8_project<SplitK, 32> for virtual CTA (vx, vy); red holds 4 * 8 * 32 floats.
template <bool SplitK>
__device__ __forceinline__ void shared_w8_cta(const half* x,
                                              const uint8_t* packed, int m,
                                              int n, int k, int vx, int vy,
                                              int vsplits, float* partial,
                                              half* output, float* red) {
  __syncwarp();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int quad = (lane >> 2) & 3;
  const int tile = vx * 4 + quad;
  const int col_tiles = hy4_w8::padded_n(n) / 8;
  const int split = vy * 4 + warp;
  const int chunks = k / 16;
  const int splits = SplitK ? vsplits * 4 : 4;
  const int begin = chunks * split / splits;
  const int end = chunks * (split + 1) / splits;
  float acc[1][8] = {};
  hy4_w8::warp_project<1, false, false, kSharedGroup>(
      x, packed, m, n, k, min(tile, col_tiles - 1), begin, end, end - begin,
      acc);
  if (tile < col_tiles) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      red[(warp * 8 + hy4_w8::acc_row(i, lane)) * 32 + quad * 8 +
          hy4_w8::acc_col(i, lane)] = acc[0][i];
    }
  }
  __syncthreads();
  const int local_col = threadIdx.x & 31;
  const int col = vx * 32 + local_col;
  if (col >= n) return;
  for (int row = warp; row < m; row += 4) {
    float sum = red[(0 * 8 + row) * 32 + local_col] +
                red[(1 * 8 + row) * 32 + local_col] +
                red[(2 * 8 + row) * 32 + local_col] +
                red[(3 * 8 + row) * 32 + local_col];
    const int offset = row * n + col;
    if constexpr (!SplitK) {
      output[offset] = __float2half_rn(sum);
    } else {
      partial[vy * m * n + offset] = sum;
    }
  }
}

// Shared gate_up CTA vb in [0, 192); the pair's last CTA finishes the pair.
template <class Plan>
__device__ __forceinline__ void shared_gate_up_cta(const SharedExpert& e,
                                                   int vb, float* red,
                                                   Plan& s) {
  const int vx = vb % kSharedTiles13;
  const int vy = vb / kSharedTiles13;
  shared_w8_cta<true>(e.x, e.w13, e.m, kSharedN13, kSharedK, vx, vy,
                      kSharedSplits, e.partial, nullptr, red);
  const int pair = vx % kSharedPairs;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    s.last = (atomicAdd(e.tickets + pair, 1u) & 15u) == 15u;
  }
  __syncthreads();
  if (!s.last) return;
  __threadfence();
  // finish (split order 0..7, one rounding) then vLLM silu_and_mul.
  const int count = e.m * kSharedN13;
  for (int idx = threadIdx.x; idx < e.m * 32; idx += blockDim.x) {
    const int row = idx >> 5;
    const int c = pair * 32 + (idx & 31);
    const float* p = e.partial + row * kSharedN13 + c;
    float g = __ldcg(p);
    float u = __ldcg(p + kSharedInter);
#pragma unroll
    for (int sp = 1; sp < kSharedSplits; ++sp) {
      g += __ldcg(p + sp * count);
      u += __ldcg(p + sp * count + kSharedInter);
    }
    const half up_h = __float2half_rn(u);
    const float gf = __half2float(__float2half_rn(g));
    e.act[row * kSharedInter + c] =
        __hmul(__float2half_rn(gf / (1.0f + expf(-gf))), up_h);
  }
}

template <int Format, int Depth>
__global__ void __launch_bounds__(128, 1)
    hy4_e4_w13_kernel(const half* __restrict__ x,
                      const uint32_t* __restrict__ weights,
                      const uint8_t* __restrict__ metadata,
                      const int32_t* __restrict__ ids, int tokens,
                      int32_t* __restrict__ counters,
                      half* __restrict__ intermediate, float swiglu_limit,
                      SharedExpert shared) {
  extern __shared__ float partial[];  // 4 warps * 8 rows * 64 cols
  __shared__ FusedPlan s;
  const int routes = tokens * kTopK;
  if (blockIdx.y >= routes) {
    shared_gate_up_cta(shared, (blockIdx.y - routes) * kTilesInter + blockIdx.x,
                       partial, s);
    return;
  }
  ring_w13_cta<Format, Depth>(x, weights, metadata, ids, tokens, counters,
                              intermediate, swiglu_limit, partial, s);
}

template <int Format, int Depth>
__global__ void __launch_bounds__(128)
    hy4_e4_w2_kernel(const half* __restrict__ intermediate,
                     const uint32_t* __restrict__ weights,
                     const uint8_t* __restrict__ metadata,
                     const int32_t* __restrict__ ids,
                     const float* __restrict__ topk, int tokens,
                     int32_t* __restrict__ counters,
                     half* __restrict__ route_out, half* __restrict__ out,
                     SharedExpert shared) {
  __shared__ float partial[2][2][8][32];
  __shared__ FusedPlan s;
  __shared__ float s_w[kFusedMaxRoutes];
  const int routes = tokens * kTopK;
  if (blockIdx.y >= routes) {
    shared_w8_cta<false>(shared.act, shared.w2, shared.m, kSharedK,
                         kSharedInter,
                         (blockIdx.y - routes) * (kTiles2 / 2) + blockIdx.x, 0,
                         1, nullptr, shared.out, &partial[0][0][0][0]);
    return;
  }
  ring_w2_cta<Format, Depth>(intermediate, weights, metadata, ids, topk, tokens,
                             counters, route_out, out, partial, s, s_w);
}

// EXPERTS4B: the EXPERTS4 kernels with the 5..8-token plan. Grids grow by the
// extra routes only; the shared roles already cover m <= 8 (warp_project's one
// 8-row block), with the same splits as w8_projection_grouped at any m.
template <int Format, int Depth>
__global__ void __launch_bounds__(128, 1)
    hy4_e4b_w13_kernel(const half* __restrict__ x,
                       const uint32_t* __restrict__ weights,
                       const uint8_t* __restrict__ metadata,
                       const int32_t* __restrict__ ids, int tokens,
                       int32_t* __restrict__ counters,
                       half* __restrict__ intermediate, float swiglu_limit,
                       SharedExpert shared) {
  extern __shared__ float partial[];  // 4 warps * 8 rows * 64 cols
  __shared__ WidePlan s;
  const int routes = tokens * kTopK;
  if (blockIdx.y >= routes) {
    shared_gate_up_cta(shared, (blockIdx.y - routes) * kTilesInter + blockIdx.x,
                       partial, s);
    return;
  }
  ring_w13_cta<Format, Depth>(x, weights, metadata, ids, tokens, counters,
                              intermediate, swiglu_limit, partial, s);
}

template <int Format, int Depth>
__global__ void __launch_bounds__(128)
    hy4_e4b_w2_kernel(const half* __restrict__ intermediate,
                      const uint32_t* __restrict__ weights,
                      const uint8_t* __restrict__ metadata,
                      const int32_t* __restrict__ ids,
                      const float* __restrict__ topk, int tokens,
                      int32_t* __restrict__ counters,
                      half* __restrict__ route_out, half* __restrict__ out,
                      SharedExpert shared) {
  __shared__ float partial[2][2][8][32];
  __shared__ WidePlan s;
  __shared__ float s_w[WidePlan::kRoutes];
  const int routes = tokens * kTopK;
  if (blockIdx.y >= routes) {
    shared_w8_cta<false>(shared.act, shared.w2, shared.m, kSharedK,
                         kSharedInter,
                         (blockIdx.y - routes) * (kTiles2 / 2) + blockIdx.x, 0,
                         1, nullptr, shared.out, &partial[0][0][0][0]);
    return;
  }
  ring_w2_cta<Format, Depth>(intermediate, weights, metadata, ids, topk, tokens,
                             counters, route_out, out, partial, s, s_w);
}

// Depth 1 measured best on the model's compact stats (EXPERTS3 report).
constexpr int kRingDepth = 1;

template <int Format>
void launch_ring(const at::Tensor& input, const at::Tensor& w13,
                 const at::Tensor& s13, const at::Tensor& w2,
                 const at::Tensor& s2, const at::Tensor& ids,
                 const at::Tensor& topk, at::Tensor& plan,
                 at::Tensor& intermediate, at::Tensor& route_out,
                 at::Tensor& out, int tokens, float swiglu_limit,
                 cudaStream_t stream) {
  const int routes = tokens * kTopK;
  constexpr size_t smem = static_cast<size_t>(4) * 8 * 64 * sizeof(float);
  hy4_ring_w13_kernel<Format, kRingDepth>
      <<<dim3(kTilesInter, routes), 128, smem, stream>>>(
          reinterpret_cast<const half*>(input.const_data_ptr()),
          reinterpret_cast<const uint32_t*>(w13.const_data_ptr()),
          reinterpret_cast<const uint8_t*>(s13.const_data_ptr()),
          ids.const_data_ptr<int32_t>(), tokens,
          plan.mutable_data_ptr<int32_t>(),
          reinterpret_cast<half*>(intermediate.mutable_data_ptr()),
          swiglu_limit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  hy4_ring_w2_kernel<Format == 1 ? 0 : 2, kRingDepth>
      <<<dim3(kTiles2 / 2, routes), 128, 0, stream>>>(
          reinterpret_cast<const half*>(intermediate.const_data_ptr()),
          reinterpret_cast<const uint32_t*>(w2.const_data_ptr()),
          reinterpret_cast<const uint8_t*>(s2.const_data_ptr()),
          ids.const_data_ptr<int32_t>(), topk.const_data_ptr<float>(), tokens,
          plan.mutable_data_ptr<int32_t>(),
          reinterpret_cast<half*>(route_out.mutable_data_ptr()),
          reinterpret_cast<half*>(out.mutable_data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Format, bool Wide>
void launch_e4(const at::Tensor& input, const at::Tensor& w13,
               const at::Tensor& s13, const at::Tensor& w2,
               const at::Tensor& s2, const at::Tensor& ids,
               const at::Tensor& topk, at::Tensor& plan,
               at::Tensor& intermediate, at::Tensor& route_out,
               at::Tensor& out, int tokens, float swiglu_limit,
               const SharedExpert& shared, cudaStream_t stream) {
  const int routes = tokens * kTopK;
  constexpr size_t smem = static_cast<size_t>(4) * 8 * 64 * sizeof(float);
  auto* w13_kernel = Wide ? hy4_e4b_w13_kernel<Format, kRingDepth>
                          : hy4_e4_w13_kernel<Format, kRingDepth>;
  auto* w2_kernel = Wide ? hy4_e4b_w2_kernel<Format == 1 ? 0 : 2, kRingDepth>
                         : hy4_e4_w2_kernel<Format == 1 ? 0 : 2, kRingDepth>;
  w13_kernel<<<dim3(kTilesInter, routes + kSharedRows13), 128, smem, stream>>>(
          reinterpret_cast<const half*>(input.const_data_ptr()),
          reinterpret_cast<const uint32_t*>(w13.const_data_ptr()),
          reinterpret_cast<const uint8_t*>(s13.const_data_ptr()),
          ids.const_data_ptr<int32_t>(), tokens,
          plan.mutable_data_ptr<int32_t>(),
          reinterpret_cast<half*>(intermediate.mutable_data_ptr()),
          swiglu_limit, shared);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  w2_kernel<<<dim3(kTiles2 / 2, routes + kSharedRows2), 128, 0, stream>>>(
          reinterpret_cast<const half*>(intermediate.const_data_ptr()),
          reinterpret_cast<const uint32_t*>(w2.const_data_ptr()),
          reinterpret_cast<const uint8_t*>(s2.const_data_ptr()),
          ids.const_data_ptr<int32_t>(), topk.const_data_ptr<float>(), tokens,
          plan.mutable_data_ptr<int32_t>(),
          reinterpret_cast<half*>(route_out.mutable_data_ptr()),
          reinterpret_cast<half*>(out.mutable_data_ptr()), shared);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Format>
void launch_fold(const at::Tensor& input, const at::Tensor& w13,
                 const at::Tensor& s13, const at::Tensor& w2,
                 const at::Tensor& s2, const at::Tensor& ids,
                 const at::Tensor& topk, at::Tensor& counters,
                 at::Tensor& intermediate, at::Tensor& route_out,
                 at::Tensor& out, int tokens, float swiglu_limit,
                 cudaStream_t stream) {
  const int routes = tokens * kTopK;
  constexpr size_t smem = static_cast<size_t>(4) * 8 * 64 * sizeof(float);
  hy4_fold_w13_swiglu_kernel<Format, 4, 1>
      <<<dim3(kTilesInter, routes), 128, smem, stream>>>(
          reinterpret_cast<const half*>(input.const_data_ptr()),
          reinterpret_cast<const uint32_t*>(w13.const_data_ptr()),
          reinterpret_cast<const uint8_t*>(s13.const_data_ptr()),
          ids.const_data_ptr<int32_t>(), tokens,
          counters.mutable_data_ptr<int32_t>(),
          reinterpret_cast<half*>(intermediate.mutable_data_ptr()),
          swiglu_limit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  hy4_fold_w2_split2_kernel<Format == 1 ? 0 : 2, 1>
      <<<dim3(kTiles2 / 2, routes), 128, 0, stream>>>(
          reinterpret_cast<const half*>(intermediate.const_data_ptr()),
          reinterpret_cast<const uint32_t*>(w2.const_data_ptr()),
          reinterpret_cast<const uint8_t*>(s2.const_data_ptr()),
          ids.const_data_ptr<int32_t>(), topk.const_data_ptr<float>(), tokens,
          counters.mutable_data_ptr<int32_t>(),
          reinterpret_cast<half*>(route_out.mutable_data_ptr()),
          reinterpret_cast<half*>(out.mutable_data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_tensor(const at::Tensor& tensor, at::ScalarType dtype,
                  at::IntArrayRef shape, const at::Device& device,
                  const char* name, uintptr_t alignment = 16) {
  TORCH_CHECK(tensor.is_cuda() && tensor.device() == device,
              "hy4_awq_union_moe: ", name, " must be CUDA on input device");
  TORCH_CHECK(tensor.scalar_type() == dtype && tensor.sizes() == shape &&
                  tensor.is_contiguous(),
              "hy4_awq_union_moe: ", name, " dtype/shape/contiguity mismatch");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(tensor.const_data_ptr()) % alignment == 0,
      "hy4_awq_union_moe: ", name, " alignment mismatch");
}

void hy4_awq_union_moe_sm70_out(
    at::Tensor out, at::Tensor intermediate, at::Tensor route_out,
    at::Tensor plan, const at::Tensor& input, const at::Tensor& w13,
    const at::Tensor& s13, const at::Tensor& w2, const at::Tensor& s2,
    const at::Tensor& ids, const at::Tensor& topk, double swiglu_limit,
    int64_t split, int64_t w2_mode, int64_t pack_rows) {
  const auto device = input.device();
  TORCH_CHECK(input.dim() == 2 && input.size(0) >= 1 &&
                  input.size(0) <= kMaxTokens && input.size(1) == kK13,
              "hy4_awq_union_moe: input must be [1..", kMaxTokens, ", 1536]");
  const int tokens = static_cast<int>(input.size(0));
  const int routes = tokens * kTopK;
  TORCH_CHECK(split == 0 || split == 4 || split == 6 || split == 8 ||
                  split == 12,
              "hy4_awq_union_moe: split must be 0, 4, 6, 8, or 12");
  TORCH_CHECK(w2_mode == 0 || w2_mode == 1 || w2_mode == 2,
              "hy4_awq_union_moe: w2_mode must be 0, 1, or 2");
  TORCH_CHECK(pack_rows == 0 ||
                  (pack_rows >= 8 && pack_rows <= kRowCap && pack_rows % 8 == 0),
              "hy4_awq_union_moe: invalid pack_rows");

  check_tensor(input, at::kHalf, {tokens, kK13}, device, "input");
  check_tensor(out, at::kHalf, {tokens, kN2}, device, "output");
  check_tensor(intermediate, at::kHalf, {routes, kInter}, device,
               "intermediate");
  check_tensor(route_out, at::kHalf, {routes, kN2}, device, "route_out");
  TORCH_CHECK(plan.is_cuda() && plan.device() == device &&
                  plan.scalar_type() == at::kInt && plan.is_contiguous() &&
                  plan.numel() >= kPlanInts,
              "hy4_awq_union_moe: plan buffer too small (need >= ", kPlanInts,
              " int32)");
  check_tensor(w13, at::kInt, {kExperts, kK13, kN13 / 8}, device, "W13");
  check_tensor(w2, at::kInt, {kExperts, kK2, kN2 / 8}, device, "W2");
  const bool compact = s13.scalar_type() == at::kByte;
  if (compact) {
    check_tensor(s13, at::kByte, {kExperts, kK13 / 32, kN13, 3}, device,
                 "W13 compact scales");
    check_tensor(s2, at::kByte, {kExperts, kK2 / 32, kN2, 3}, device,
                 "W2 compact scales");
  } else {
    check_tensor(s13, at::kInt, {kExperts, kK13 / 32, kN13}, device,
                 "W13 scales");
    check_tensor(s2, at::kInt, {kExperts, kK2 / 32, kN2}, device, "W2 scales");
  }
  check_tensor(ids, at::kInt, {tokens, kTopK}, device, "ids", 4);
  check_tensor(topk, at::kFloat, {tokens, kTopK}, device, "topk", 4);

  const c10::cuda::CUDAGuard guard(device);
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0,
              "hy4_awq_union_moe requires SM70");
  const auto stream = at::cuda::getCurrentCUDAStream();

  int rb = (tokens + 7) / 8;
  if (pack_rows > 0) rb = std::min(rb, static_cast<int>(pack_rows / 8));
  hy4_grouped_plan_kernel<<<1, kExperts, 0, stream>>>(
      ids.const_data_ptr<int32_t>(), routes, rb * 8,
      plan.mutable_data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int w13_split = split == 0 ? 4 : static_cast<int>(split);
  const float limit_f = static_cast<float>(swiglu_limit);
  if (compact) {
    dispatch_rb<1, 0>(rb, input, w13, s13, w2, s2, ids, topk, plan,
                      intermediate, route_out, out, tokens, w13_split,
                      static_cast<int>(w2_mode), limit_f, stream);
  } else {
    dispatch_rb<2, 2>(rb, input, w13, s13, w2, s2, ids, topk, plan,
                      intermediate, route_out, out, tokens, w13_split,
                      static_cast<int>(w2_mode), limit_f, stream);
  }
}

// Checks the fold entries' arguments; returns whether the stats are compact.
bool check_fold_args(const at::Tensor& out, const at::Tensor& intermediate,
                     const at::Tensor& route_out, const at::Tensor& plan,
                     const at::Tensor& input, const at::Tensor& w13,
                     const at::Tensor& s13, const at::Tensor& w2,
                     const at::Tensor& s2, const at::Tensor& ids,
                     const at::Tensor& topk, int64_t plan_ints,
                     int max_tokens = kFusedMaxTokens) {
  const auto device = input.device();
  TORCH_CHECK(input.dim() == 2 && input.size(0) >= 1 &&
                  input.size(0) <= max_tokens && input.size(1) == kK13,
              "hy4_awq_union_moe_fold: input must be [1..", max_tokens,
              ", 1536]");
  const int tokens = static_cast<int>(input.size(0));
  const int routes = tokens * kTopK;
  check_tensor(input, at::kHalf, {tokens, kK13}, device, "input");
  check_tensor(out, at::kHalf, {tokens, kN2}, device, "output");
  check_tensor(intermediate, at::kHalf, {routes, kInter}, device,
               "intermediate");
  check_tensor(route_out, at::kHalf, {routes, kN2}, device, "route_out");
  TORCH_CHECK(plan.is_cuda() && plan.device() == device &&
                  plan.scalar_type() == at::kInt && plan.is_contiguous() &&
                  plan.numel() >= plan_ints,
              "hy4_awq_union_moe_fold: plan buffer too small");
  check_tensor(w13, at::kInt, {kExperts, kK13, kN13 / 8}, device, "W13");
  check_tensor(w2, at::kInt, {kExperts, kK2, kN2 / 8}, device, "W2");
  const bool compact = s13.scalar_type() == at::kByte;
  if (compact) {
    check_tensor(s13, at::kByte, {kExperts, kK13 / 32, kN13, 3}, device,
                 "W13 compact scales");
    check_tensor(s2, at::kByte, {kExperts, kK2 / 32, kN2, 3}, device,
                 "W2 compact scales");
  } else {
    check_tensor(s13, at::kInt, {kExperts, kK13 / 32, kN13}, device,
                 "W13 scales");
    check_tensor(s2, at::kInt, {kExperts, kK2 / 32, kN2}, device, "W2 scales");
  }
  check_tensor(ids, at::kInt, {tokens, kTopK}, device, "ids", 4);
  check_tensor(topk, at::kFloat, {tokens, kTopK}, device, "topk", 4);
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0,
              "hy4_awq_union_moe_fold requires SM70");
  return compact;
}

// Plan and reduce folding entry (tokens 1..4): default W13/W2 bodies, inline
// plan, W2-side weighted reduction. plan is reused as 24 int32 tile tickets.
void hy4_awq_union_moe_fold_sm70_out(
    at::Tensor out, at::Tensor intermediate, at::Tensor route_out,
    at::Tensor plan, const at::Tensor& input, const at::Tensor& w13,
    const at::Tensor& s13, const at::Tensor& w2, const at::Tensor& s2,
    const at::Tensor& ids, const at::Tensor& topk, double swiglu_limit) {
  const c10::cuda::CUDAGuard guard(input.device());
  const bool compact = check_fold_args(out, intermediate, route_out, plan,
                                       input, w13, s13, w2, s2, ids, topk,
                                       kFusedTiles2);
  const int tokens = static_cast<int>(input.size(0));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const float limit_f = static_cast<float>(swiglu_limit);
  if (compact) {
    launch_fold<1>(input, w13, s13, w2, s2, ids, topk, plan, intermediate,
                   route_out, out, tokens, limit_f, stream);
  } else {
    launch_fold<2>(input, w13, s13, w2, s2, ids, topk, plan, intermediate,
                   route_out, out, tokens, limit_f, stream);
  }
}

// EXPERTS3 ring entry: same contract and output bits as the fold entry.
void experts3_ring_fold_sm70_out(
    at::Tensor out, at::Tensor intermediate, at::Tensor route_out,
    at::Tensor plan, const at::Tensor& input, const at::Tensor& w13,
    const at::Tensor& s13, const at::Tensor& w2, const at::Tensor& s2,
    const at::Tensor& ids, const at::Tensor& topk, double swiglu_limit) {
  const c10::cuda::CUDAGuard guard(input.device());
  const bool compact = check_fold_args(out, intermediate, route_out, plan,
                                       input, w13, s13, w2, s2, ids, topk,
                                       kFusedTiles2);
  const int tokens = static_cast<int>(input.size(0));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const float limit_f = static_cast<float>(swiglu_limit);
  if (compact) {
    launch_ring<1>(input, w13, s13, w2, s2, ids, topk, plan, intermediate,
                   route_out, out, tokens, limit_f, stream);
  } else {
    launch_ring<2>(input, w13, s13, w2, s2, ids, topk, plan, intermediate,
                   route_out, out, tokens, limit_f, stream);
  }
}

// EXPERTS4 entry: the EXPERTS3 ring op plus the shared expert in the same two
// launches. Routed outputs equal experts3_ring_fold_out's; the returned shared
// output equals w8 gate_up -> silu_and_mul -> w8 down on shared_x.
// EXPERTS4B: 5..8 tokens run the WidePlan kernels; their routed outputs equal
// awq_union_moe_out(split=0|4, w2_mode=2, pack_rows=0), whose rows do not
// depend on M.
at::Tensor experts4_ring_shared_sm70_out(
    at::Tensor out, at::Tensor intermediate, at::Tensor route_out,
    at::Tensor plan, const at::Tensor& input, const at::Tensor& w13,
    const at::Tensor& s13, const at::Tensor& w2, const at::Tensor& s2,
    const at::Tensor& ids, const at::Tensor& topk, double swiglu_limit,
    const at::Tensor& shared_x, const at::Tensor& shared_w13,
    const at::Tensor& shared_w2, at::Tensor tickets) {
  const c10::cuda::CUDAGuard guard(input.device());
  const bool compact = check_fold_args(out, intermediate, route_out, plan,
                                       input, w13, s13, w2, s2, ids, topk,
                                       kFusedTiles2, kWideMaxTokens);
  const int tokens = static_cast<int>(input.size(0));
  const bool wide = tokens > kFusedMaxTokens;
  const auto device = input.device();
  check_tensor(shared_x, at::kHalf, {tokens, kSharedK}, device, "shared_x");
  TORCH_CHECK(shared_w13.is_cuda() && shared_w13.device() == device &&
                  shared_w13.scalar_type() == at::kByte &&
                  shared_w13.is_contiguous() &&
                  shared_w13.numel() ==
                      hy4_w8::matrix_bytes(kSharedN13, kSharedK, kSharedGroup) &&
                  reinterpret_cast<uintptr_t>(shared_w13.const_data_ptr()) % 16 == 0,
              "experts4: shared gate_up must be W8 g32 [768, 3072]");
  TORCH_CHECK(shared_w2.is_cuda() && shared_w2.device() == device &&
                  shared_w2.scalar_type() == at::kByte &&
                  shared_w2.is_contiguous() &&
                  shared_w2.numel() ==
                      hy4_w8::matrix_bytes(kSharedK, kSharedInter, kSharedGroup) &&
                  reinterpret_cast<uintptr_t>(shared_w2.const_data_ptr()) % 16 == 0,
              "experts4: shared down must be W8 g32 [3072, 384]");
  TORCH_CHECK(tickets.is_cuda() && tickets.device() == device &&
                  tickets.scalar_type() == at::kInt && tickets.is_contiguous() &&
                  tickets.numel() >= kSharedPairs,
              "experts4: tickets must be >= 12 int32 on the input device");
  auto partial = at::empty({kSharedSplits, tokens * kSharedN13},
                           shared_x.options().dtype(at::kFloat));
  auto act = at::empty({tokens, kSharedInter}, shared_x.options());
  auto shared_out = at::empty({tokens, kSharedK}, shared_x.options());
  const SharedExpert shared{
      reinterpret_cast<const half*>(shared_x.const_data_ptr()),
      shared_w13.const_data_ptr<uint8_t>(),
      shared_w2.const_data_ptr<uint8_t>(),
      partial.mutable_data_ptr<float>(),
      reinterpret_cast<half*>(act.mutable_data_ptr()),
      reinterpret_cast<unsigned*>(tickets.mutable_data_ptr<int32_t>()),
      reinterpret_cast<half*>(shared_out.mutable_data_ptr()),
      tokens};
  const auto stream = at::cuda::getCurrentCUDAStream();
  const float limit_f = static_cast<float>(swiglu_limit);
  if (compact && wide) {
    launch_e4<1, true>(input, w13, s13, w2, s2, ids, topk, plan, intermediate,
                       route_out, out, tokens, limit_f, shared, stream);
  } else if (compact) {
    launch_e4<1, false>(input, w13, s13, w2, s2, ids, topk, plan, intermediate,
                        route_out, out, tokens, limit_f, shared, stream);
  } else if (wide) {
    launch_e4<2, true>(input, w13, s13, w2, s2, ids, topk, plan, intermediate,
                       route_out, out, tokens, limit_f, shared, stream);
  } else {
    launch_e4<2, false>(input, w13, s13, w2, s2, ids, topk, plan, intermediate,
                        route_out, out, tokens, limit_f, shared, stream);
  }
  return shared_out;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(hy4_sm70, m) {
  m.def(
      "awq_union_moe_out(Tensor(a!) out, Tensor(b!) intermediate, "
      "Tensor(c!) route_out, Tensor(d!) plan, Tensor input, Tensor w13, "
      "Tensor s13, Tensor w2, Tensor s2, Tensor ids, Tensor topk, "
      "float swiglu_limit=10.0, int split=0, int w2_mode=2, "
      "int pack_rows=0) -> ()");
  m.def(
      "awq_union_moe_fold_out(Tensor(a!) out, Tensor(b!) intermediate, "
      "Tensor(c!) route_out, Tensor(d!) plan, Tensor input, Tensor w13, "
      "Tensor s13, Tensor w2, Tensor s2, Tensor ids, Tensor topk, "
      "float swiglu_limit=10.0) -> ()");
  m.def(
      "experts3_ring_fold_out(Tensor(a!) out, Tensor(b!) intermediate, "
      "Tensor(c!) route_out, Tensor(d!) plan, Tensor input, Tensor w13, "
      "Tensor s13, Tensor w2, Tensor s2, Tensor ids, Tensor topk, "
      "float swiglu_limit=10.0) -> ()");
  m.def(
      "experts4_ring_shared_out(Tensor(a!) out, Tensor(b!) intermediate, "
      "Tensor(c!) route_out, Tensor(d!) plan, Tensor input, Tensor w13, "
      "Tensor s13, Tensor w2, Tensor s2, Tensor ids, Tensor topk, "
      "float swiglu_limit, Tensor shared_x, Tensor shared_w13, "
      "Tensor shared_w2, Tensor(e!) tickets) -> Tensor");
}

TORCH_LIBRARY_IMPL(hy4_sm70, CUDA, m) {
  m.impl("awq_union_moe_out", &hy4_awq_union_moe_sm70_out);
  m.impl("awq_union_moe_fold_out", &hy4_awq_union_moe_fold_sm70_out);
  m.impl("experts3_ring_fold_out", &experts3_ring_fold_sm70_out);
  m.impl("experts4_ring_shared_out", &experts4_ring_shared_sm70_out);
}
