// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// F7 Stage 5 (HY4_F7_IDX_FUSE) SM70 IndexPool kernels for Hy4-flash:
//   - K3a (idx_pool_update_sm70): single-launch cooperative token_key_proj +
//     token_key_norm + ring/pool gather/pool/key_norm/no-FMA RoPE/store for both
//     decode (width=1, skips pooling on non-closing rows on device) and verify
//     (width>1, writes staged buffer and publishes closing groups). Handles 1D
//     and 3-coord MRoPE positions and dcp_world >= 1 (4 cache shards).
//   - K3b (idx_query_prep_sm70): single-launch long-context (>2048) query_proj +
//     inline no-FMA RoPE + head_weight_proj + FP32 scaling.
//
// Volta sm_70 + nvcc 12.8 mandatory correctness rules:
//   Rule 1: Every global read of an inter-phase buffer written before grid.sync()
//           MUST use __ldcg (ld.global.cg) to bypass Volta non-coherent L1.
//   Rule 2: Every shared-memory reduction pointer passed to device helpers MUST
//           be volatile float* to prevent nvcc 12.8 dead-load elision across BAR.SYNC.

#include "hy4_w8_sm70.cuh"
#include "hy4_next_mma.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <optional>

namespace cg = cooperative_groups;

namespace {

constexpr int kThreads = 256;
constexpr int kHidden = 3072;
constexpr int kQRank = 1024;
constexpr int kIdxDim = 128;
constexpr int kIdxHeads = 16;
constexpr int kIdxRope = 64;
constexpr int kIdxRatio = 4;
constexpr int kTkTileN = 16;      // 128 / 16 = 8 N-tiles
constexpr int kTkNTiles = kIdxDim / kTkTileN;  // 8
constexpr int kTkSplits = 16;     // Fixed across RB=1,2,3 for 0-bit row invariance (8*16 = 128 CTAs)
constexpr int kTkKPerSplit = kHidden / kTkSplits;  // 192 (6 k=32 steps)



__device__ __forceinline__ float ld_f32_cg(const float* p) {
  return __ldcg(p);
}

__device__ __forceinline__ int mrope_axis_for_pair(
    int pair, int pos_dims, int sec1, int sec2) {
  if (pos_dims == 1) return 0;
  // Matches rotary.py:36-41:
  //   axis 1 at offsets 1, 4, 7, ... < sec1*3
  //   axis 2 at offsets 2, 5, 8, ... < sec2*3
  int rem = pair % 3;
  if (rem == 1 && pair < sec1 * 3) return 1;
  if (rem == 2 && pair < sec2 * 3) return 2;
  return 0;
}

// =============================================================================
// K3a: Cooperative IndexPool Update (Decode & Verify)
// =============================================================================

struct IdxUpdateParams {
  // Inputs
  const half* hidden;          // [M, 3072] FP16
  const uint8_t* tk8;
  const half* w_tk;            // [128, 3072] FP16
  const half* gamma_tk;        // [128] FP16
  const float* w_kv;           // [128, 128] FP32
  const float* w_kg;           // [128, 128] FP32
  const float* ape;            // [4, 128] FP32
  const float* gamma_key;      // [128] FP32
  const float* inv_freq;       // [32] FP32
  const int64_t* positions;    // [M] if pos_dims==1 else [3, M] (row-major [3, M])
  const int32_t* seq_lens;     // [batch] (batch == M for decode, M/width for verify)
  const int64_t* slots;        // [M] int64 or int32? Checked on host -> pass int64 or int32
  const int32_t* slots_i32;    // non-null if slots is int32
  const int32_t* pool_table;   // [batch, pool_table_stride]
  const int32_t* ring_table;   // [batch, ring_table_stride]

  // Scratch & Outputs
  float* tk_partials;          // [M, 16, 128] FP32
  float* staged;               // [M, 128 + pos_dims] FP32 (verify only, nullptr for decode)
  half* pool;                  // [num_pool_pages, page, 128] FP16
  float* ring;                 // [num_ring_pages, 4, 128 + pos_dims] FP32

  int m;
  int width;                   // 1 for decode, >1 (e.g. 6) for verify
  int closes;                  // 1 for decode, cdiv(width, 4) for verify
  int pos_dims;                // 1 or 3
  int pos_stride_axis;         // M (when pos_dims==3)
  int sec1;                    // mrope_section[1]
  int sec2;                    // mrope_section[2]
  int pool_table_stride;
  int ring_table_stride;
  int64_t pool_page_stride;
  int64_t pool_row_stride;
  int64_t ring_page_stride;
  int64_t ring_row_stride;
  int page;
  int pages_per_block;
  int dcp_world;
  int dcp_rank;
  float eps;
};

__device__ __forceinline__ int64_t load_slot(const IdxUpdateParams& p, int idx) {
  if (p.slots != nullptr) return p.slots[idx];
  return static_cast<int64_t>(p.slots_i32[idx]);
}

// Compute pooled + key_norm + no-FMA RoPE for one closing group already in smem_raw[4][128],
// and store directly to pool[phys_page, local_group % page, 0..127].
// Uses all 256 threads (8 warps); 8 threads per output feature d in 4 iterations of 32 features.
__device__ __forceinline__ void pool_norm_rotate_store(
    const IdxUpdateParams& p,
    const float smem_raw[kIdxRatio][kIdxDim],
    const int64_t first_pos[3],
    int req_idx,
    int group,
    volatile float* smem_red,
    float* smem_pooled_f,
    half* smem_pooled_h) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int sub = tid & 7;      // 0..7 along K (each handles 16 floats = 4 float4s)
  const int row_in_iter = tid >> 3;  // 0..31

  // Compute all 128 output features in 4 iterations of 32 features (coalesced float4 loads on W_kv/W_kg).
#pragma unroll
  for (int iter = 0; iter < 4; ++iter) {
    const int d = iter * 32 + row_in_iter;
    const float4* wkv4 = reinterpret_cast<const float4*>(p.w_kv + d * kIdxDim);
    const float4* wkg4 = reinterpret_cast<const float4*>(p.w_kg + d * kIdxDim);

    float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
    float g0 = 0.f, g1 = 0.f, g2 = 0.f, g3 = 0.f;

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int k4 = sub + j * 8;  // 0..31 (float4 index -> k = k4*4)
      const int k = k4 * 4;
      float4 kv = __ldg(wkv4 + k4);
      float4 kg = __ldg(wkg4 + k4);

      float r00 = smem_raw[0][k + 0], r01 = smem_raw[0][k + 1], r02 = smem_raw[0][k + 2], r03 = smem_raw[0][k + 3];
      v0 = fmaf(r00, kv.x, fmaf(r01, kv.y, fmaf(r02, kv.z, fmaf(r03, kv.w, v0))));
      g0 = fmaf(r00, kg.x, fmaf(r01, kg.y, fmaf(r02, kg.z, fmaf(r03, kg.w, g0))));

      float r10 = smem_raw[1][k + 0], r11 = smem_raw[1][k + 1], r12 = smem_raw[1][k + 2], r13 = smem_raw[1][k + 3];
      v1 = fmaf(r10, kv.x, fmaf(r11, kv.y, fmaf(r12, kv.z, fmaf(r13, kv.w, v1))));
      g1 = fmaf(r10, kg.x, fmaf(r11, kg.y, fmaf(r12, kg.z, fmaf(r13, kg.w, g1))));

      float r20 = smem_raw[2][k + 0], r21 = smem_raw[2][k + 1], r22 = smem_raw[2][k + 2], r23 = smem_raw[2][k + 3];
      v2 = fmaf(r20, kv.x, fmaf(r21, kv.y, fmaf(r22, kv.z, fmaf(r23, kv.w, v2))));
      g2 = fmaf(r20, kg.x, fmaf(r21, kg.y, fmaf(r22, kg.z, fmaf(r23, kg.w, g2))));

      float r30 = smem_raw[3][k + 0], r31 = smem_raw[3][k + 1], r32 = smem_raw[3][k + 2], r33 = smem_raw[3][k + 3];
      v3 = fmaf(r30, kv.x, fmaf(r31, kv.y, fmaf(r32, kv.z, fmaf(r33, kv.w, v3))));
      g3 = fmaf(r30, kg.x, fmaf(r31, kg.y, fmaf(r32, kg.z, fmaf(r33, kg.w, g3))));
    }

    // Reduce across 8-thread subgroup (fixed butterfly order).
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
      v0 += __shfl_down_sync(0xffffffffu, v0, off, 8);
      v1 += __shfl_down_sync(0xffffffffu, v1, off, 8);
      v2 += __shfl_down_sync(0xffffffffu, v2, off, 8);
      v3 += __shfl_down_sync(0xffffffffu, v3, off, 8);
      g0 += __shfl_down_sync(0xffffffffu, g0, off, 8);
      g1 += __shfl_down_sync(0xffffffffu, g1, off, 8);
      g2 += __shfl_down_sync(0xffffffffu, g2, off, 8);
      g3 += __shfl_down_sync(0xffffffffu, g3, off, 8);
    }

    if (sub == 0) {
      g0 += __ldg(p.ape + 0 * kIdxDim + d);
      g1 += __ldg(p.ape + 1 * kIdxDim + d);
      g2 += __ldg(p.ape + 2 * kIdxDim + d);
      g3 += __ldg(p.ape + 3 * kIdxDim + d);
      float mx = fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
      float e0 = expf(g0 - mx);
      float e1 = expf(g1 - mx);
      float e2 = expf(g2 - mx);
      float e3 = expf(g3 - mx);
      float inv = 1.0f / (e0 + e1 + e2 + e3);
      smem_pooled_f[d] = (v0 * e0 + v1 * e1 + v2 * e2 + v3 * e3) * inv;
    }
  }
  __syncthreads();

  // FP32 RMSNorm (key_norm) over 128 elements -> FP16 round.
  float sq = 0.f;
  if (tid < kIdxDim) {
    float x = smem_pooled_f[tid];
    sq = __fmul_rn(x, x);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    sq = __fadd_rn(sq, __shfl_down_sync(0xffffffffu, sq, off));
  }
  if (lane == 0) smem_red[warp] = sq;
  __syncthreads();
  if (tid == 0) {
    float tot = __fadd_rn(__fadd_rn(smem_red[0], smem_red[1]),
                          __fadd_rn(smem_red[2], smem_red[3]));
    smem_red[8] = rsqrtf(tot / static_cast<float>(kIdxDim) + p.eps);
  }
  __syncthreads();
  const float rrms = smem_red[8];
  if (tid < kIdxDim) {
    float norm_v = smem_pooled_f[tid] * rrms * __ldg(p.gamma_key + tid);
    smem_pooled_h[tid] = __float2half(norm_v);
  }
  __syncthreads();

  // Exact no-FMA RoPE on [64..127] and direct store to k_cache pool.
  if (tid < kIdxDim) {
    half out_h;
    if (tid < kIdxDim - kIdxRope) {
      out_h = smem_pooled_h[tid];
    } else {
      const int r = tid - (kIdxDim - kIdxRope);
      const int pair = r >> 1;
      const int axis = mrope_axis_for_pair(pair, p.pos_dims, p.sec1, p.sec2);
      const float pos_f = static_cast<float>(first_pos[axis]);
      const float angle = __fmul_rn(pos_f, __ldg(p.inv_freq + pair));
      const float c = cosf(angle);
      const float s = sinf(angle);
      const float even = __half2float(smem_pooled_h[(kIdxDim - kIdxRope) + 2 * pair]);
      const float odd = __half2float(smem_pooled_h[(kIdxDim - kIdxRope) + 2 * pair + 1]);
      float rot;
      if ((r & 1) == 0) {
        rot = __fsub_rn(__fmul_rn(even, c), __fmul_rn(odd, s));
      } else {
        rot = __fadd_rn(__fmul_rn(odd, c), __fmul_rn(even, s));
      }
      out_h = __float2half(rot);
    }
    const int local_group = group / p.dcp_world;
    const int logical_page = local_group / p.page;
    const int block = p.pool_table[req_idx * p.pool_table_stride + logical_page / p.pages_per_block];
    const int64_t phys_page =
        static_cast<int64_t>(block) * p.pages_per_block + (logical_page % p.pages_per_block);
    half* pool_dst = p.pool + phys_page * p.pool_page_stride +
                     static_cast<int64_t>(local_group % p.page) * p.pool_row_stride;
    pool_dst[tid] = out_h;
  }
}

template <int RB, bool IsVerify, bool W8>
__global__ __launch_bounds__(kThreads, 2)
void hy4_idx_pool_update_coop_kernel(IdxUpdateParams p) {
  cg::grid_group grid = cg::this_grid();
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bid = blockIdx.x;

  const int wn = warp & 1;         // 0..1 (2 warps along N -> 16 cols/CTA)

  // Shared memory: 2048 B for Phase 0 MMA red + Phase 1/2 raw[4][128] (2048 B) + pooled/red
  __shared__ union {
    float mma_red[4][RB * 8][kTkTileN];
    struct {
      float raw[kIdxRatio][kIdxDim];
      float pooled_f[kIdxDim];
      float red[16];
      half pooled_h[kIdxDim];
      int64_t first_pos[4];
    } pool;
  } smem;

  // =========================================================================
  // Phase 0: token_key_projection hidden[M, 3072] x W_tk[128, 3072]^T
  // Fixed grid: 8 N-tiles x 16 K-splits = 128 CTAs.
  // =========================================================================
  if (bid < kTkNTiles * kTkSplits) {
    const int nt = bid % kTkNTiles;
    const int sp = bid / kTkNTiles;
    const int n_base = nt * kTkTileN + wn * 8;
    const int k_start = sp * kTkKPerSplit;
    const int k_end = k_start + kTkKPerSplit;


    volatile float* red_base = &smem.mma_red[0][0][0];
    if constexpr (W8) {
      hy4_w8::slice16<RB, false, 4>(
          p.hidden, p.tk8, p.m, kIdxDim, kHidden, nt * kTkTileN,
          k_start, k_end, red_base);
    } else {
      hy4_next::project_tile<RB, 16>(
          p.hidden, p.w_tk + static_cast<int64_t>(n_base) * kHidden,
          p.m, kHidden, k_start, k_end, red_base);
    }

    const int total = p.m * kTkTileN;
    for (int idx = tid; idx < total; idx += kThreads) {
      const int rm = idx / kTkTileN;
      const int cm = idx - rm * kTkTileN;
      float s0 = red_base[(0 * (RB * 8) + rm) * kTkTileN + cm];
      float s1 = red_base[(1 * (RB * 8) + rm) * kTkTileN + cm];
      float s2 = red_base[(2 * (RB * 8) + rm) * kTkTileN + cm];
      float s3 = red_base[(3 * (RB * 8) + rm) * kTkTileN + cm];
      float sum = __fadd_rn(__fadd_rn(s0, s1), __fadd_rn(s2, s3));
      p.tk_partials[(static_cast<int64_t>(rm) * kTkSplits + sp) * kIdxDim +
                    nt * kTkTileN + cm] = sum;
    }
  }

  grid.sync();

  // =========================================================================
  // Phase 1: Reduce tk_partials -> FP16 -> token_key_norm -> FP16 keys[r, 128]
  //   - Decode mode: immediately write ring (if non-closing) or pool+RoPE+store!
  //   - Verify mode: write staged[r, 0..128+pos_dims-1] and sync for Phase 2!
  // =========================================================================
  if (bid < p.m) {
    const int r = bid;
    volatile float* const red = smem.pool.red;
    float key_f = 0.f;
    if (tid < kIdxDim) {
      // Fixed-order reduction over 16 splits via __ldcg (Volta Rule 1).
      const float* part_base = p.tk_partials + static_cast<int64_t>(r) * kTkSplits * kIdxDim + tid;
      float s = 0.f;
#pragma unroll
      for (int sp = 0; sp < kTkSplits; ++sp) {
        s = __fadd_rn(s, ld_f32_cg(part_base + sp * kIdxDim));
      }
      // Match ReplicatedLinear FP16 materialization before RMSNorm.
      half h = __float2half(s);
      key_f = __half2float(h);
    }

    // RMSNorm (token_key_norm, dim=128).
    float sq = __fmul_rn(key_f, key_f);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sq = __fadd_rn(sq, __shfl_down_sync(0xffffffffu, sq, off));
    }
    if (lane == 0) red[warp] = sq;
    __syncthreads();
    if (tid == 0) {
      float tot = __fadd_rn(__fadd_rn(red[0], red[1]),
                            __fadd_rn(red[2], red[3]));
      red[8] = rsqrtf(tot / static_cast<float>(kIdxDim) + p.eps);
    }
    __syncthreads();
    const float rrms = red[8];
    half key_h = __float2half(0.f);
    if (tid < kIdxDim) {
      float gam = __half2float(__ldg(p.gamma_tk + tid));
      key_h = __float2half(key_f * rrms * gam);
    }

    if constexpr (IsVerify) {
      const int row_stride = kIdxDim + p.pos_dims;
      float* dst = p.staged + static_cast<int64_t>(r) * row_stride;
      if (tid < kIdxDim) {
        dst[tid] = __half2float(key_h);
      } else if (tid < kIdxDim + p.pos_dims) {
        const int c = tid - kIdxDim;
        const int64_t pos_val = (p.pos_dims == 1)
                                    ? p.positions[r]
                                    : p.positions[static_cast<int64_t>(c) * p.pos_stride_axis + r];
        dst[tid] = static_cast<float>(pos_val);
      }
    } else {
      // Decode mode (width == 1): check slots & seq_lens on device!
      const int64_t slot = load_slot(p, r);
      if (slot >= 0) {
        const int logical = p.seq_lens[r] - 1;
        const int member = logical % kIdxRatio;
        const int group = logical / kIdxRatio;
        const int ring_id = p.ring_table[r * p.ring_table_stride];
        float* ring_base = p.ring + static_cast<int64_t>(ring_id) * p.ring_page_stride;

        if (member != kIdxRatio - 1) {
          // Non-closing row (75% of decode steps): write raw key + pos to ring and finish!
          float* ring_dst = ring_base + static_cast<int64_t>(member) * p.ring_row_stride;
          if (tid < kIdxDim) {
            ring_dst[tid] = __half2float(key_h);
          } else if (tid < kIdxDim + p.pos_dims) {
            const int c = tid - kIdxDim;
            const int64_t pos_val =
                (p.pos_dims == 1)
                    ? p.positions[r]
                    : p.positions[static_cast<int64_t>(c) * p.pos_stride_axis + r];
            ring_dst[tid] = static_cast<float>(pos_val);
          }
        } else if (group % p.dcp_world == p.dcp_rank) {
          // Closing row owned by this DCP rank: gather ring[0..2] + current key[3] -> pool!
          if (tid < kIdxDim) {
            smem.pool.raw[0][tid] = ring_base[0 * p.ring_row_stride + tid];
            smem.pool.raw[1][tid] = ring_base[1 * p.ring_row_stride + tid];
            smem.pool.raw[2][tid] = ring_base[2 * p.ring_row_stride + tid];
            smem.pool.raw[3][tid] = __half2float(key_h);
          } else if (tid < kIdxDim + p.pos_dims) {
            const int c = tid - kIdxDim;
            smem.pool.first_pos[c] =
                static_cast<int64_t>(ring_base[0 * p.ring_row_stride + kIdxDim + c]);
          }
          __syncthreads();
          int64_t fpos[3] = {smem.pool.first_pos[0], 0, 0};
          if (p.pos_dims == 3) {
            fpos[1] = smem.pool.first_pos[1];
            fpos[2] = smem.pool.first_pos[2];
          }
          pool_norm_rotate_store(
              p, smem.pool.raw, fpos, r, group,
              smem.pool.red, smem.pool.pooled_f, smem.pool.pooled_h);
        }
      }
    }
  }

  if constexpr (IsVerify) {
    grid.sync();
    const int batch = p.m / p.width;
    const int total_closes = batch * p.closes;
    if (bid < total_closes) {
      const int req = bid / p.closes;
      const int cl = bid - req * p.closes;
      const int length = p.seq_lens[req];
      const int prefix = length - p.width;
      const int group = prefix / kIdxRatio + cl;
      const int first = group * kIdxRatio;
      if ((group + 1) * kIdxRatio <= length &&
          load_slot(p, req * p.width) >= 0 &&
          (group % p.dcp_world == p.dcp_rank)) {
        const int ring_id = p.ring_table[req * p.ring_table_stride];
        const float* ring_base = p.ring + static_cast<int64_t>(ring_id) * p.ring_page_stride;
        const int staged_stride = kIdxDim + p.pos_dims;
        const float* staged_req = p.staged + static_cast<int64_t>(req * p.width) * staged_stride;

        if (tid < kIdxDim) {
#pragma unroll
          for (int m_idx = 0; m_idx < kIdxRatio; ++m_idx) {
            const int rel = (first - prefix) + m_idx;
            if (rel < 0) {
              smem.pool.raw[m_idx][tid] = ring_base[m_idx * p.ring_row_stride + tid];
            } else {
              // Inter-phase read of staged written in Phase 1 -> MUST use __ldcg (Volta Rule 1).
              smem.pool.raw[m_idx][tid] = ld_f32_cg(staged_req + rel * staged_stride + tid);
            }
          }
        } else if (tid < kIdxDim + p.pos_dims) {
          const int c = tid - kIdxDim;
          const int first_rel = first - prefix;
          float pos_f;
          if (first_rel < 0) {
            pos_f = ring_base[0 * p.ring_row_stride + kIdxDim + c];
          } else {
            pos_f = ld_f32_cg(staged_req + first_rel * staged_stride + kIdxDim + c);
          }
          smem.pool.first_pos[c] = static_cast<int64_t>(pos_f);
        }
        __syncthreads();
        int64_t fpos[3] = {smem.pool.first_pos[0], 0, 0};
          if (p.pos_dims == 3) {
            fpos[1] = smem.pool.first_pos[1];
            fpos[2] = smem.pool.first_pos[2];
          }
        pool_norm_rotate_store(
            p, smem.pool.raw, fpos, req, group,
            smem.pool.red, smem.pool.pooled_f, smem.pool.pooled_h);
      }
    }
  }
}

// =============================================================================
// K3b: Long-Context (>2048) Query Projection + Inline No-FMA RoPE + Head Weight
// =============================================================================

struct IdxQueryPrepParams {
  const half* q_c;         // [M, 1024] FP16
  const half* hidden;      // [M, 3072] FP16
  const uint8_t* q8;
  const uint8_t* hw8;
  const half* w_q;         // [2048, 1024] FP16
  const half* w_hw;        // [16, 3072] FP16
  const float* cos_buf;    // [M, 32] FP32
  const float* sin_buf;    // [M, 32] FP32
  half* q_out;             // [M, 16, 128] FP16
  float* w_out;            // [M, 16] FP32
  int m;
  float hw_scale;          // 16^-0.5 * 128^-0.5
};

template <int RB, bool W8>
__global__ __launch_bounds__(kThreads, 2)
void hy4_idx_query_prep_kernel(IdxQueryPrepParams p) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int bid = blockIdx.x;

  const int wn = warp & 1;

  __shared__ float red[4][RB * 8][16];
  __shared__ half tmp_h[RB * 8][16];
  volatile float* red_v = &red[0][0][0];

  if (bid < 128) {
    // Query projection: 2048 output cols / 16 cols per CTA = 128 CTAs.
    // Note: each head has 128 dims = 8 CTAs (4 nope CTAs [0..63], 4 rope CTAs [64..127]).
    const int n_tile_base = bid * 16;
    const int n_base = n_tile_base + wn * 8;
    const int head = n_tile_base / kIdxDim;
    const int col_in_head_base = n_tile_base - head * kIdxDim;

    if (W8 && p.q8) {
      hy4_w8::slice16<RB, false, 4>(
          p.q_c, p.q8, p.m, kIdxHeads * kIdxDim, kQRank, n_tile_base, 0, kQRank, red_v);
    } else {
      hy4_next::project_tile<RB, 16>(
          p.q_c, p.w_q + static_cast<int64_t>(n_base) * kQRank,
          p.m, kQRank, 0, kQRank, red_v);
    }

    const int total = p.m * 16;
    if (col_in_head_base < kIdxDim - kIdxRope) {
      // Non-rotary half [0..63]: directly round to FP16 and store.
      for (int idx = tid; idx < total; idx += kThreads) {
        const int rm = idx >> 4;
        const int cm = idx & 15;
        float s0 = red_v[(0 * (RB * 8) + rm) * 16 + cm];
        float s1 = red_v[(1 * (RB * 8) + rm) * 16 + cm];
        float s2 = red_v[(2 * (RB * 8) + rm) * 16 + cm];
        float s3 = red_v[(3 * (RB * 8) + rm) * 16 + cm];
        float sum = __fadd_rn(__fadd_rn(s0, s1), __fadd_rn(s2, s3));
        p.q_out[(static_cast<int64_t>(rm) * kIdxHeads + head) * kIdxDim +
                col_in_head_base + cm] = __float2half(sum);
      }
    } else {
      // Rotary half [64..127]: round to FP16 first (matching ReplicatedLinear), then no-FMA RoPE.
      for (int idx = tid; idx < total; idx += kThreads) {
        const int rm = idx >> 4;
        const int cm = idx & 15;
        float s0 = red_v[(0 * (RB * 8) + rm) * 16 + cm];
        float s1 = red_v[(1 * (RB * 8) + rm) * 16 + cm];
        float s2 = red_v[(2 * (RB * 8) + rm) * 16 + cm];
        float s3 = red_v[(3 * (RB * 8) + rm) * 16 + cm];
        float sum = __fadd_rn(__fadd_rn(s0, s1), __fadd_rn(s2, s3));
        tmp_h[rm][cm] = __float2half(sum);
      }
      __syncthreads();
      const int rope_off = col_in_head_base - (kIdxDim - kIdxRope);
      for (int idx = tid; idx < total; idx += kThreads) {
        const int rm = idx >> 4;
        const int cm = idx & 15;
        const int pair = (rope_off + cm) >> 1;
        const int local_pair_base = cm & ~1;
        const float even = __half2float(tmp_h[rm][local_pair_base]);
        const float odd = __half2float(tmp_h[rm][local_pair_base + 1]);
        const float c = __ldg(p.cos_buf + rm * 32 + pair);
        const float s = __ldg(p.sin_buf + rm * 32 + pair);
        float rot;
        if ((cm & 1) == 0) {
          rot = __fsub_rn(__fmul_rn(even, c), __fmul_rn(odd, s));
        } else {
          rot = __fadd_rn(__fmul_rn(odd, c), __fmul_rn(even, s));
        }
        p.q_out[(static_cast<int64_t>(rm) * kIdxHeads + head) * kIdxDim +
                col_in_head_base + cm] = __float2half(rot);
      }
    }
  } else if (bid == 128) {
    // Head weight projection: hidden[M, 3072] x W_hw[16, 3072]^T -> FP16 -> FP32 * hw_scale.
    const int n_base = wn * 8;
    if (W8 && p.hw8) {
      hy4_w8::slice16<RB, false, 4>(
          p.hidden, p.hw8, p.m, kIdxHeads, kHidden, 0, 0, kHidden, red_v);
    } else {
      hy4_next::project_tile<RB, 16>(
          p.hidden, p.w_hw + static_cast<int64_t>(n_base) * kHidden,
          p.m, kHidden, 0, kHidden, red_v);
    }

    const int total = p.m * 16;
    for (int idx = tid; idx < total; idx += kThreads) {
      const int rm = idx >> 4;
      const int cm = idx & 15;
      float s0 = red_v[(0 * (RB * 8) + rm) * 16 + cm];
      float s1 = red_v[(1 * (RB * 8) + rm) * 16 + cm];
      float s2 = red_v[(2 * (RB * 8) + rm) * 16 + cm];
      float s3 = red_v[(3 * (RB * 8) + rm) * 16 + cm];
      float sum = __fadd_rn(__fadd_rn(s0, s1), __fadd_rn(s2, s3));
      // Match ReplicatedLinear FP16 output -> .float() * scale.
      float h_f = __half2float(__float2half(sum));
      p.w_out[rm * 16 + cm] = h_f * p.hw_scale;
    }
  }
}

void idx_pool_update_sm70(
    const at::Tensor& hidden,
    const at::Tensor& w_tk,
    const at::Tensor& gamma_tk,
    const at::Tensor& w_kv,
    const at::Tensor& w_kg,
    const at::Tensor& ape,
    const at::Tensor& gamma_key,
    const at::Tensor& inv_freq,
    const at::Tensor& positions,
    const at::Tensor& seq_lens,
    const at::Tensor& slots,
    const at::Tensor& pool_table,
    const at::Tensor& ring_table,
    at::Tensor& tk_partials,
    const std::optional<at::Tensor>& staged,
    at::Tensor& pool,
    at::Tensor& ring,
    int64_t width,
    int64_t page,
    int64_t pages_per_block,
    int64_t dcp_world,
    int64_t dcp_rank,
    int64_t sec1,
    int64_t sec2,
    double eps) {
  TORCH_CHECK(hidden.is_cuda() && hidden.scalar_type() == at::kHalf);
  TORCH_CHECK(hidden.dim() == 2 && hidden.size(1) == kHidden);
  const int m = static_cast<int>(hidden.size(0));
  TORCH_CHECK(m >= 1 && m <= 24, "idx_pool_update_sm70 supports 1..24 rows, got ", m);
  TORCH_CHECK(width >= 1 && m % width == 0);
  const int pos_dims = (positions.dim() == 2 && positions.size(0) == 3) ? 3 : 1;
  const bool is_verify = (width > 1);
  TORCH_CHECK(width == 1 || (width <= 6 && dcp_world == 1));
  TORCH_CHECK(hidden.is_contiguous() && positions.is_contiguous());
  TORCH_CHECK(positions.numel() == m * pos_dims && seq_lens.numel() >= m / width);
  TORCH_CHECK(slots.is_contiguous() && slots.numel() >= m);
  TORCH_CHECK(pool_table.stride(1) == 1 && ring_table.stride(1) == 1);
  TORCH_CHECK(tk_partials.is_contiguous() &&
              tk_partials.numel() >= m * kTkSplits * kIdxDim);
  TORCH_CHECK(dcp_world >= 1 && dcp_rank >= 0 && dcp_rank < dcp_world);
  TORCH_CHECK(page > 0 && pages_per_block > 0);
  if (is_verify) {
    TORCH_CHECK(staged.has_value() && staged->is_cuda() &&
                staged->scalar_type() == at::kFloat && staged->is_contiguous() &&
                staged->sizes() == at::IntArrayRef({m, kIdxDim + pos_dims}));
  }

  IdxUpdateParams p{};
  p.hidden = reinterpret_cast<const half*>(hidden.const_data_ptr<at::Half>());
  if (w_tk.scalar_type() == at::kByte) {
    TORCH_CHECK(w_tk.is_contiguous() && w_tk.device() == hidden.device() &&
                w_tk.numel() == hy4_w8::matrix_bytes(kIdxDim, kHidden));
    p.tk8 = w_tk.const_data_ptr<uint8_t>();
  } else {
    p.w_tk = reinterpret_cast<const half*>(w_tk.const_data_ptr<at::Half>());
  }
  p.gamma_tk = reinterpret_cast<const half*>(gamma_tk.const_data_ptr<at::Half>());
  p.w_kv = w_kv.const_data_ptr<float>();
  p.w_kg = w_kg.const_data_ptr<float>();
  p.ape = ape.const_data_ptr<float>();
  p.gamma_key = gamma_key.const_data_ptr<float>();
  p.inv_freq = inv_freq.const_data_ptr<float>();
  p.positions = positions.const_data_ptr<int64_t>();
  p.seq_lens = seq_lens.const_data_ptr<int32_t>();
  if (slots.scalar_type() == at::kLong) {
    p.slots = slots.const_data_ptr<int64_t>();
    p.slots_i32 = nullptr;
  } else {
    TORCH_CHECK(slots.scalar_type() == at::kInt);
    p.slots = nullptr;
    p.slots_i32 = slots.const_data_ptr<int32_t>();
  }
  p.pool_table = pool_table.const_data_ptr<int32_t>();
  p.ring_table = ring_table.const_data_ptr<int32_t>();
  p.tk_partials = tk_partials.mutable_data_ptr<float>();
  p.staged = is_verify ? staged->mutable_data_ptr<float>() : nullptr;
  p.pool = reinterpret_cast<half*>(pool.mutable_data_ptr<at::Half>());
  p.ring = ring.mutable_data_ptr<float>();

  p.m = m;
  p.width = static_cast<int>(width);
  p.closes = (static_cast<int>(width) + kIdxRatio - 1) / kIdxRatio;
  p.pos_dims = pos_dims;
  p.pos_stride_axis = pos_dims == 3 ? static_cast<int>(positions.stride(0)) : 0;
  p.sec1 = static_cast<int>(sec1);
  p.sec2 = static_cast<int>(sec2);
  p.pool_table_stride = static_cast<int>(pool_table.stride(0));
  p.ring_table_stride = static_cast<int>(ring_table.stride(0));
  p.pool_page_stride = pool.stride(0);
  p.pool_row_stride = pool.stride(1);
  p.ring_page_stride = ring.stride(0);
  p.ring_row_stride = ring.stride(1);
  p.page = static_cast<int>(page);
  p.pages_per_block = static_cast<int>(pages_per_block);
  p.dcp_world = static_cast<int>(dcp_world);
  p.dcp_rank = static_cast<int>(dcp_rank);
  p.eps = static_cast<float>(eps);

  const c10::cuda::CUDAGuard guard(hidden.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int num_ctas = kTkNTiles * kTkSplits;  // 128 CTAs (<= 160 even at RB=3)
  void* args[] = {&p};

  auto launch = [&](auto fn) {
    int max_blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, fn, kThreads, 0));
    int dev = hidden.get_device();
    int sm_count = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
    TORCH_CHECK(num_ctas <= max_blocks_per_sm * sm_count,
                "Cooperative grid ", num_ctas, " exceeds resident CTAs ",
                max_blocks_per_sm * sm_count);
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(fn), dim3(num_ctas), dim3(kThreads), args, 0, stream));
  };

  const auto dispatch = [&](auto packed) {
    constexpr bool w8 = decltype(packed)::value;
    if (m <= 8) {
      if (is_verify) launch(hy4_idx_pool_update_coop_kernel<1, true, w8>);
      else launch(hy4_idx_pool_update_coop_kernel<1, false, w8>);
    } else if (m <= 16) {
      if (is_verify) launch(hy4_idx_pool_update_coop_kernel<2, true, w8>);
      else launch(hy4_idx_pool_update_coop_kernel<2, false, w8>);
    } else {
      if (is_verify) launch(hy4_idx_pool_update_coop_kernel<3, true, w8>);
      else launch(hy4_idx_pool_update_coop_kernel<3, false, w8>);
    }
  };
  if (p.tk8) {
    dispatch(std::true_type{});
  } else {
    dispatch(std::false_type{});
  }
}

void idx_pool_update_sm70_meta(
    const at::Tensor& hidden,
    const at::Tensor& w_tk,
    const at::Tensor& gamma_tk,
    const at::Tensor& w_kv,
    const at::Tensor& w_kg,
    const at::Tensor& ape,
    const at::Tensor& gamma_key,
    const at::Tensor& inv_freq,
    const at::Tensor& positions,
    const at::Tensor& seq_lens,
    const at::Tensor& slots,
    const at::Tensor& pool_table,
    const at::Tensor& ring_table,
    at::Tensor& tk_partials,
    const std::optional<at::Tensor>& staged,
    at::Tensor& pool,
    at::Tensor& ring,
    int64_t width,
    int64_t page,
    int64_t pages_per_block,
    int64_t dcp_world,
    int64_t dcp_rank,
    int64_t sec1,
    int64_t sec2,
    double eps) {
  TORCH_CHECK(hidden.dim() == 2 && hidden.size(1) == kHidden);
  TORCH_CHECK(hidden.size(0) >= 1 && hidden.size(0) <= 24);
  const int width_k = kHidden + (w_tk.scalar_type() == at::kByte
      ? 2 * hy4_w8::groups(kHidden) : 0);
  TORCH_CHECK(w_tk.sizes() == at::IntArrayRef({kIdxDim, width_k}));
}

void idx_query_prep_sm70(
    const at::Tensor& q_c,
    const at::Tensor& hidden,
    const at::Tensor& w_q,
    const at::Tensor& w_hw,
    const at::Tensor& cos_buf,
    const at::Tensor& sin_buf,
    at::Tensor& q_out,
    at::Tensor& w_out) {
  TORCH_CHECK(q_c.is_cuda() && hidden.is_cuda());
  const int m = static_cast<int>(q_c.size(0));
  TORCH_CHECK(m >= 1 && m <= 24);

  IdxQueryPrepParams p{};
  p.q_c = reinterpret_cast<const half*>(q_c.const_data_ptr<at::Half>());
  p.hidden = reinterpret_cast<const half*>(hidden.const_data_ptr<at::Half>());
  if (w_q.scalar_type() == at::kByte) {
    TORCH_CHECK(w_q.is_contiguous() && w_q.device() == q_c.device() &&
                w_q.numel() == hy4_w8::matrix_bytes(kIdxHeads * kIdxDim, kQRank));
    p.q8 = w_q.const_data_ptr<uint8_t>();
  } else {
    p.w_q = reinterpret_cast<const half*>(w_q.const_data_ptr<at::Half>());
  }
  if (w_hw.scalar_type() == at::kByte) {
    TORCH_CHECK(w_hw.is_contiguous() && w_hw.device() == hidden.device() &&
                w_hw.numel() == hy4_w8::matrix_bytes(kIdxHeads, kHidden));
    p.hw8 = w_hw.const_data_ptr<uint8_t>();
  } else {
    p.w_hw = reinterpret_cast<const half*>(w_hw.const_data_ptr<at::Half>());
  }
  p.cos_buf = cos_buf.const_data_ptr<float>();
  p.sin_buf = sin_buf.const_data_ptr<float>();
  p.q_out = reinterpret_cast<half*>(q_out.mutable_data_ptr<at::Half>());
  p.w_out = w_out.mutable_data_ptr<float>();
  p.m = m;
  p.hw_scale = 1.0f / sqrtf(static_cast<float>(kIdxHeads * kIdxDim));

  const c10::cuda::CUDAGuard guard(q_c.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int blocks = 129;  // 128 query CTAs + 1 head_weight CTA
  const auto launch = [&](auto packed) {
    constexpr bool w8 = decltype(packed)::value;
    if (m <= 8) {
      hy4_idx_query_prep_kernel<1, w8><<<blocks, kThreads, 0, stream>>>(p);
    } else if (m <= 16) {
      hy4_idx_query_prep_kernel<2, w8><<<blocks, kThreads, 0, stream>>>(p);
    } else {
      hy4_idx_query_prep_kernel<3, w8><<<blocks, kThreads, 0, stream>>>(p);
    }
  };
  if (p.q8 || p.hw8) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void idx_query_prep_sm70_meta(
    const at::Tensor& q_c,
    const at::Tensor& hidden,
    const at::Tensor& w_q,
    const at::Tensor& w_hw,
    const at::Tensor& cos_buf,
    const at::Tensor& sin_buf,
    at::Tensor& q_out,
    at::Tensor& w_out) {
  const int64_t m = q_c.size(0);
  TORCH_CHECK(m >= 1 && m <= 24);
  TORCH_CHECK(q_out.sizes() == at::IntArrayRef({m, kIdxHeads, kIdxDim}));
  TORCH_CHECK(w_out.sizes() == at::IntArrayRef({m, kIdxHeads}));
}

}  // namespace

TORCH_LIBRARY(hy4_f7_next, m) {
  m.def(
      "idx_pool_update_sm70("
      "Tensor hidden, Tensor w_tk, Tensor gamma_tk, Tensor w_kv, Tensor w_kg,"
      "Tensor ape, Tensor gamma_key, Tensor inv_freq, Tensor positions,"
      "Tensor seq_lens, Tensor slots, Tensor pool_table, Tensor ring_table,"
      "Tensor(a!) tk_partials, Tensor(b!)? staged, Tensor(c!) pool, Tensor(d!) ring,"
      "int width, int page, int pages_per_block, int dcp_world, int dcp_rank,"
      "int sec1, int sec2, float eps) -> ()");
  m.def(
      "idx_query_prep_sm70("
      "Tensor q_c, Tensor hidden, Tensor w_q, Tensor w_hw,"
      "Tensor cos_buf, Tensor sin_buf, Tensor(a!) q_out, Tensor(b!) w_out) -> ()");
}

TORCH_LIBRARY_IMPL(hy4_f7_next, CUDA, m) {
  m.impl("idx_pool_update_sm70", &idx_pool_update_sm70);
  m.impl("idx_query_prep_sm70", &idx_query_prep_sm70);
}

TORCH_LIBRARY_IMPL(hy4_f7_next, Meta, m) {
  m.impl("idx_pool_update_sm70", &idx_pool_update_sm70_meta);
  m.impl("idx_query_prep_sm70", &idx_query_prep_sm70_meta);
}
