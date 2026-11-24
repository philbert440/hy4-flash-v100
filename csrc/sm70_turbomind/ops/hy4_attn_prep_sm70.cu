// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// F7 Stage 1 (HY4_F7_ATTN_PREP) SM70 attention input-side kernels for Hy4-flash:
//   - K1 (rope_phases_sm70): single-launch FP32 cos/sin phase generation for 1D text positions.
//   - K2 (attn_qkv_prep_sm70): cooperative multi-phase QKV + gate preparation kernel with
//     Stage-1b SWA Phase-1 tail split-K wave fix.
//
// Volta sm_70 + nvcc 12.8 mandatory correctness rules (carried from F7 prototype):
//   Rule 1 (Cross-SM L1 non-coherence): Volta per-SM L1 caches are not coherent across SMs.
//     Every global read of an inter-phase buffer written before grid.sync() MUST bypass L1
//     via L2-coherent __ldcg (ld.global.cg). Read-only weights and pre-launch inputs may use
//     __ldcs / __ldg safely.
//   Rule 2 (nvcc 12.8 __restrict__ dead-load elision across BAR.SYNC): Passing a non-volatile
//     __restrict__ shared-memory reduction pointer to a device helper under nvcc 12.8 -arch=sm_70
//     allows the compiler to elide LDS reloads after __syncthreads(). Every shared-memory
//     reduction pointer passed to helpers MUST be volatile-qualified (volatile float*).

#include "hy4_w8_sm70.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>

namespace cg = cooperative_groups;

namespace {

constexpr int kThreads = 256;  // 8 warps: warps_n = 2, warps_k = 4
constexpr int kWarpsK = 4;
constexpr int kTileN = 16;     // 8 * warps_n
constexpr int kHidden = 3072;
constexpr int kQRank = 1024;
constexpr int kKvRank = 512;
constexpr int kRopeDim = 64;
constexpr int kKvADim = kKvRank + kRopeDim;  // 576
constexpr int kQkNope = 192;
constexpr int kVHead = 256;
constexpr int kSwaTailStart = 320;
constexpr int kSwaTailTiles = 36;  // 356 - 320 (all inside linear_gate)
constexpr int kSwaTailSplits = 4;  // Fixed across RB=1,2,3 for exact row invariance

// =============================================================================
// K1: Single-launch FP32 rotary cos/sin phase kernel (1D text positions)
// =============================================================================

template <typename PosT>
__global__ void hy4_rope_phases_kernel(
    const PosT* __restrict__ positions,
    const float* __restrict__ inv_freq,
    float* __restrict__ cos_out,
    float* __restrict__ sin_out,
    int m,
    int half_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * half_dim;
  if (idx >= total) return;
  const int row = idx / half_dim;
  const int col = idx - row * half_dim;
  const float pos = static_cast<float>(positions[row]);
  const float freq = __ldg(inv_freq + col);
  const float angle = __fmul_rn(pos, freq);
  cos_out[idx] = cosf(angle);
  sin_out[idx] = sinf(angle);
}

void rope_phases_sm70(
    const at::Tensor& positions,
    const at::Tensor& inv_freq,
    at::Tensor& cos_out,
    at::Tensor& sin_out) {
  TORCH_CHECK(positions.is_cuda() && inv_freq.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(positions.dim() == 1 && inv_freq.dim() == 1,
              "rope_phases_sm70 expects 1D positions and 1D inv_freq");
  TORCH_CHECK(positions.is_contiguous() && inv_freq.is_contiguous(),
              "rope_phases_sm70 expects contiguous inputs");
  TORCH_CHECK(inv_freq.scalar_type() == at::kFloat, "inv_freq must be float32");
  const int m = static_cast<int>(positions.size(0));
  const int half_dim = static_cast<int>(inv_freq.size(0));
  if (m == 0 || half_dim == 0) return;

  const c10::cuda::CUDAGuard guard(positions.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int total = m * half_dim;
  const int threads = 128;
  const int blocks = (total + threads - 1) / threads;

  if (positions.scalar_type() == at::kLong) {
    hy4_rope_phases_kernel<int64_t><<<blocks, threads, 0, stream>>>(
        positions.const_data_ptr<int64_t>(),
        inv_freq.const_data_ptr<float>(),
        cos_out.mutable_data_ptr<float>(),
        sin_out.mutable_data_ptr<float>(),
        m,
        half_dim);
  } else if (positions.scalar_type() == at::kInt) {
    hy4_rope_phases_kernel<int32_t><<<blocks, threads, 0, stream>>>(
        positions.const_data_ptr<int32_t>(),
        inv_freq.const_data_ptr<float>(),
        cos_out.mutable_data_ptr<float>(),
        sin_out.mutable_data_ptr<float>(),
        m,
        half_dim);
  } else {
    TORCH_CHECK(false, "positions must be int64 or int32, got ", positions.scalar_type());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rope_phases_sm70_meta(
    const at::Tensor& positions,
    const at::Tensor& inv_freq,
    at::Tensor& cos_out,
    at::Tensor& sin_out) {
  TORCH_CHECK(positions.dim() == 1 && inv_freq.dim() == 1);
  TORCH_CHECK(cos_out.sizes() == at::IntArrayRef({positions.size(0), inv_freq.size(0)}));
  TORCH_CHECK(sin_out.sizes() == at::IntArrayRef({positions.size(0), inv_freq.size(0)}));
}

// =============================================================================
// K2: Cooperative fused attention QKV + gate preparation (with Stage-1b fix)
// =============================================================================

struct AttnPrepParams {
  // Inputs
  const half* x_in;         // [M, 3072] pre-norm (if gamma_in != nullptr) or post-norm
  const half* gamma_in;     // optional [3072] input_layernorm weight (nullptr => skip Phase 0)
  const float* cos_buf;     // [M, 32] FP32 from K1
  const float* sin_buf;     // [M, 32] FP32 from K1
  const half* w_qa;         // [1024, 3072]
  const half* w_kva;        // [576, 3072]
  const half* w_gate;       // [H*256, 3072]
  const half* gamma_qa;     // [1024]
  const half* gamma_kva;    // [512]
  const half* w_qb;         // [H*256, 1024]
  const half* w_uk_t;       // [H, 512, 192] contiguous

  const uint8_t* qa8;
  const uint8_t* kva8;
  const uint8_t* gate8;
  const uint8_t* qb8;
  const uint8_t* uk8;

  // Scratch (reused across layers)
  half* x_attn;             // [M, 3072] (only written if gamma_in != nullptr)
  half* q_a_raw;            // [M, 1024]
  half* kv_a_raw;           // [M, 576]
  half* q_nope;             // [H, M, 192]
  float* tail_partials;     // [36, 4, M, 16] FP32 for Stage-1b SWA tail split-K

  // Outputs
  half* q_out;              // [M, H, 576]
  half* kv_c_out;           // [M, 512]
  half* k_pe_out;           // [M, 64]
  half* q_c_out;            // [M, 1024]
  half* gate_sig_out;       // [M, H*256]

  uint8_t* cache;
  const int64_t* slots;
  const int32_t* lengths;
  const int32_t* table;
  int32_t* indices;
  int32_t* counts;
  int64_t cache_page_stride;
  int64_t cache_row_stride;
  int64_t table_stride;
  int64_t indices_stride;
  int cache_page;
  int cache_tokens;
  int index_width;

  int m;
  int heads;                // 8 (sparse) or 16 (SWA)
  float eps;
};

#define HY4_F7_MMA(C, A0, A1, B0, B1)                               \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};"                                  \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

__device__ __forceinline__ int acc_row(int i, int lane) {
  return (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
}
__device__ __forceinline__ int acc_col(int i, int lane) {
  return (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
}
__device__ __forceinline__ uint4 zero4() { return make_uint4(0, 0, 0, 0); }

__device__ __forceinline__ uint4 ld_w(const half* p) {
  return __ldcs(reinterpret_cast<const uint4*>(p));
}

// sm_70 Rule 1: inter-phase global reads across grid.sync() MUST use __ldcg
// (ld.global.cg) to bypass Volta non-coherent per-SM L1 cache.
template <bool CoherentX>
__device__ __forceinline__ uint4 ld_x(const half* p) {
  if constexpr (CoherentX) {
    return __ldcg(reinterpret_cast<const uint4*>(p));
  } else {
    return __ldg(reinterpret_cast<const uint4*>(p));
  }
}

__device__ __forceinline__ half2 ld_h2_cg(const half2* p) {
  uint32_t u = __ldcg(reinterpret_cast<const uint32_t*>(p));
  return *reinterpret_cast<half2*>(&u);
}

__device__ __forceinline__ half ld_h_cg(const half* p) {
  uint16_t u = __ldcg(reinterpret_cast<const uint16_t*>(p));
  return *reinterpret_cast<half*>(&u);
}

// Deterministic RMSNorm for a single row [DIM] (DIM divisible by 512).
// sm_70 Rule 2: smem_warp MUST be volatile float* to prevent nvcc 12.8
// from eliding LDS reloads after __syncthreads().
template <int DIM, bool CoherentSrc>
__device__ __forceinline__ void rmsnorm_row_256(
    const half* __restrict__ src,
    const half* __restrict__ gamma,
    half* __restrict__ dst,
    float eps,
    volatile float* smem_warp) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const half2* src2 = reinterpret_cast<const half2*>(src);
  const half2* gam2 = reinterpret_cast<const half2*>(gamma);
  half2* dst2 = reinterpret_cast<half2*>(dst);

  float sq = 0.f;
  constexpr int kPairs = DIM / 2;
#pragma unroll
  for (int i = tid; i < kPairs; i += kThreads) {
    half2 v = CoherentSrc ? ld_h2_cg(src2 + i) : __ldg(src2 + i);
    float a = __half2float(v.x);
    float b = __half2float(v.y);
    sq = __fadd_rn(sq, __fmul_rn(a, a));
    sq = __fadd_rn(sq, __fmul_rn(b, b));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    sq = __fadd_rn(sq, __shfl_down_sync(0xffffffffu, sq, off));
  }
  if (lane == 0) smem_warp[warp] = sq;
  __syncthreads();
  if (tid == 0) {
    float tot = smem_warp[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) tot = __fadd_rn(tot, smem_warp[w]);
    smem_warp[8] = rsqrtf(tot / static_cast<float>(DIM) + eps);
  }
  __syncthreads();
  const float rrms = smem_warp[8];
#pragma unroll
  for (int i = tid; i < kPairs; i += kThreads) {
    half2 v = CoherentSrc ? ld_h2_cg(src2 + i) : __ldg(src2 + i);
    half2 g = __ldg(gam2 + i);
    half2 out;
    out.x = __float2half_rn(__half2float(v.x) * rrms * __half2float(g.x));
    out.y = __float2half_rn(__half2float(v.y) * rrms * __half2float(g.y));
    dst2[i] = out;
  }
}

// Compute 16 output columns of X[m, K_stride] @ W[16, K_stride]^T over K slice
// [k_slice_begin, k_slice_end) into shared memory red[kWarpsK][RB*8][16].
// (k_slice_end - k_slice_begin) must be a multiple of 64; k_slice_begin multiple of 16.
// sm_70 Rule 1: CoherentX=true uses __ldcg for inter-phase X written before grid.sync().
// sm_70 Rule 2: red is volatile float* so nvcc 12.8 preserves LDS after __syncthreads().
template <int RB, bool CoherentX>
__device__ __forceinline__ void mma_tile_16_slice(
    const half* __restrict__ X,
    const half* __restrict__ W_tile,
    int m,
    int K_stride,
    int k_slice_begin,
    int k_slice_end,
    volatile float* red) {
  constexpr int kRows = RB * 8;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int wn = warp & 1;
  const int wk = warp >> 1;
  const int qp = (lane >> 2) & 3;
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0);

  const int slice_len = k_slice_end - k_slice_begin;
  const int steps16 = slice_len >> 4;
  const int k_begin = k_slice_begin + (wk * steps16 / kWarpsK) * 16;
  const int k_end = k_slice_begin + ((wk + 1) * steps16 / kWarpsK) * 16;

  const half* wrow = W_tile + static_cast<size_t>(wn * 8 + r8) * K_stride;
  const half* xrow[RB];
  bool row_ok[RB];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    const int r = rb * 8 + r8;
    row_ok[rb] = r < m;
    xrow[rb] = X + static_cast<size_t>(row_ok[rb] ? r : 0) * K_stride;
  }

  float acc[RB][8];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[rb][i] = 0.f;
  }

  const int first_lo = k_begin + qp * 8;
  const int first_hi = first_lo + 32;
  uint4 b_lo = (first_lo < k_end) ? ld_w(wrow + first_lo) : zero4();
  uint4 b_hi = (first_hi < k_end) ? ld_w(wrow + first_hi) : zero4();

#pragma unroll 1
  for (int e = k_begin; e < k_end; e += 64) {
    const int lo_k = e + qp * 8;
    const int hi_k = lo_k + 32;
    uint4 n_lo = zero4();
    uint4 n_hi = zero4();
    if (e + 64 < k_end) {
      const int nlo_k = lo_k + 64;
      const int nhi_k = hi_k + 64;
      n_lo = (nlo_k < k_end) ? ld_w(wrow + nlo_k) : zero4();
      n_hi = (nhi_k < k_end) ? ld_w(wrow + nhi_k) : zero4();
    }
    const bool lo_in = lo_k < k_end;
    const bool hi_in = hi_k < k_end;
#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
      const uint4 a_lo = (row_ok[rb] && lo_in) ? ld_x<CoherentX>(xrow[rb] + lo_k) : zero4();
      const uint4 a_hi = (row_ok[rb] && hi_in) ? ld_x<CoherentX>(xrow[rb] + hi_k) : zero4();
      HY4_F7_MMA(acc[rb], a_lo.x, a_lo.y, b_lo.x, b_lo.y);
      HY4_F7_MMA(acc[rb], a_lo.z, a_lo.w, b_lo.z, b_lo.w);
      HY4_F7_MMA(acc[rb], a_hi.x, a_hi.y, b_hi.x, b_hi.y);
      HY4_F7_MMA(acc[rb], a_hi.z, a_hi.w, b_hi.z, b_hi.w);
    }
    b_lo = n_lo;
    b_hi = n_hi;
  }

  // Deterministic quadpair reduction q0 + q1 + q2 + q3 (100% row-invariant).
  const int base_lane = lane & ~12;
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float v = __shfl_sync(0xffffffffu, acc[rb][i], base_lane);
#pragma unroll
      for (int q = 1; q < 4; ++q) {
        v = __fadd_rn(v, __shfl_sync(0xffffffffu, acc[rb][i], base_lane | (q << 2)));
      }
      acc[rb][i] = v;
    }
  }
  if (qp == 0) {
#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int row = rb * 8 + acc_row(i, lane);
        const int col = wn * 8 + acc_col(i, lane);
        red[(wk * kRows + row) * kTileN + col] = acc[rb][i];
      }
    }
  }
  __syncthreads();
}

// The norm CTA owns this row, including the FP16 rounding before FP8 packing.
__device__ __forceinline__ void pack_cache_row(
    const AttnPrepParams& p, int row, volatile float* reduce) {
  const int64_t slot = p.slots[row];
  if (slot < 0) return;
  uint8_t* dst = p.cache + (slot / p.cache_page) * p.cache_page_stride +
                 (slot % p.cache_page) * p.cache_row_stride;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const half* src = p.kv_c_out + row * kKvRank;
  const float a = __half2float(ld_h_cg(src + 2 * tid));
  const float b = __half2float(ld_h_cg(src + 2 * tid + 1));
  float maximum = fmaxf(fabsf(a), fabsf(b));
#pragma unroll
  for (int step = 16; step > 0; step >>= 1) {
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, step));
  }
  if (lane == 0) reduce[warp] = maximum;
  __syncthreads();
  const int group = warp >> 1;
  const float scale = fmaxf(__fdiv_rn(fmaxf(reduce[group * 2], reduce[group * 2 + 1]),
                                      448.f), 0x1p-126f);
  if ((tid & 63) == 0) reinterpret_cast<float*>(dst + kKvRank)[group] = scale;
  dst[2 * tid] = __nv_cvt_float_to_fp8(__fdiv_rn(a, scale), __NV_SATFINITE, __NV_E4M3);
  dst[2 * tid + 1] = __nv_cvt_float_to_fp8(__fdiv_rn(b, scale), __NV_SATFINITE, __NV_E4M3);
  if (tid < kRopeDim) {
    reinterpret_cast<half*>(dst + kKvRank + 16)[tid] =
        ld_h_cg(p.k_pe_out + row * kRopeDim + tid);
  }
}

// Slot translation for a sliding window or an entire bounded sparse prefix.
// Four CTAs per row leave the projection work unchanged.
template <int Threads = kThreads>
__device__ __forceinline__ void write_decode_indices(const AttnPrepParams& p) {
  constexpr int kBlocksPerRow = 4;
  const int row = blockIdx.x / kBlocksPerRow;
  if (row >= p.cache_tokens) return;
  const int shard = blockIdx.x % kBlocksPerRow;
  const int length = p.lengths[row];
  const bool active = p.slots[row] >= 0;
  const int count = active ? min(length, p.index_width) : 0;
  const int first = max(length - p.index_width, 0);
  if (shard == 0 && threadIdx.x == 0) p.counts[row] = count;
  for (int col = threadIdx.x + shard * Threads; col < p.index_width;
       col += kBlocksPerRow * Threads) {
    int physical = -1;
    if (col < count) {
      const int logical = first + col;
      const int block = p.table[row * p.table_stride + logical / p.cache_page];
      physical = block * p.cache_page + logical % p.cache_page;
    }
    p.indices[row * p.indices_stride + col] = physical;
  }
}

template <int RB, bool FuseInputNorm, bool W8>
__global__ void __launch_bounds__(kThreads, (RB == 1 ? 4 : 2))
    hy4_attn_qkv_prep_coop_kernel(const AttnPrepParams p) {
  constexpr int kRows = RB * 8;
  // sm_70 Rule 2: volatile shared memory buffer for cross-warp reductions.
  __shared__ volatile float red[kWarpsK * kRows * kTileN];
  cg::grid_group grid = cg::this_grid();
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int gdim = gridDim.x;
  const int m = p.m;
  const int H = p.heads;

  // =========================================================================
  // PHASE 0 (optional): input_layernorm (RMSNorm 3072)
  // When FuseInputNorm=false, p.x_in is already post-input_layernorm and we
  // skip Phase 0 + its grid.sync(), reading p.x_in through L1 (__ldg) in Phase 1.
  // =========================================================================
  const half* x_phase1 = p.x_in;
  if constexpr (FuseInputNorm) {
    if (bid < m) {
      rmsnorm_row_256<kHidden, false>(
          p.x_in + static_cast<size_t>(bid) * kHidden,
          p.gamma_in,
          p.x_attn + static_cast<size_t>(bid) * kHidden,
          p.eps,
          red);
    }
    x_phase1 = p.x_attn;
    grid.sync();
  }

  // =========================================================================
  // PHASE 1: Paired input projections on x_phase1[M, 3072]
  //   tiles [0, 64):          q_a_proj (1024 cols) -> q_a_raw
  //   tiles [64, 100):        kv_a_proj_with_mqa (576 cols) -> kv_a_raw
  //   tiles [100, 100+H*16):  linear_gate (H*256 cols) -> sigmoid -> gate_sig_out
  //
  // Stage-1b SWA wave-quantization fix:
  //   On SWA (H=16, 356 tiles), tiles [320, 356) (36 tiles, all in linear_gate)
  //   formerly ran as an 11%-utilized tail wave on 320 CTAs (or 22% on 160 CTAs).
  //   We split K=3072 for those 36 tail tiles into kSwaTailSplits=4 slices of 768
  //   across 36*4=144 CTAs (fixed split count across RB=1,2,3 -> exact 0-bit row
  //   invariance across M=1..24), write FP32 partials to p.tail_partials, and
  //   reduce + sigmoid them on otherwise-idle CTAs during Phase 2 with ZERO extra
  //   grid.sync() barriers.
  // =========================================================================
  const int gate_tiles = H * 16;
  const int phase1_tiles = 100 + gate_tiles;  // 356 (SWA) or 228 (Sparse)
  const int gate_dim = H * kVHead;
  const bool use_swa_tail_splitk =
      (H == 16) && (gdim >= kSwaTailTiles * kSwaTailSplits) &&
      (2 * m + kSwaTailTiles <= gdim);
  const int main_phase1_tiles = use_swa_tail_splitk ? kSwaTailStart : phase1_tiles;

  for (int tile = bid; tile < main_phase1_tiles; tile += gdim) {
    const half* w;
    const uint8_t* w8;
    int n;
    int seg_kind;
    int col_base;
    if (tile < 64) {
      seg_kind = 0;
      col_base = tile * kTileN;
      w = p.w_qa;
      w8 = p.qa8;
      n = kQRank;
    } else if (tile < 100) {
      seg_kind = 1;
      col_base = (tile - 64) * kTileN;
      w = p.w_kva;
      w8 = p.kva8;
      n = kKvADim;
    } else {
      seg_kind = 2;
      col_base = (tile - 100) * kTileN;
      w = p.w_gate;
      w8 = p.gate8;
      n = gate_dim;
    }

    if (W8 && w8) {
      hy4_w8::slice16<RB, FuseInputNorm, kWarpsK>(
          x_phase1, w8, m, n, kHidden, col_base, 0, kHidden, red);
    } else {
      mma_tile_16_slice<RB, FuseInputNorm>(
          x_phase1, w + static_cast<size_t>(col_base) * kHidden, m, kHidden, 0, kHidden, red);
    }

    for (int idx = tid; idx < m * kTileN; idx += kThreads) {
      const int row = idx >> 4;
      const int c = idx & 15;
      float v = red[row * kTileN + c];
#pragma unroll
      for (int q = 1; q < kWarpsK; ++q) {
        v = __fadd_rn(v, red[(q * kRows + row) * kTileN + c]);
      }
      const half h = __float2half_rn(v);
      if (seg_kind == 0) {
        p.q_a_raw[row * kQRank + col_base + c] = h;
      } else if (seg_kind == 1) {
        p.kv_a_raw[row * kKvADim + col_base + c] = h;
      } else {
        // Match FP16 materialization of linear_gate before sigmoid.
        const float gf = __half2float(h);
        const float sig = 1.0f / (1.0f + expf(-gf));
        p.gate_sig_out[row * gate_dim + col_base + c] = __float2half_rn(sig);
      }
    }
    __syncthreads();
  }

  if (use_swa_tail_splitk && bid < kSwaTailTiles * kSwaTailSplits) {
    const int t_idx = bid / kSwaTailSplits;  // 0..35
    const int s_idx = bid - t_idx * kSwaTailSplits;  // 0..3
    const int tile = kSwaTailStart + t_idx;  // 320..355 (all linear_gate)
    const int col_base = (tile - 100) * kTileN;
    constexpr int kSlice = kHidden / kSwaTailSplits;  // 768
    const int k_lo = s_idx * kSlice;
    const int k_hi = k_lo + kSlice;

    if (W8 && p.gate8) {
      hy4_w8::slice16<RB, FuseInputNorm, kWarpsK>(
          x_phase1, p.gate8, m, gate_dim, kHidden, col_base, k_lo, k_hi, red);
    } else {
      const half* w_ptr = p.w_gate + static_cast<size_t>(col_base) * kHidden;
      mma_tile_16_slice<RB, FuseInputNorm>(
          x_phase1, w_ptr, m, kHidden, k_lo, k_hi, red);
    }

    float* dst = p.tail_partials +
                 static_cast<size_t>(t_idx * kSwaTailSplits + s_idx) * m * kTileN;
    for (int idx = tid; idx < m * kTileN; idx += kThreads) {
      const int row = idx >> 4;
      const int c = idx & 15;
      float v = red[row * kTileN + c];
#pragma unroll
      for (int q = 1; q < kWarpsK; ++q) {
        v = __fadd_rn(v, red[(q * kRows + row) * kTileN + c]);
      }
      dst[row * kTileN + c] = v;
    }
    __syncthreads();
  }
  grid.sync();

  // =========================================================================
  // PHASE 2: q_a_layernorm (1024), kv_a_layernorm (512), k_pe RoPE (64),
  //          and Stage-1b deterministic tail reduction on idle CTAs.
  // =========================================================================
  if (bid < m) {
    rmsnorm_row_256<kQRank, true>(
        p.q_a_raw + static_cast<size_t>(bid) * kQRank,
        p.gamma_qa,
        p.q_c_out + static_cast<size_t>(bid) * kQRank,
        p.eps,
        red);
  } else if (bid < 2 * m) {
    const int row = bid - m;
    const half* kv_row = p.kv_a_raw + static_cast<size_t>(row) * kKvADim;
    rmsnorm_row_256<kKvRank, true>(
        kv_row,
        p.gamma_kva,
        p.kv_c_out + static_cast<size_t>(row) * kKvRank,
        p.eps,
        red);
    if (tid < 32) {
      const half* kpe_in = kv_row + kKvRank;
      // sm_70 Rule 1: kv_a_raw was written in Phase 1 -> __ldcg.
      const float even = __half2float(ld_h_cg(kpe_in + 2 * tid));
      const float odd = __half2float(ld_h_cg(kpe_in + 2 * tid + 1));
      // cos_buf / sin_buf were written by K1 before K2 launched -> __ldg safe.
      const float c = __ldg(p.cos_buf + row * 32 + tid);
      const float s = __ldg(p.sin_buf + row * 32 + tid);
      const float rot_e = __fsub_rn(__fmul_rn(even, c), __fmul_rn(odd, s));
      const float rot_o = __fadd_rn(__fmul_rn(odd, c), __fmul_rn(even, s));
      p.k_pe_out[row * kRopeDim + 2 * tid] = __float2half_rn(rot_e);
      p.k_pe_out[row * kRopeDim + 2 * tid + 1] = __float2half_rn(rot_o);
    }
    if (row < p.cache_tokens) {
      __syncthreads();
      pack_cache_row(p, row, red);
    }
  } else if (use_swa_tail_splitk && bid < 2 * m + kSwaTailTiles) {
    const int t_idx = bid - 2 * m;  // 0..35
    const int tile = kSwaTailStart + t_idx;
    const int col_base = (tile - 100) * kTileN;
    const float* base_p = p.tail_partials +
                          static_cast<size_t>(t_idx * kSwaTailSplits) * m * kTileN;
    for (int idx = tid; idx < m * kTileN; idx += kThreads) {
      const int row = idx >> 4;
      const int c = idx & 15;
      // sm_70 Rule 1: tail_partials was written in Phase 1 -> __ldcg.
      float v = __ldcg(base_p + row * kTileN + c);
#pragma unroll
      for (int s = 1; s < kSwaTailSplits; ++s) {
        v = __fadd_rn(v, __ldcg(base_p + static_cast<size_t>(s) * m * kTileN +
                                row * kTileN + c));
      }
      const float gf = __half2float(__float2half_rn(v));
      const float sig = 1.0f / (1.0f + expf(-gf));
      p.gate_sig_out[row * gate_dim + col_base + c] = __float2half_rn(sig);
    }
  }
  grid.sync();

  // =========================================================================
  // PHASE 3: q_b_proj (H*256 x 1024)
  //   d0 < 192: write head-major q_nope[head, row, d]
  //   d0 >= 192: exact no-FMA RoPE in registers -> Q[row, head, 512 + d - 192]
  // =========================================================================
  const int phase3_tiles = H * 16;  // 256 (SWA) or 128 (sparse)
  for (int tile = bid; tile < phase3_tiles; tile += gdim) {
    const int head = tile >> 4;
    const int d0 = (tile & 15) * kTileN;
    // q_c_out was written in Phase 2 and requires coherent loads.
    if (W8 && p.qb8) {
      hy4_w8::slice16<RB, true, kWarpsK>(
          p.q_c_out, p.qb8, m, H * (kQkNope + kRopeDim), kQRank,
          tile * kTileN, 0, kQRank, red);
    } else {
      const half* w_ptr = p.w_qb + static_cast<size_t>(tile * kTileN) * kQRank;
      mma_tile_16_slice<RB, true>(p.q_c_out, w_ptr, m, kQRank, 0, kQRank, red);
    }

    if (d0 < kQkNope) {
      for (int idx = tid; idx < m * kTileN; idx += kThreads) {
        const int row = idx >> 4;
        const int c = idx & 15;
        float v = red[row * kTileN + c];
#pragma unroll
        for (int q = 1; q < kWarpsK; ++q) {
          v = __fadd_rn(v, red[(q * kRows + row) * kTileN + c]);
        }
        p.q_nope[(static_cast<size_t>(head) * m + row) * kQkNope + d0 + c] =
            __float2half_rn(v);
      }
    } else {
      const int pe_off = d0 - kQkNope;  // 0, 16, 32, 48
      const int pair_base = pe_off >> 1;
      for (int idx = tid; idx < m * 8; idx += kThreads) {
        const int row = idx >> 3;
        const int pl = idx & 7;
        const int ce = pl * 2;
        const int co = ce + 1;
        float ve = red[row * kTileN + ce];
        float vo = red[row * kTileN + co];
#pragma unroll
        for (int q = 1; q < kWarpsK; ++q) {
          ve = __fadd_rn(ve, red[(q * kRows + row) * kTileN + ce]);
          vo = __fadd_rn(vo, red[(q * kRows + row) * kTileN + co]);
        }
        // Preserve FP16 materialization of q_b_proj before FP32 rotary.
        const float even = __half2float(__float2half_rn(ve));
        const float odd = __half2float(__float2half_rn(vo));
        const float c = __ldg(p.cos_buf + row * 32 + pair_base + pl);
        const float s = __ldg(p.sin_buf + row * 32 + pair_base + pl);
        const float rot_e = __fsub_rn(__fmul_rn(even, c), __fmul_rn(odd, s));
        const float rot_o = __fadd_rn(__fmul_rn(odd, c), __fmul_rn(even, s));
        half* q_dst =
            p.q_out + (static_cast<size_t>(row) * H + head) * kKvADim + kKvRank + pe_off;
        q_dst[ce] = __float2half_rn(rot_e);
        q_dst[co] = __float2half_rn(rot_o);
      }
    }
    __syncthreads();
  }
  grid.sync();

  // =========================================================================
  // PHASE 4: Absorbed w_uk BMM (H x 512 x 192) -> Q[row, head, 0:512]
  // =========================================================================
  const int phase4_tiles = H * 32;  // 512 (SWA) or 256 (sparse)
  for (int tile = bid; tile < phase4_tiles; tile += gdim) {
    const int head = tile >> 5;
    const int c0 = (tile & 31) * kTileN;
    const half* x_head = p.q_nope + static_cast<size_t>(head) * m * kQkNope;
    // q_nope was written in Phase 3 and requires coherent loads.
    if (W8 && p.uk8) {
      const auto* w = p.uk8 + head * hy4_w8::matrix_bytes(kKvRank, kQkNope);
      hy4_w8::slice16<RB, true, kWarpsK>(
          x_head, w, m, kKvRank, kQkNope, c0, 0, kQkNope, red);
    } else {
      const half* w_ptr = p.w_uk_t + (static_cast<size_t>(head) * kKvRank + c0) * kQkNope;
      mma_tile_16_slice<RB, true>(x_head, w_ptr, m, kQkNope, 0, kQkNope, red);
    }

    for (int idx = tid; idx < m * kTileN; idx += kThreads) {
      const int row = idx >> 4;
      const int c = idx & 15;
      float v = red[row * kTileN + c];
#pragma unroll
      for (int q = 1; q < kWarpsK; ++q) {
        v = __fadd_rn(v, red[(q * kRows + row) * kTileN + c]);
      }
      p.q_out[(static_cast<size_t>(row) * H + head) * kKvADim + c0 + c] =
          __float2half_rn(v);
    }
    __syncthreads();
  }
  if (p.indices) write_decode_indices(p);
}

// Separate graph nodes let Q-B and W_UK reuse normalized activations through
// the read-only cache. Eight-column tiles retain all four K warp groups.
constexpr int kNarrowThreads = 128;
constexpr int kNarrowTile = 8;
constexpr int kNarrowRows = 8;
constexpr int kInputQTiles = kQRank / kNarrowTile;
constexpr int kInputKVTiles = kKvADim / kNarrowTile;
constexpr int kGateFirst = kInputQTiles + kInputKVTiles;
constexpr int kNarrowTailFirst = kSwaTailStart * kTileN / kNarrowTile;
constexpr int kNarrowTailTiles = kSwaTailTiles * kTileN / kNarrowTile;
constexpr int kQBTilesPerHead = (kQkNope + kRopeDim) / kNarrowTile;
constexpr int kUKTilesPerHead = kKvRank / kNarrowTile;

// Match the K split and quadpair reduction of slice16 with one N tile.
__device__ __forceinline__ void project_narrow_tile(
    const half* x, const uint8_t* weights, int rows, int cols, int k, int col,
    int k_begin, int k_end, volatile float* red) {
  __syncwarp();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int quad = (lane >> 2) & 3;
  const int split = warp * 4 + quad;
  constexpr int kSplits = 4 * kWarpsK;
  const int chunks = (k_end - k_begin) / 16;
  const int begin = k_begin / 16 + split * chunks / kSplits;
  const int end = k_begin / 16 + (split + 1) * chunks / kSplits;
  const int steps = (chunks + kSplits - 1) / kSplits;
  float accum[1][8] = {};
  hy4_w8::warp_project<1, false>(
      x, weights, rows, cols, k, col / kNarrowTile, begin, end, steps, accum);

  // Masked activation loads can diverge between Volta lanes.
  __syncwarp();
  const int base_lane = lane & ~12;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float sum = __shfl_sync(0xffffffffu, accum[0][i], base_lane);
#pragma unroll
    for (int q = 1; q < 4; ++q) {
      sum = __fadd_rn(sum, __shfl_sync(0xffffffffu, accum[0][i], base_lane | (q << 2)));
    }
    if (quad == 0) {
      const int row = hy4_w8::acc_row(i, lane);
      const int slot = hy4_w8::acc_col(i, lane);
      red[(warp * kNarrowRows + row) * kNarrowTile + slot] = sum;
    }
  }
  __syncthreads();
}

__device__ __forceinline__ float reduce_narrow_tile(
    volatile float* red, int row, int col) {
  float total = red[row * kNarrowTile + col];
#pragma unroll
  for (int warp = 1; warp < kWarpsK; ++warp) {
    total = __fadd_rn(total, red[(warp * kNarrowRows + row) * kNarrowTile + col]);
  }
  return total;
}

__global__ void __launch_bounds__(kNarrowThreads, 8)
hy4_prep_inputs_narrow(const AttnPrepParams p) {
  __syncwarp();
  __shared__ volatile float red[kWarpsK * kNarrowRows * kNarrowTile];
  const int tile = blockIdx.x;
  const int gate_dim = p.heads * kVHead;
  const bool tail = p.heads == 16 && tile >= kNarrowTailFirst;
  const bool gate = tile >= kGateFirst;
  const uint8_t* weights;
  half* dest = nullptr;
  float* partial = nullptr;
  int cols;
  int start_col;
  int k_begin = 0;
  int k_end = kHidden;
  if (tail) {
    // Keep the stock four-way gate split and its 16-column scratch layout.
    const int tail_tile = (tile - kNarrowTailFirst) / kSwaTailSplits;
    const int split = (tile - kNarrowTailFirst) % kSwaTailSplits;
    const int wide_tail = tail_tile / 2;
    const int half_tile_offset = (tail_tile % 2) * kNarrowTile;
    constexpr int kSlice = kHidden / kSwaTailSplits;
    k_begin = split * kSlice;
    k_end = k_begin + kSlice;
    weights = p.gate8;
    cols = gate_dim;
    start_col = (kNarrowTailFirst + tail_tile - kGateFirst) * kNarrowTile;
    partial = p.tail_partials +
        static_cast<size_t>(wide_tail * kSwaTailSplits + split) * p.m * kTileN +
        half_tile_offset;
  } else if (tile < kInputQTiles) {
    weights = p.qa8;
    dest = p.q_a_raw;
    cols = kQRank;
    start_col = tile * kNarrowTile;
  } else if (!gate) {
    weights = p.kva8;
    dest = p.kv_a_raw;
    cols = kKvADim;
    start_col = (tile - kInputQTiles) * kNarrowTile;
  } else {
    weights = p.gate8;
    dest = p.gate_sig_out;
    cols = gate_dim;
    start_col = (tile - kGateFirst) * kNarrowTile;
  }
  project_narrow_tile(
      p.x_in, weights, p.m, cols, kHidden, start_col, k_begin, k_end, red);
  const int row = threadIdx.x / kNarrowTile;
  const int col = threadIdx.x % kNarrowTile;
  if (row < p.m) {
    const float accum = reduce_narrow_tile(red, row, col);
    if (tail) {
      partial[row * kTileN + col] = accum;
    } else {
      half value = __float2half_rn(accum);
      if (gate) {
        const float x = __half2float(value);
        value = __float2half_rn(1.0f / (1.0f + expf(-x)));
      }
      dest[row * cols + start_col + col] = value;
    }
  }
}

__global__ void __launch_bounds__(kThreads, 4)
hy4_prep_norms(const AttnPrepParams p) {
  __syncwarp();
  __shared__ volatile float red[kWarpsK * kNarrowRows * kTileN];
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int m = p.m;
  const int gate_dim = p.heads * kVHead;
  const bool use_swa_tail_splitk = p.heads == 16;
  if (bid < m) {
    rmsnorm_row_256<kQRank, true>(
        p.q_a_raw + static_cast<size_t>(bid) * kQRank,
        p.gamma_qa,
        p.q_c_out + static_cast<size_t>(bid) * kQRank,
        p.eps,
        red);
  } else if (bid < 2 * m) {
    const int row = bid - m;
    const half* kv_row = p.kv_a_raw + static_cast<size_t>(row) * kKvADim;
    rmsnorm_row_256<kKvRank, true>(
        kv_row,
        p.gamma_kva,
        p.kv_c_out + static_cast<size_t>(row) * kKvRank,
        p.eps,
        red);
    if (tid < 32) {
      const half* kpe_in = kv_row + kKvRank;
      // sm_70 Rule 1: kv_a_raw was written in Phase 1 -> __ldcg.
      const float even = __half2float(ld_h_cg(kpe_in + 2 * tid));
      const float odd = __half2float(ld_h_cg(kpe_in + 2 * tid + 1));
      // cos_buf / sin_buf were written by K1 before K2 launched -> __ldg safe.
      const float c = __ldg(p.cos_buf + row * 32 + tid);
      const float s = __ldg(p.sin_buf + row * 32 + tid);
      const float rot_e = __fsub_rn(__fmul_rn(even, c), __fmul_rn(odd, s));
      const float rot_o = __fadd_rn(__fmul_rn(odd, c), __fmul_rn(even, s));
      p.k_pe_out[row * kRopeDim + 2 * tid] = __float2half_rn(rot_e);
      p.k_pe_out[row * kRopeDim + 2 * tid + 1] = __float2half_rn(rot_o);
    }
    if (row < p.cache_tokens) {
      __syncthreads();
      pack_cache_row(p, row, red);
    }
  } else if (use_swa_tail_splitk && bid < 2 * m + kSwaTailTiles) {
    const int t_idx = bid - 2 * m;  // 0..35
    const int tile = kSwaTailStart + t_idx;
    const int col_base = (tile - 100) * kTileN;
    const float* base_p = p.tail_partials +
                          static_cast<size_t>(t_idx * kSwaTailSplits) * m * kTileN;
    for (int idx = tid; idx < m * kTileN; idx += kThreads) {
      const int row = idx >> 4;
      const int c = idx & 15;
      // sm_70 Rule 1: tail_partials was written in Phase 1 -> __ldcg.
      float v = __ldcg(base_p + row * kTileN + c);
#pragma unroll
      for (int s = 1; s < kSwaTailSplits; ++s) {
        v = __fadd_rn(v, __ldcg(base_p + static_cast<size_t>(s) * m * kTileN +
                                row * kTileN + c));
      }
      const float gf = __half2float(__float2half_rn(v));
      const float sig = 1.0f / (1.0f + expf(-gf));
      p.gate_sig_out[row * gate_dim + col_base + c] = __float2half_rn(sig);
    }
  }
}

__global__ void __launch_bounds__(kNarrowThreads, 8)
hy4_prep_qb_narrow(const AttnPrepParams p) {
  __syncwarp();
  __shared__ volatile float red[kWarpsK * kNarrowRows * kNarrowTile];
  const int tile = blockIdx.x;
  const int head = tile / kQBTilesPerHead;
  const int d = (tile % kQBTilesPerHead) * kNarrowTile;
  project_narrow_tile(
      p.q_c_out, p.qb8, p.m, p.heads * (kQkNope + kRopeDim), kQRank,
      tile * kNarrowTile, 0, kQRank, red);
  if (d < kQkNope) {
    const int row = threadIdx.x / kNarrowTile;
    const int col = threadIdx.x % kNarrowTile;
    if (row < p.m) {
      p.q_nope[(static_cast<size_t>(head) * p.m + row) * kQkNope + d + col] =
          __float2half_rn(reduce_narrow_tile(red, row, col));
    }
  } else {
    constexpr int kPairs = kNarrowTile / 2;
    const int row = threadIdx.x / kPairs;
    const int pair = threadIdx.x % kPairs;
    if (row < p.m) {
      const int even_col = pair * 2;
      const int odd_col = even_col + 1;
      const int offset = d - kQkNope;
      const float even = __half2float(__float2half_rn(reduce_narrow_tile(red, row, even_col)));
      const float odd = __half2float(__float2half_rn(reduce_narrow_tile(red, row, odd_col)));
      const int phase = row * (kRopeDim / 2) + offset / 2 + pair;
      const float cosine = __ldg(p.cos_buf + phase);
      const float sine = __ldg(p.sin_buf + phase);
      const float rotated_even = __fsub_rn(__fmul_rn(even, cosine), __fmul_rn(odd, sine));
      const float rotated_odd = __fadd_rn(__fmul_rn(odd, cosine), __fmul_rn(even, sine));
      half* dest = p.q_out + (static_cast<size_t>(row) * p.heads + head) * kKvADim +
          kKvRank + offset;
      dest[even_col] = __float2half_rn(rotated_even);
      dest[odd_col] = __float2half_rn(rotated_odd);
    }
  }
}

__global__ void __launch_bounds__(kNarrowThreads, 8)
hy4_prep_uk_narrow(const AttnPrepParams p) {
  __syncwarp();
  __shared__ volatile float red[kWarpsK * kNarrowRows * kNarrowTile];
  const int tile = blockIdx.x;
  const int head = tile / kUKTilesPerHead;
  const int first_col = (tile % kUKTilesPerHead) * kNarrowTile;
  const half* x = p.q_nope + static_cast<size_t>(head) * p.m * kQkNope;
  const uint8_t* weights = p.uk8 + head * hy4_w8::matrix_bytes(kKvRank, kQkNope);
  project_narrow_tile(
      x, weights, p.m, kKvRank, kQkNope, first_col, 0, kQkNope, red);

  const int row = threadIdx.x / kNarrowTile;
  const int col = threadIdx.x % kNarrowTile;
  if (row < p.m) {
    p.q_out[(static_cast<size_t>(row) * p.heads + head) * kKvADim + first_col + col] =
        __float2half_rn(reduce_narrow_tile(red, row, col));
  }
  if (p.indices) write_decode_indices<kNarrowThreads>(p);
}

void launch_split_w8_prep(AttnPrepParams p, cudaStream_t stream) {
  const bool swa = p.heads == 16;
  const int input_blocks = swa ? kNarrowTailFirst + kNarrowTailTiles * kSwaTailSplits
                               : kGateFirst + p.heads * kVHead / kNarrowTile;
  const int norm_blocks = 2 * p.m + (swa ? kSwaTailTiles : 0);
  hy4_prep_inputs_narrow<<<input_blocks, kNarrowThreads, 0, stream>>>(p);
  hy4_prep_norms<<<norm_blocks, kThreads, 0, stream>>>(p);
  hy4_prep_qb_narrow<<<p.heads * kQBTilesPerHead, kNarrowThreads, 0, stream>>>(p);
  hy4_prep_uk_narrow<<<p.heads * kUKTilesPerHead, kNarrowThreads, 0, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int RB, bool FuseInputNorm, bool W8>
void launch_coop_prep(AttnPrepParams p, cudaStream_t stream) {
  static int cached_blocks_by_dev[16] = {0};
  int dev = 0;
  cudaGetDevice(&dev);
  int blocks = (dev >= 0 && dev < 16) ? cached_blocks_by_dev[dev] : 0;
  if (blocks == 0) {
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    int max_bpsm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_bpsm,
        (const void*)hy4_attn_qkv_prep_coop_kernel<RB, FuseInputNorm, W8>,
        kThreads,
        0);
    TORCH_CHECK(err == cudaSuccess && max_bpsm > 0,
                "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed: ",
                cudaGetErrorString(err));
    blocks = max_bpsm * sm_count;
    if (dev >= 0 && dev < 16) cached_blocks_by_dev[dev] = blocks;
  }
  void* args[] = {&p};
  cudaError_t err = cudaLaunchCooperativeKernel(
      (const void*)hy4_attn_qkv_prep_coop_kernel<RB, FuseInputNorm, W8>,
      dim3(blocks),
      dim3(kThreads),
      args,
      0,
      stream);
  TORCH_CHECK(err == cudaSuccess,
              "cudaLaunchCooperativeKernel(K2) failed (blocks=", blocks, "): ",
              cudaGetErrorString(err));
}

void attn_qkv_prep_sm70(
    const at::Tensor& x_in,
    const std::optional<at::Tensor>& gamma_in,
    const at::Tensor& cos_buf,
    const at::Tensor& sin_buf,
    const at::Tensor& w_qa,
    const at::Tensor& w_kva,
    const at::Tensor& w_gate,
    const at::Tensor& gamma_qa,
    const at::Tensor& gamma_kva,
    const at::Tensor& w_qb,
    const at::Tensor& w_uk_t,
    const std::optional<at::Tensor>& x_attn,
    at::Tensor& q_a_raw,
    at::Tensor& kv_a_raw,
    at::Tensor& q_nope,
    at::Tensor& tail_partials,
    at::Tensor& q_out,
    at::Tensor& kv_c_out,
    at::Tensor& k_pe_out,
    at::Tensor& q_c_out,
    at::Tensor& gate_sig_out,
    double eps,
    const std::optional<at::Tensor>& cache,
    const std::optional<at::Tensor>& slots,
    const std::optional<at::Tensor>& lengths,
    const std::optional<at::Tensor>& table,
    const std::optional<at::Tensor>& indices,
    const std::optional<at::Tensor>& counts,
    std::optional<bool> split_w8) {
  const int m = static_cast<int>(x_in.size(0));
  const int heads = static_cast<int>(w_uk_t.size(0));
  TORCH_CHECK(x_in.is_cuda(), "x_in must be CUDA");
  TORCH_CHECK(m >= 1 && m <= 24, "m must be in [1, 24], got ", m);
  TORCH_CHECK(heads == 8 || heads == 16, "heads must be 8 or 16, got ", heads);
  TORCH_CHECK(x_in.size(1) == kHidden, "x_in hidden must be 3072");

  const c10::cuda::CUDAGuard guard(x_in.device());
  const bool fuse_in_norm = gamma_in.has_value();
  if (fuse_in_norm) {
    TORCH_CHECK(x_attn.has_value(), "x_attn scratch required when gamma_in is provided");
  }

  AttnPrepParams p{};
  p.x_in = reinterpret_cast<const half*>(x_in.const_data_ptr<at::Half>());
  p.gamma_in = fuse_in_norm
                   ? reinterpret_cast<const half*>(gamma_in->const_data_ptr<at::Half>())
                   : nullptr;
  p.cos_buf = cos_buf.const_data_ptr<float>();
  p.sin_buf = sin_buf.const_data_ptr<float>();
  if (w_qa.scalar_type() == at::kByte) {
    TORCH_CHECK(w_qa.is_contiguous() &&
                w_qa.device() == x_in.device() &&
                w_qa.numel() == 1 * hy4_w8::matrix_bytes(kQRank, kHidden));
    p.qa8 = w_qa.const_data_ptr<uint8_t>();
  } else {
    p.w_qa = reinterpret_cast<const half*>(w_qa.const_data_ptr<at::Half>());
  }
  if (w_kva.scalar_type() == at::kByte) {
    TORCH_CHECK(w_kva.is_contiguous() &&
                w_kva.device() == x_in.device() &&
                w_kva.numel() == 1 * hy4_w8::matrix_bytes(kKvADim, kHidden));
    p.kva8 = w_kva.const_data_ptr<uint8_t>();
  } else {
    p.w_kva = reinterpret_cast<const half*>(w_kva.const_data_ptr<at::Half>());
  }
  if (w_gate.scalar_type() == at::kByte) {
    TORCH_CHECK(w_gate.is_contiguous() &&
                w_gate.device() == x_in.device() &&
                w_gate.numel() == 1 * hy4_w8::matrix_bytes(heads * kVHead, kHidden));
    p.gate8 = w_gate.const_data_ptr<uint8_t>();
  } else {
    p.w_gate = reinterpret_cast<const half*>(w_gate.const_data_ptr<at::Half>());
  }
  p.gamma_qa = reinterpret_cast<const half*>(gamma_qa.const_data_ptr<at::Half>());
  p.gamma_kva = reinterpret_cast<const half*>(gamma_kva.const_data_ptr<at::Half>());
  if (w_qb.scalar_type() == at::kByte) {
    TORCH_CHECK(w_qb.is_contiguous() &&
                w_qb.device() == x_in.device() &&
                w_qb.numel() == 1 * hy4_w8::matrix_bytes(heads * (kQkNope + kRopeDim), kQRank));
    p.qb8 = w_qb.const_data_ptr<uint8_t>();
  } else {
    p.w_qb = reinterpret_cast<const half*>(w_qb.const_data_ptr<at::Half>());
  }
  if (w_uk_t.scalar_type() == at::kByte) {
    TORCH_CHECK(w_uk_t.is_contiguous() &&
                w_uk_t.device() == x_in.device() &&
                w_uk_t.numel() == heads * hy4_w8::matrix_bytes(kKvRank, kQkNope));
    p.uk8 = w_uk_t.const_data_ptr<uint8_t>();
  } else {
    p.w_uk_t = reinterpret_cast<const half*>(w_uk_t.const_data_ptr<at::Half>());
  }
  p.x_attn = fuse_in_norm
                 ? reinterpret_cast<half*>(x_attn->mutable_data_ptr<at::Half>())
                 : nullptr;
  p.q_a_raw = reinterpret_cast<half*>(q_a_raw.mutable_data_ptr<at::Half>());
  p.kv_a_raw = reinterpret_cast<half*>(kv_a_raw.mutable_data_ptr<at::Half>());
  p.q_nope = reinterpret_cast<half*>(q_nope.mutable_data_ptr<at::Half>());
  p.tail_partials = tail_partials.mutable_data_ptr<float>();
  p.q_out = reinterpret_cast<half*>(q_out.mutable_data_ptr<at::Half>());
  p.kv_c_out = reinterpret_cast<half*>(kv_c_out.mutable_data_ptr<at::Half>());
  p.k_pe_out = reinterpret_cast<half*>(k_pe_out.mutable_data_ptr<at::Half>());
  p.q_c_out = reinterpret_cast<half*>(q_c_out.mutable_data_ptr<at::Half>());
  p.gate_sig_out = reinterpret_cast<half*>(gate_sig_out.mutable_data_ptr<at::Half>());
  if (cache) {
    TORCH_CHECK(cache->is_cuda() && cache->scalar_type() == at::kByte &&
                cache->dim() == 3 && cache->size(2) == 656 && cache->stride(2) == 1);
    TORCH_CHECK(slots && slots->is_contiguous() && slots->numel() <= m);
    p.cache = cache->data_ptr<uint8_t>();
    p.slots = slots->const_data_ptr<int64_t>();
    p.cache_page_stride = cache->stride(0);
    p.cache_row_stride = cache->stride(1);
    p.cache_page = cache->size(1);
    p.cache_tokens = slots->numel();
    if (indices) {
      TORCH_CHECK(lengths && table && counts && indices->dim() == 2 &&
                  indices->size(0) == p.cache_tokens && indices->stride(1) == 1 &&
                  lengths->numel() >= p.cache_tokens && lengths->is_contiguous() &&
                  counts->numel() >= p.cache_tokens && counts->is_contiguous() &&
                  table->size(0) >= p.cache_tokens && table->stride(1) == 1);
      p.lengths = lengths->const_data_ptr<int32_t>();
      p.table = table->const_data_ptr<int32_t>();
      p.indices = indices->data_ptr<int32_t>();
      p.counts = counts->data_ptr<int32_t>();
      p.table_stride = table->stride(0);
      p.indices_stride = indices->stride(0);
      p.index_width = indices->size(1);
    }
  }
  p.m = m;
  p.heads = heads;
  p.eps = static_cast<float>(eps);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int rb = (m + 7) / 8;
  // Preserve the env switch for callers of the native op without a profile.
  const char* split_w8_env = std::getenv("HY4_PREP_SPLIT_W8");
  const bool split = split_w8.value_or(
      split_w8_env && split_w8_env[0] == '1' && split_w8_env[1] == '\0');
  if (split &&
      rb == 1 && !fuse_in_norm && (heads == 8 || heads == 16) &&
      p.qa8 && p.kva8 && p.gate8 && p.qb8 && p.uk8) {
    launch_split_w8_prep(p, stream);
    return;
  }
  const auto launch = [&](auto packed) {
    constexpr bool w8 = decltype(packed)::value;
    if (fuse_in_norm) {
      if (rb == 1) {
        launch_coop_prep<1, true, w8>(p, stream);
      } else if (rb == 2) {
        launch_coop_prep<2, true, w8>(p, stream);
      } else {
        launch_coop_prep<3, true, w8>(p, stream);
      }
    } else {
      if (rb == 1) {
        launch_coop_prep<1, false, w8>(p, stream);
      } else if (rb == 2) {
        launch_coop_prep<2, false, w8>(p, stream);
      } else {
        launch_coop_prep<3, false, w8>(p, stream);
      }
    }
  };
  if (p.qa8 || p.kva8 || p.gate8 || p.qb8 || p.uk8) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
}

void attn_qkv_prep_sm70_meta(
    const at::Tensor& x_in,
    const std::optional<at::Tensor>& gamma_in,
    const at::Tensor& cos_buf,
    const at::Tensor& sin_buf,
    const at::Tensor& w_qa,
    const at::Tensor& w_kva,
    const at::Tensor& w_gate,
    const at::Tensor& gamma_qa,
    const at::Tensor& gamma_kva,
    const at::Tensor& w_qb,
    const at::Tensor& w_uk_t,
    const std::optional<at::Tensor>& x_attn,
    at::Tensor& q_a_raw,
    at::Tensor& kv_a_raw,
    at::Tensor& q_nope,
    at::Tensor& tail_partials,
    at::Tensor& q_out,
    at::Tensor& kv_c_out,
    at::Tensor& k_pe_out,
    at::Tensor& q_c_out,
    at::Tensor& gate_sig_out,
    double eps,
    const std::optional<at::Tensor>& cache,
    const std::optional<at::Tensor>& slots,
    const std::optional<at::Tensor>& lengths,
    const std::optional<at::Tensor>& table,
    const std::optional<at::Tensor>& indices,
    const std::optional<at::Tensor>& counts,
    std::optional<bool> /*split_w8*/) {
  const int64_t m = x_in.size(0);
  const int64_t heads = w_uk_t.size(0);
  TORCH_CHECK(m >= 1 && m <= 24);
  TORCH_CHECK(heads == 8 || heads == 16);
  TORCH_CHECK(q_out.sizes() == at::IntArrayRef({m, heads, kKvADim}));
  TORCH_CHECK(kv_c_out.sizes() == at::IntArrayRef({m, kKvRank}));
  TORCH_CHECK(k_pe_out.sizes() == at::IntArrayRef({m, kRopeDim}));
  TORCH_CHECK(q_c_out.sizes() == at::IntArrayRef({m, kQRank}));
  TORCH_CHECK(gate_sig_out.sizes() == at::IntArrayRef({m, heads * kVHead}));
}

}  // namespace

TORCH_LIBRARY(hy4_f7_attn, m) {
  m.def(
      "rope_phases_sm70("
      "Tensor positions, Tensor inv_freq, Tensor(a!) cos_out, Tensor(b!) sin_out) -> ()");
  m.def(
      "attn_qkv_prep_sm70("
      "Tensor x_in, Tensor? gamma_in, Tensor cos_buf, Tensor sin_buf,"
      "Tensor w_qa, Tensor w_kva, Tensor w_gate, Tensor gamma_qa, Tensor gamma_kva,"
      "Tensor w_qb, Tensor w_uk_t,"
      "Tensor(a!)? x_attn, Tensor(b!) q_a_raw, Tensor(c!) kv_a_raw, Tensor(d!) q_nope,"
      "Tensor(e!) tail_partials,"
      "Tensor(f!) q_out, Tensor(g!) kv_c_out, Tensor(h!) k_pe_out,"
      "Tensor(i!) q_c_out, Tensor(j!) gate_sig_out, float eps,"
      "Tensor(k!)? cache=None, Tensor? slots=None, Tensor? lengths=None, Tensor? table=None,"
      "Tensor(l!)? indices=None, Tensor(m!)? counts=None, bool? split_w8=None) -> ()");
}

TORCH_LIBRARY_IMPL(hy4_f7_attn, CUDA, m) {
  m.impl("rope_phases_sm70", &rope_phases_sm70);
  m.impl("attn_qkv_prep_sm70", &attn_qkv_prep_sm70);
}

TORCH_LIBRARY_IMPL(hy4_f7_attn, Meta, m) {
  m.impl("rope_phases_sm70", &rope_phases_sm70_meta);
  m.impl("attn_qkv_prep_sm70", &attn_qkv_prep_sm70_meta);
}
