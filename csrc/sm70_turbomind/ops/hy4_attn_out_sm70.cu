// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// F7 Stage 2 (HY4_F7_ATTN_OUT) SM70 attention read-side kernels for Hy4-flash:
//   - K4 (attn_splitk_sm70): paged MLA split-K reader with device-side dynamic split
//     selection (compute_num_splits from live counts[row]), supporting both packed
//     fp8_ds_mla (656 B/row) and fp16 (576 elems/row) caches plus DFly draft_kv.
//   - K5 (attn_merge_v_gate_sm70): cooperative 2-phase kernel fusing split-K merge
//     (with learnable sinks) + absorbed W_uv BMM (in-place strided kv_b_proj slice) +
//     elementwise gate_sig multiply in a single launch.
//
// Volta sm_70 + nvcc 12.8 mandatory correctness rules (carried from F7 prototype):
//   Rule 1 (Cross-SM L1 non-coherence): Inter-phase reads of o_latent_scratch in K5
//     Phase 1 after grid.sync() MUST bypass per-SM L1 via L2-coherent __ldcg.
//   Rule 2 (nvcc 12.8 __restrict__ dead-load elision across BAR.SYNC): Every shared-
//     memory reduction buffer passed to device helpers MUST be volatile-qualified
//     (volatile float*).

#include "hy4_w8_sm70.cuh"

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
constexpr int kWarpsK = 4;
constexpr int kTileN = 16;
constexpr int kKvRank = 512;
constexpr int kRopeDim = 64;
constexpr int kKvADim = kKvRank + kRopeDim;  // 576
constexpr int kVHead = 256;
constexpr int kMaxSplitsCap = 32;
constexpr int kShortKeys = 512;
constexpr int kShortTile = 8;
constexpr int kShortSplits = kShortKeys / kShortTile;

// Shared device-side split count formula used identically by K4 and K5.
// Depends ONLY on counts[row] and static max_splits -> 100% row-invariant
// and zero host-side data-dependent graph capture decisions.
__device__ __forceinline__ int compute_num_splits(int length, int max_splits) {
  if (length <= 0) return 0;
  int ns = (length + 15) >> 4;  // ~16 keys/split at short prefixes, caps at max_splits
  return ns < max_splits ? ns : max_splits;
}

// Exact bit-for-bit IEEE E4M3FN byte -> FP32 decode (zero LUT bank conflicts).
__device__ __forceinline__ float fp8_e4m3fn_to_fp32(uint8_t b) {
  const uint32_t s = (static_cast<uint32_t>(b & 0x80u)) << 24;
  const uint32_t em = b & 0x7Fu;
  if (em == 0x7Fu) {
    return __uint_as_float(s | 0x7FC00000u);
  }
  if (em >= 8u) {
    return __uint_as_float(s | ((em + (120u << 3)) << 20));
  }
  const float v = static_cast<float>(em) * (1.0f / 512.0f);
  return s ? -v : v;
}

// Expand four finite E4M3 codes into exact half values scaled by 2^-8.
// This is the byte-permute decoder from the NATIVE reader probe.
__device__ __forceinline__ uint2 fp8x4_to_half4(uint32_t codes, float scale256) {
  const uint32_t lo = __byte_perm(codes, 0, 0x1404);
  const uint32_t hi = __byte_perm(codes, 0, 0x3424);
  const uint32_t halves0 = (lo & 0x80008000u) | ((lo & 0x7f007f00u) >> 1);
  const uint32_t halves1 = (hi & 0x80008000u) | ((hi & 0x7f007f00u) >> 1);
  const float2 f0 = __half22float2(*reinterpret_cast<const half2*>(&halves0));
  const float2 f1 = __half22float2(*reinterpret_cast<const half2*>(&halves1));
  const half2 out0 = __floats2half2_rn(f0.x * scale256, f0.y * scale256);
  const half2 out1 = __floats2half2_rn(f1.x * scale256, f1.y * scale256);
  return make_uint2(*reinterpret_cast<const uint32_t*>(&out0),
                    *reinterpret_cast<const uint32_t*>(&out1));
}

__device__ __forceinline__ bool fp8_has_nan(uint32_t codes) {
  return (((codes & 0x7f7f7f7fu) + 0x01010101u) & 0x80808080u) != 0;
}

template <bool FastDecode>
__device__ __forceinline__ uint2 decode_legacy_fp8x4(uint32_t codes, float scale) {
  if constexpr (FastDecode) {
    const float scale256 = scale * 256.0f;
    if (isfinite(scale256) && !fp8_has_nan(codes)) {
      return fp8x4_to_half4(codes, scale256);
    }
  }
  const half2 lo = __floats2half2_rn(fp8_e4m3fn_to_fp32(codes & 0xffu) * scale,
                                    fp8_e4m3fn_to_fp32((codes >> 8) & 0xffu) * scale);
  const half2 hi = __floats2half2_rn(fp8_e4m3fn_to_fp32((codes >> 16) & 0xffu) * scale,
                                    fp8_e4m3fn_to_fp32(codes >> 24) * scale);
  return make_uint2(*reinterpret_cast<const uint32_t*>(&lo),
                    *reinterpret_cast<const uint32_t*>(&hi));
}

// =============================================================================
// K4: Paged MLA split-K attention reader (FP8 packed & FP16, target & DFly)
// =============================================================================

struct SplitKParams {
  const half* q;               // [M, H, 576]
  const void* kv_cache;        // uint8 [num_blocks, page, 656] or half [num_blocks, page, 576]
  const void* draft_kv;        // optional [batch, query_width, row_bytes]
  const int32_t* indices;      // [M, width] or [batch, width]
  const int32_t* counts;       // [M]
  float* state;                // [>=M, H, ws_splits, 2]
  float* acc;                  // [>=M, H, ws_splits, 512]
  int64_t q_row_stride;
  int64_t q_head_stride;
  int64_t kv_page_stride;      // in bytes (IsFp8) or halfs (!IsFp8)
  int64_t kv_row_stride;       // in bytes (IsFp8) or halfs (!IsFp8)
  int64_t draft_row_stride;    // in bytes (IsFp8) or halfs (!IsFp8)
  int64_t indices_row_stride;
  int m;
  int heads;
  int page_size;
  int ws_splits;
  int max_splits;
  int query_width;
  float scale;
};

template <bool FastDecode>
__device__ void read_short_split(const SplitKParams& p, int length);

template <bool IsFp8, bool HasDraft, bool Short = false, bool Bounded = false, bool FastDecode = false>
__global__ void __launch_bounds__(kThreads, 4)
    hy4_attn_splitk_kernel(const SplitKParams p) {
  const int split = blockIdx.x;
  const int h_base = blockIdx.y * 8;
  const int row = blockIdx.z;

  const int length = p.counts[row];
  if constexpr (Short) {
    if (Bounded || length <= kShortKeys) {
      read_short_split<FastDecode>(p, length);
      return;
    }
  }
  const int ns = compute_num_splits(length, p.max_splits);
  if (split >= ns) return;

  const int k_start = (split * length) / ns;
  const int k_end = ((split + 1) * length) / ns;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;   // 0..7 -> head offset within 8-head tile
  const int lane = tid & 31;   // 0..31 -> 4 groups x 4 elems = 16 latent elems + 2 rope elems
  const int h = h_base + warp;
  const bool head_ok = h < p.heads;

  // Bank-conflict-free element mapping: group g in 0..3, elem = g*128 + lane*4 + {0..3}.
  // Store Q_c as packed half2[8] (8 regs vs 16) so K4 hits 4 CTAs/SM with 0 stack spills.
  half2 q_c_h2[8];
  float q_r[2];
  float acc_r[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc_r[i] = 0.f;

  if (head_ok) {
    const half* qb = p.q + static_cast<int64_t>(row) * p.q_row_stride +
                     static_cast<int64_t>(h) * p.q_head_stride;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const uint2 q4 = __ldg(reinterpret_cast<const uint2*>(qb + g * 128 + lane * 4));
      q_c_h2[g * 2 + 0] = *reinterpret_cast<const half2*>(&q4.x);
      q_c_h2[g * 2 + 1] = *reinterpret_cast<const half2*>(&q4.y);
    }
    const uint32_t qr_u = __ldg(reinterpret_cast<const uint32_t*>(qb + kKvRank + lane * 2));
    const half2 qr_h = *reinterpret_cast<const half2*>(&qr_u);
    q_r[0] = __half2float(qr_h.x);
    q_r[1] = __half2float(qr_h.y);
  }

  float m_val = -INFINITY;
  float total = 0.f;

  // Cooperative 4-token KV buffer in shared memory (4.5 KB/CTA -> 4 CTAs/SM).
  __shared__ half smem_k[4][kKvRank];
  __shared__ half smem_kr[4][kRopeDim];

  const int tok_j = tid >> 6;   // 0..3
  const int t_lane = tid & 63;  // 0..63 (64 threads/token -> 8 latent elems + 1 rope elem)

  const int request = HasDraft ? (row / p.query_width) : row;
  const int history = HasDraft ? (length - (row % p.query_width) - 1) : length;

  for (int t = k_start; t < k_end; t += 4) {
    const int n_tok = (k_end - t) < 4 ? (k_end - t) : 4;
    if (tok_j < n_tok) {
      const int pos_n = t + tok_j;
      const bool in_history = !HasDraft || (pos_n < history);
      if constexpr (IsFp8) {
        const uint8_t* row_u8;
        if (in_history) {
          const int32_t slot =
              __ldg(p.indices + static_cast<int64_t>(request) * p.indices_row_stride + pos_n);
          const int64_t page = static_cast<int64_t>(slot / p.page_size);
          const int64_t off = static_cast<int64_t>(slot - static_cast<int32_t>(page) * p.page_size);
          row_u8 = reinterpret_cast<const uint8_t*>(p.kv_cache) +
                   page * p.kv_page_stride + off * p.kv_row_stride;
        } else {
          const int64_t draft_row =
              static_cast<int64_t>(request) * p.query_width + (pos_n - history);
          row_u8 = reinterpret_cast<const uint8_t*>(p.draft_kv) +
                   draft_row * p.draft_row_stride;
        }
        // Coalesced 64-bit load of 8 FP8 codes + group FP32 scale.
        const uint2 raw8 = __ldg(reinterpret_cast<const uint2*>(row_u8 + t_lane * 8));
        const int grp = t_lane >> 4;  // 0..3 (16 threads x 8 = 128 elems per group)
        const float sc = __ldg(reinterpret_cast<const float*>(row_u8 + kKvRank + grp * 4));
        if constexpr (FastDecode) {
          const uint2 lo = decode_legacy_fp8x4<true>(raw8.x, sc);
          const uint2 hi = decode_legacy_fp8x4<true>(raw8.y, sc);
          *reinterpret_cast<uint4*>(&smem_k[tok_j][t_lane * 8]) =
              make_uint4(lo.x, lo.y, hi.x, hi.y);
        } else {
          const uint32_t w0 = raw8.x;
          const uint32_t w1 = raw8.y;
          half2 out0, out1, out2, out3;
          out0.x = __float2half_rn(fp8_e4m3fn_to_fp32(w0 & 0xFFu) * sc);
          out0.y = __float2half_rn(fp8_e4m3fn_to_fp32((w0 >> 8) & 0xFFu) * sc);
          out1.x = __float2half_rn(fp8_e4m3fn_to_fp32((w0 >> 16) & 0xFFu) * sc);
          out1.y = __float2half_rn(fp8_e4m3fn_to_fp32((w0 >> 24) & 0xFFu) * sc);
          out2.x = __float2half_rn(fp8_e4m3fn_to_fp32(w1 & 0xFFu) * sc);
          out2.y = __float2half_rn(fp8_e4m3fn_to_fp32((w1 >> 8) & 0xFFu) * sc);
          out3.x = __float2half_rn(fp8_e4m3fn_to_fp32((w1 >> 16) & 0xFFu) * sc);
          out3.y = __float2half_rn(fp8_e4m3fn_to_fp32((w1 >> 24) & 0xFFu) * sc);
          uint4 pack;
          pack.x = *reinterpret_cast<const uint32_t*>(&out0);
          pack.y = *reinterpret_cast<const uint32_t*>(&out1);
          pack.z = *reinterpret_cast<const uint32_t*>(&out2);
          pack.w = *reinterpret_cast<const uint32_t*>(&out3);
          *reinterpret_cast<uint4*>(&smem_k[tok_j][t_lane * 8]) = pack;
        }
        smem_kr[tok_j][t_lane] =
            __ldg(reinterpret_cast<const half*>(row_u8 + kKvRank + 16) + t_lane);
      } else {
        const half* row_h;
        if (in_history) {
          const int32_t slot =
              __ldg(p.indices + static_cast<int64_t>(request) * p.indices_row_stride + pos_n);
          const int64_t page = static_cast<int64_t>(slot / p.page_size);
          const int64_t off = static_cast<int64_t>(slot - static_cast<int32_t>(page) * p.page_size);
          row_h = reinterpret_cast<const half*>(p.kv_cache) +
                  page * p.kv_page_stride + off * p.kv_row_stride;
        } else {
          const int64_t draft_row =
              static_cast<int64_t>(request) * p.query_width + (pos_n - history);
          row_h = reinterpret_cast<const half*>(p.draft_kv) +
                  draft_row * p.draft_row_stride;
        }
        *reinterpret_cast<uint4*>(&smem_k[tok_j][t_lane * 8]) =
            __ldg(reinterpret_cast<const uint4*>(row_h + t_lane * 8));
        smem_kr[tok_j][t_lane] = __ldg(row_h + kKvRank + t_lane);
      }
    }
    __syncthreads();

    if (head_ok) {
      for (int j = 0; j < n_tok; ++j) {
        uint2 k4_g[4];
        float dot = 0.f;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          k4_g[g] = *reinterpret_cast<const uint2*>(&smem_k[j][g * 128 + lane * 4]);
          const half2 kh0 = *reinterpret_cast<const half2*>(&k4_g[g].x);
          const half2 kh1 = *reinterpret_cast<const half2*>(&k4_g[g].y);
          const half2 qh0 = q_c_h2[g * 2 + 0];
          const half2 qh1 = q_c_h2[g * 2 + 1];
          dot = fmaf(__half2float(qh0.x), __half2float(kh0.x), dot);
          dot = fmaf(__half2float(qh0.y), __half2float(kh0.y), dot);
          dot = fmaf(__half2float(qh1.x), __half2float(kh1.x), dot);
          dot = fmaf(__half2float(qh1.y), __half2float(kh1.y), dot);
        }
        const half2 kr_h = *reinterpret_cast<const half2*>(&smem_kr[j][lane * 2]);
        dot = fmaf(q_r[0], __half2float(kr_h.x), dot);
        dot = fmaf(q_r[1], __half2float(kr_h.y), dot);

        // Deterministic 32-lane warp all-reduce.
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          dot += __shfl_xor_sync(0xffffffffu, dot, off);
        }
        const float logit = dot * p.scale;
        const float new_m = fmaxf(m_val, logit);
        const float rescale = (m_val == -INFINITY) ? 0.f : expf(m_val - new_m);
        const float prob = expf(logit - new_m);
        total = total * rescale + prob;
        m_val = new_m;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          const half2 kh0 = *reinterpret_cast<const half2*>(&k4_g[g].x);
          const half2 kh1 = *reinterpret_cast<const half2*>(&k4_g[g].y);
          acc_r[g * 4 + 0] = fmaf(prob, __half2float(kh0.x), acc_r[g * 4 + 0] * rescale);
          acc_r[g * 4 + 1] = fmaf(prob, __half2float(kh0.y), acc_r[g * 4 + 1] * rescale);
          acc_r[g * 4 + 2] = fmaf(prob, __half2float(kh1.x), acc_r[g * 4 + 2] * rescale);
          acc_r[g * 4 + 3] = fmaf(prob, __half2float(kh1.y), acc_r[g * 4 + 3] * rescale);
        }
      }
    }
    __syncthreads();
  }

  if (head_ok) {
    const int64_t slot_idx =
        (static_cast<int64_t>(row) * p.heads + h) * p.ws_splits + split;
    if (lane == 0) {
      p.state[slot_idx * 2 + 0] = m_val;
      p.state[slot_idx * 2 + 1] = total;
    }
    float* acc_dst = p.acc + slot_idx * kKvRank;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      float4 out4 = make_float4(
          acc_r[g * 4 + 0], acc_r[g * 4 + 1], acc_r[g * 4 + 2], acc_r[g * 4 + 3]);
      *reinterpret_cast<float4*>(acc_dst + g * 128 + lane * 4) = out4;
    }
  }
}

__global__ void __launch_bounds__(256, 3)
hy4_attn_swa_tensor_split(const SplitKParams p);

void attn_splitk_sm70(
    const at::Tensor& q,
    const at::Tensor& kv_cache,
    const std::optional<at::Tensor>& draft_kv,
    const at::Tensor& indices,
    const at::Tensor& counts,
    at::Tensor& state,
    at::Tensor& acc,
    double scale,
    int64_t max_splits,
    int64_t query_width,
    bool short_splits,
    bool prmt_fp8,
    bool swa_tensor) {
  TORCH_CHECK(q.is_cuda() && kv_cache.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(q.dim() == 3 && q.size(2) == kKvADim, "q must be [M, H, 576]");
  TORCH_CHECK(q.scalar_type() == at::kHalf, "q must be float16");
  TORCH_CHECK(q.stride(2) == 1, "q last dim must be contiguous");
  const int m = static_cast<int>(q.size(0));
  const int heads = static_cast<int>(q.size(1));
  const int ws_splits = static_cast<int>(state.size(2));
  TORCH_CHECK(m >= 1 && heads >= 1, "non-empty q required");
  TORCH_CHECK(state.is_contiguous() && acc.is_contiguous(), "state/acc must be contiguous");
  TORCH_CHECK(state.size(0) >= m && state.size(1) == heads && state.size(3) == 2);
  TORCH_CHECK(acc.size(0) >= m && acc.size(1) == heads && acc.size(2) == ws_splits &&
              acc.size(3) == kKvRank);
  TORCH_CHECK(max_splits >= 1 && max_splits <= ws_splits && max_splits <= kMaxSplitsCap,
              "invalid max_splits=", max_splits, " ws_splits=", ws_splits);

  const bool is_fp8 = kv_cache.scalar_type() == at::kByte;
  TORCH_CHECK(is_fp8 || kv_cache.scalar_type() == at::kHalf,
              "kv_cache must be uint8 (fp8_ds_mla) or float16");
  const bool has_draft = draft_kv.has_value();
  if (has_draft) {
    TORCH_CHECK(query_width > 0 && (m % query_width == 0),
                "invalid query_width=", query_width, " for m=", m);
    TORCH_CHECK(draft_kv->scalar_type() == kv_cache.scalar_type());
  }

  const c10::cuda::CUDAGuard guard(q.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  SplitKParams p{};
  p.q = reinterpret_cast<const half*>(q.const_data_ptr<at::Half>());
  p.kv_cache = kv_cache.const_data_ptr();
  p.draft_kv = has_draft ? draft_kv->const_data_ptr() : nullptr;
  p.indices = indices.const_data_ptr<int32_t>();
  p.counts = counts.const_data_ptr<int32_t>();
  p.state = state.mutable_data_ptr<float>();
  p.acc = acc.mutable_data_ptr<float>();
  p.q_row_stride = q.stride(0);
  p.q_head_stride = q.stride(1);
  p.kv_page_stride = kv_cache.stride(0);
  p.kv_row_stride = kv_cache.stride(1);
  p.draft_row_stride = has_draft ? draft_kv->stride(1) : 0;
  p.indices_row_stride = indices.stride(0);
  p.m = m;
  p.heads = heads;
  p.page_size = static_cast<int>(kv_cache.size(1));
  p.ws_splits = ws_splits;
  p.max_splits = static_cast<int>(max_splits);
  p.query_width = static_cast<int>(query_width);
  p.scale = static_cast<float>(scale);

  TORCH_CHECK(!short_splits || (is_fp8 && !has_draft && m == 1 &&
                                ws_splits >= kShortSplits && (heads == 8 || heads == 16)),
              "short splits require one FP8 query row, 8 or 16 heads, and 64 workspace splits");
  const dim3 grid(static_cast<unsigned>(short_splits ? kShortSplits : max_splits),
                  static_cast<unsigned>((heads + 7) / 8),
                  static_cast<unsigned>(m));
  if (swa_tensor && is_fp8 && prmt_fp8 && !has_draft && !short_splits &&
      m >= 2 && m <= 4 && heads == 16 && indices.size(1) <= kShortKeys &&
      indices.size(1) <= 32 * max_splits &&
      p.kv_row_stride % 16 == 0 && p.kv_page_stride % 16 == 0 &&
      reinterpret_cast<uintptr_t>(p.kv_cache) % 16 == 0 &&
      p.q_head_stride % 4 == 0 && p.q_row_stride % 4 == 0 &&
      reinterpret_cast<uintptr_t>(p.q) % 8 == 0) {
    hy4_attn_swa_tensor_split<<<grid, kThreads, 0, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  const auto launch_fp8 = [&](auto fast_decode) {
    constexpr bool fast = decltype(fast_decode)::value;
    if (short_splits) {
      if (indices.size(1) <= kShortKeys) {
        hy4_attn_splitk_kernel<true, false, true, true, fast><<<grid, kThreads, 0, stream>>>(p);
      } else {
        hy4_attn_splitk_kernel<true, false, true, false, fast><<<grid, kThreads, 0, stream>>>(p);
      }
    } else if (has_draft) {
      hy4_attn_splitk_kernel<true, true, false, false, fast><<<grid, kThreads, 0, stream>>>(p);
    } else {
      hy4_attn_splitk_kernel<true, false, false, false, fast><<<grid, kThreads, 0, stream>>>(p);
    }
  };
  if (is_fp8) {
    if (prmt_fp8) {
      launch_fp8(std::true_type{});
    } else {
      launch_fp8(std::false_type{});
    }
  } else {
    if (has_draft) {
      hy4_attn_splitk_kernel<false, true><<<grid, kThreads, 0, stream>>>(p);
    } else {
      hy4_attn_splitk_kernel<false, false><<<grid, kThreads, 0, stream>>>(p);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void attn_splitk_sm70_meta(
    const at::Tensor& q,
    const at::Tensor& kv_cache,
    const std::optional<at::Tensor>& draft_kv,
    const at::Tensor& indices,
    const at::Tensor& counts,
    at::Tensor& state,
    at::Tensor& acc,
    double scale,
    int64_t max_splits,
    int64_t query_width,
    bool short_splits,
    bool prmt_fp8,
    bool swa_tensor) {
  TORCH_CHECK(q.dim() == 3 && q.size(2) == kKvADim);
  TORCH_CHECK(max_splits >= 1 && max_splits <= kMaxSplitsCap);
}

// =============================================================================
// K5: Cooperative split-K merge + absorbed W_uv BMM + elementwise gate_sig
// =============================================================================

struct MergeVGateParams {
  const float* state;          // [>=M, H, ws_splits, 2]
  const float* acc;            // [>=M, H, ws_splits, 512]
  const int32_t* counts;       // [M]
  const float* sinks;          // [H]
  const half* w_uv;            // [H, 256, 512] row-contiguous, head stride = w_uv_head_stride
  const half* gate_sig;        // [M, H * 256] contiguous
  half* o_latent;              // [H, M, 512] scratch
  half* out;                   // [M, H * 256] contiguous
  int64_t w_uv_head_stride;
  const uint8_t* uv8;
  int m;
  int heads;
  int ws_splits;
  int max_splits;
  bool short_splits;
};

#define HY4_F7_OUT_MMA(C, A0, A1, B0, B1)                           \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};"                                  \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

__device__ __forceinline__ int out_acc_row(int i, int lane) {
  return (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
}
__device__ __forceinline__ int out_acc_col(int i, int lane) {
  return (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
}
__device__ __forceinline__ uint4 out_zero4() { return make_uint4(0, 0, 0, 0); }

__device__ __forceinline__ uint32_t pack_halves(half lo, half hi) {
  return __half_as_ushort(lo) | (static_cast<uint32_t>(__half_as_ushort(hi)) << 16);
}

// One CTA consumes eight packed keys for eight heads, with no online rescaling
// within the split. Four independent m8n8k4 groups per warp accumulate QK over
// disjoint dimensions, then cover separate output columns for PV.
template <bool FastDecode>
__device__ void read_short_split(const SplitKParams& p, int length) {
  const int split = blockIdx.x;
  const int row = blockIdx.z;
  const int start = split * kShortTile;
  if (start >= length) return;
  const int live = min(kShortTile, length - start);
  const int head_base = blockIdx.y * 8;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int quad = (lane >> 2) & 3;
  const int fragment_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  constexpr int kPackedStride = kKvADim + 4;
  __shared__ __align__(8) half key[8][kPackedStride];
  __shared__ volatile float dot_parts[8][8][8];
  __shared__ __align__(16) half probability[8][8];

  uint4 raw = make_uint4(0, 0, 0, 0);
  float scale = 0.f;
  uint32_t rope = 0;
  if (warp < live) {
    const int slot = __ldg(p.indices + row * p.indices_row_stride + start + warp);
    const uint8_t* src = static_cast<const uint8_t*>(p.kv_cache) +
                         static_cast<int64_t>(slot / p.page_size) * p.kv_page_stride +
                         (slot % p.page_size) * p.kv_row_stride;
    raw = __ldg(reinterpret_cast<const uint4*>(src + lane * 16));
    scale = __ldg(reinterpret_cast<const float*>(src + kKvRank) + (lane >> 3)) * 256.f;
    rope = __ldg(reinterpret_cast<const uint32_t*>(src + kKvRank + 16) + lane);
  }
  const uint32_t codes[] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
  for (int word = 0; word < 4; ++word) {
    if constexpr (FastDecode) {
      if (!fp8_has_nan(codes[word])) {
        *reinterpret_cast<uint2*>(&key[warp][lane * 16 + word * 4]) =
            fp8x4_to_half4(codes[word], scale);
        continue;
      }
    }
#pragma unroll
    for (int byte = 0; byte < 4; ++byte) {
      const uint32_t code = (codes[word] >> (byte * 8)) & 0xffu;
      const uint16_t half_bits = ((code & 0x80u) << 8) | ((code & 0x7fu) << 7);
      const float value = (code & 0x7fu) == 0x7fu
          ? __int_as_float(0x7fc00000) : __half2float(__ushort_as_half(half_bits)) * scale;
      key[warp][lane * 16 + word * 4 + byte] = __float2half_rn(value);
    }
  }
  *reinterpret_cast<uint32_t*>(&key[warp][kKvRank + lane * 2]) = rope;
  __syncthreads();

  const half* query = p.q + row * p.q_row_stride +
                      (head_base + fragment_row) * p.q_head_stride;
  float logits[8] = {};
  // Cover 144 MMA steps across 32 independent fragments.
#pragma unroll
  for (int base = 0; base < kKvADim; base += 128) {
    const int d = base + (warp * 4 + quad) * 4;
    uint2 a = make_uint2(0, 0);
    uint2 b = make_uint2(0, 0);
    if (d < kKvADim) {
      a = __ldg(reinterpret_cast<const uint2*>(query + d));
      b = *reinterpret_cast<const uint2*>(&key[fragment_row][d]);
    }
    HY4_F7_OUT_MMA(logits, a.x, a.y, b.x, b.y);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int r = out_acc_row(i, lane);
    const int c = out_acc_col(i, lane);
    float sum = __shfl_sync(0xffffffffu, logits[i], lane & ~12);
#pragma unroll
    for (int q = 1; q < 4; ++q) {
      sum += __shfl_sync(0xffffffffu, logits[i], (lane & ~12) | (q << 2));
    }
    if (quad == 0) dot_parts[warp][r][c] = sum;
  }
  __syncthreads();

  // Process one key per lane. Softmax reductions use eight-lane groups,
  // with round-to-half only on the probabilities fed to PV.
  if (tid < 8 * kShortTile) {
    const int h = tid / kShortTile;
    const int n = tid % kShortTile;
    float dot = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) dot += dot_parts[w][h][n];
    const float score = n < live ? dot * p.scale : -INFINITY;
    float maximum = score;
#pragma unroll
    for (int delta = kShortTile / 2; delta > 0; delta /= 2) {
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta, kShortTile));
    }
    const float prob = expf(score - maximum);
    probability[h][n] = __float2half_rn(prob);
    // Accumulate probabilities in key order.
    float total = __shfl_sync(0xffffffffu, prob, 0, kShortTile);
#pragma unroll
    for (int i = 1; i < kShortTile; ++i) total += __shfl_sync(0xffffffffu, prob, i, kShortTile);
    if (n == 0) {
      const int64_t index =
          (static_cast<int64_t>(row) * p.heads + head_base + h) * p.ws_splits + split;
      p.state[2 * index] = maximum;
      p.state[2 * index + 1] = total;
    }
  }
  __syncthreads();

  const uint4 probs = *reinterpret_cast<const uint4*>(probability[fragment_row]);
#pragma unroll
  for (int base = 0; base < kKvRank; base += 256) {
    const int col = base + (warp * 4 + quad) * 8;
    const int d = col + fragment_row;
    float values[8] = {};
    const uint32_t k01 = pack_halves(key[0][d], key[1][d]);
    const uint32_t k23 = pack_halves(key[2][d], key[3][d]);
    const uint32_t k45 = pack_halves(key[4][d], key[5][d]);
    const uint32_t k67 = pack_halves(key[6][d], key[7][d]);
    HY4_F7_OUT_MMA(values, probs.x, probs.y, k01, k23);
    HY4_F7_OUT_MMA(values, probs.z, probs.w, k45, k67);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int64_t index =
          (static_cast<int64_t>(row) * p.heads + head_base + out_acc_row(i, lane)) *
              p.ws_splits + split;
      p.acc[index * kKvRank + col + out_acc_col(i, lane)] = values[i];
    }
  }
}

// Compute up to 32 keys in each legacy SWA partition. Two warps share the
// QK reduction for each group of eight keys. The merge sees one FP32
// numerator and two FP32 softmax statistics per head and partition.
__global__ void __launch_bounds__(256, 3)
hy4_attn_swa_tensor_split(const SplitKParams p) {
  const int row = blockIdx.z;
  const int split = blockIdx.x;
  const int head_base = blockIdx.y * 8;
  const int length = p.counts[row];
  const int ns = compute_num_splits(length, p.max_splits);
  if (split >= ns) return;
  const int start = split * length / ns;
  const int live = (split + 1) * length / ns - start;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int quad = (lane >> 2) & 3;
  const int frag = (lane & 3) + ((lane & 16) ? 4 : 0);
  constexpr int kKeys = 32;
  constexpr int kPackedStride = kKvADim + 8;
  __shared__ __align__(16) half keys[kKeys][kPackedStride];
  __shared__ float scores[2][8][kKeys];
  __shared__ __align__(16) half probs[8][kKeys];

  const int groups = (live + 7) / 8;
#pragma unroll 1
  for (int j = warp; j < groups * 8; j += 8) {
    uint4 codes = {};
    uint32_t rope = 0;
    float scale = 0.f;
    if (j < live) {
      const int slot = __ldg(p.indices + row * p.indices_row_stride + start + j);
      const int page = slot / p.page_size;
      const auto* src = static_cast<const uint8_t*>(p.kv_cache) +
                        page * p.kv_page_stride + (slot - page * p.page_size) * p.kv_row_stride;
      codes = __ldg(reinterpret_cast<const uint4*>(src) + lane);
      scale = __ldg(reinterpret_cast<const float*>(src + kKvRank) + (lane >> 3));
      rope = __ldg(reinterpret_cast<const uint32_t*>(src + kKvRank + 16) + lane);
    }
    const uint32_t words[] = {codes.x, codes.y, codes.z, codes.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      *reinterpret_cast<uint2*>(&keys[j][lane * 16 + i * 4]) =
          decode_legacy_fp8x4<true>(words[i], scale);
    }
    reinterpret_cast<uint32_t*>(keys[j] + kKvRank)[lane] = rope;
  }
  __syncthreads();

  const int group = warp / 2;
  const int k_half = warp % 2;
  if (group < groups) {
    const half* query = p.q + row * p.q_row_stride + (head_base + frag) * p.q_head_stride;
    float dots[8] = {};
    __syncwarp();
#pragma unroll 1
    for (int offset = 0; offset < kKvADim; offset += 32) {
      const int d = k_half * 16 + quad * 4 + offset;
      const uint2 q = __ldg(reinterpret_cast<const uint2*>(query + d));
      const uint2 k = *reinterpret_cast<const uint2*>(&keys[group * 8 + frag][d]);
      HY4_F7_OUT_MMA(dots, q.x, q.y, k.x, k.y);
    }
    __syncwarp();
    float quad_sums[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float sum = __shfl_sync(0xffffffffu, dots[i], lane & ~12);
#pragma unroll
      for (int q = 1; q < 4; ++q) {
        sum += __shfl_sync(0xffffffffu, dots[i], (lane & ~12) | (q << 2));
      }
      quad_sums[i] = sum;
    }
    if (quad == 0) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        scores[k_half][out_acc_row(i, lane)][group * 8 + out_acc_col(i, lane)] = quad_sums[i];
      }
    }
  }
  __syncthreads();

  const float score = lane < live
      ? (scores[0][warp][lane] + scores[1][warp][lane]) * p.scale
      : -INFINITY;
  float maximum = score;
  __syncwarp();
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
  }
  const float prob = expf(score - maximum);
  probs[warp][lane] = __float2half_rn(prob);
  float total = prob;
  __syncwarp();
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    total += __shfl_xor_sync(0xffffffffu, total, offset);
  }
  if (lane == 0) {
    const int64_t slot = (static_cast<int64_t>(row) * p.heads + head_base + warp) *
                          p.ws_splits + split;
    p.state[slot * 2] = maximum;
    p.state[slot * 2 + 1] = total;
  }
  __syncthreads();

  __syncwarp();
#pragma unroll 1
  for (int base = 0; base < kKvRank; base += 256) {
    const int col = base + (warp * 4 + quad) * 8;
    const int d = col + frag;
    float values[8] = {};
#pragma unroll 1
    for (int j = 0; j < groups * 8; j += 4) {
      const uint2 a = *reinterpret_cast<const uint2*>(&probs[frag][j]);
      const uint32_t b0 = pack_halves(keys[j][d], keys[j + 1][d]);
      const uint32_t b1 = pack_halves(keys[j + 2][d], keys[j + 3][d]);
      HY4_F7_OUT_MMA(values, a.x, a.y, b0, b1);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int64_t slot = (static_cast<int64_t>(row) * p.heads + head_base +
                            out_acc_row(i, lane)) * p.ws_splits + split;
      p.acc[slot * kKvRank + col + out_acc_col(i, lane)] = values[i];
    }
  }
}


// One CTA owns eight heads of one query. Keep the 16-key online softmax
// boundary of the Triton reader, including its FP16 probability rounding.
//
// Mode bits (HY4_PF_ATTN_FIX) bring the reader closer to the Triton FMA chains:
//   kPrefillPromotePv: each 16-key tile's PV product starts from zero in the
//     HMMA accumulator and is added to the FP32 running output with RN, as in
//     Triton's acc * rescale + p @ v. Volta HMMA truncates when it accumulates,
//     so feeding the running output through 512 sparse-layer HMMAs biases it.
//   kPrefillPromoteQk: the same for each four-channel QK step.
//   kPrefillDivFull: normalize with div.full.f32 like the compiled Triton reader.
//   kPrefillPromotePvStep: start every four-key PV HMMA from zero and add it to
//     the FP32 output with RN, so no running sum passes through HMMA.
constexpr int kPrefillPromotePv = 1;
constexpr int kPrefillPromoteQk = 2;
constexpr int kPrefillDivFull = 4;
constexpr int kPrefillPromotePvStep = 8;

struct PrefillParams {
  const half* q;
  const char* cache;
  const int32_t* slots;
  const int32_t* counts;
  const float* sinks;
  half* out;
  float* partial;
  float* lse;
  int64_t q_row_stride, q_head_stride;
  int64_t cache_page_bytes, cache_row_bytes, slots_stride;
  int heads, page;
  float scale;
};

// Match Triton's FP32 base conversion and approximate exponent instruction.
__device__ __forceinline__ float prefill_exp(float x) {
  float result;
  const float exponent = x * 1.4426950408889634f;
  asm("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(exponent));
  return result;
}

template <bool IsFp8, int Mode>
__global__ void __launch_bounds__(256, 3)
hy4_paged_prefill(const PrefillParams p) {
  constexpr bool PromotePvStep = Mode & kPrefillPromotePvStep;
  constexpr bool PromotePv = (Mode & kPrefillPromotePv) && !PromotePvStep;
  constexpr bool PromoteQk = Mode & kPrefillPromoteQk;
  constexpr bool DivFull = Mode & kPrefillDivFull;
  constexpr int Tile = 16;
  // The four quadpairs read adjacent 8-byte vectors for each key.
  // Stagger successive keys by eight shared-memory banks.
  constexpr int KeyStride = kKvADim + 16;
  const int row = blockIdx.x;
  const int head_base = blockIdx.y * 8;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int quad = (lane >> 2) & 3;
  const int frag = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int length = p.counts[row];
  __shared__ __align__(16) half keys[Tile][KeyStride];
  __shared__ volatile float scores[4][8][Tile];
  __shared__ volatile float rope_scores[4][8][Tile];
  __shared__ __align__(16) half probs[8][Tile];
  __shared__ float rescale[8];
  __shared__ float inv_denominator[8];
  __shared__ float denominator[8];

  const int h = tid / Tile;
  const int n = tid % Tile;
  const int head = head_base + h;
  float maximum = -INFINITY;
  float total = 0.f;
  if (h < 8 && head < p.heads && p.sinks) {
    maximum = p.sinks[head];
    total = 1.f;
  }

  // Duplicate the final real head to fill an incomplete tile.
  const int q_head = min(head_base + frag, p.heads - 1);
  const half* query = p.q + row * p.q_row_stride + q_head * p.q_head_stride;
  const int pv_col = (warp * 4 + quad) * 8;
  float values[2][8] = {};
#pragma unroll 1
  for (int start = 0; start < length; start += Tile) {
    const int live = min(Tile, length - start);
    // The selectors have already applied causal/window masks. Each live
    // entry here is a physical slot and the rest of the tile is padding.
#pragma unroll 1
    for (int j = warp; j < Tile; j += 8) {
      const char* src = nullptr;
      if (j < live) {
        const int slot = p.slots[row * p.slots_stride + start + j];
        const int block = slot / p.page;
        src = p.cache + block * p.cache_page_bytes +
              (slot - block * p.page) * p.cache_row_bytes;
      }
      if constexpr (IsFp8) {
        uint4 codes = {};
        uint32_t rope = 0;
        float scale = 0.f;
        if (src) {
          codes = __ldg(reinterpret_cast<const uint4*>(src) + lane);
          scale = __ldg(reinterpret_cast<const float*>(src + kKvRank) + (lane >> 3));
          rope = __ldg(reinterpret_cast<const uint32_t*>(src + kKvRank + 16) + lane);
        }
        const uint32_t words[] = {codes.x, codes.y, codes.z, codes.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          *reinterpret_cast<uint2*>(&keys[j][lane * 16 + i * 4]) =
              decode_legacy_fp8x4<true>(words[i], scale);
        }
        reinterpret_cast<uint32_t*>(keys[j] + kKvRank)[lane] = rope;
      } else {
        auto* dst = reinterpret_cast<uint4*>(keys[j]);
        const auto* packed = reinterpret_cast<const uint4*>(src);
        dst[lane] = src ? __ldg(packed + lane) : uint4{};
        dst[lane + 32] = src ? __ldg(packed + lane + 32) : uint4{};
        if (lane < 8) {
          reinterpret_cast<uint4*>(keys[j] + kKvRank)[lane] =
              src ? __ldg(reinterpret_cast<const uint4*>(src + 2 * kKvRank) + lane) : uint4{};
        }
      }
    }
    __syncthreads();

    const half* key = keys[(warp / 4) * 8 + frag];
    const int split_dim = (warp % 4 * 4 + quad) * 4;
    float dots[8] = {};
    __syncwarp();
#pragma unroll 1
    for (int offset = 0; offset < kKvRank / 4; offset += 16) {
      const int d = offset + (warp % 4) * (kKvRank / 4) + quad * 4;
      const uint2 a = __ldg(reinterpret_cast<const uint2*>(query + d));
      const uint2 b = *reinterpret_cast<const uint2*>(key + d);
      if constexpr (PromoteQk) {
        float step[8] = {};
        HY4_F7_OUT_MMA(step, a.x, a.y, b.x, b.y);
#pragma unroll
        for (int i = 0; i < 8; ++i) dots[i] = __fadd_rn(dots[i], step[i]);
      } else {
        HY4_F7_OUT_MMA(dots, a.x, a.y, b.x, b.y);
      }
    }
    const int rope_dim = kKvRank + split_dim;
    const uint2 qr = __ldg(reinterpret_cast<const uint2*>(query + rope_dim));
    const uint2 kr = *reinterpret_cast<const uint2*>(key + rope_dim);
    float rope_dot[8] = {};
    __syncwarp();
    HY4_F7_OUT_MMA(rope_dot, qr.x, qr.y, kr.x, kr.y);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float latent_sum = dots[i] + __shfl_xor_sync(0xffffffffu, dots[i], 4);
      float rope_sum = rope_dot[i] + __shfl_xor_sync(0xffffffffu, rope_dot[i], 4);
      latent_sum += __shfl_xor_sync(0xffffffffu, latent_sum, 8);
      rope_sum += __shfl_xor_sync(0xffffffffu, rope_sum, 8);
      if (quad == 0) {
        const int h = out_acc_row(i, lane);
        const int n = (warp / 4) * 8 + out_acc_col(i, lane);
        scores[warp % 4][h][n] = latent_sum;
        rope_scores[warp % 4][h][n] = rope_sum;
      }
    }
    __syncthreads();

    if (h < 8) {
      const float latent_sum = (scores[0][h][n] + scores[1][h][n]) +
                               (scores[2][h][n] + scores[3][h][n]);
      const float rope_sum = (rope_scores[0][h][n] + rope_scores[1][h][n]) +
                             (rope_scores[2][h][n] + rope_scores[3][h][n]);
      const float score = n < live ? (latent_sum + rope_sum) * p.scale : -INFINITY;
      float tile_max = score;
      __syncwarp();
#pragma unroll
      for (int offset = Tile / 2; offset > 0; offset /= 2) {
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, offset, Tile));
      }
      const float next_max = fmaxf(maximum, tile_max);
      const float alpha = prefill_exp(maximum - next_max);
      const float prob = prefill_exp(score - next_max);
      maximum = next_max;
      probs[h][n] = __float2half_rn(prob);
      float sum = prob;
      __syncwarp();
#pragma unroll
      for (int offset = Tile / 2; offset > 0; offset /= 2) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset, Tile);
      }
      total = total * alpha + sum;
      if (n == 0) rescale[h] = alpha;
    }
    __syncthreads();

#pragma unroll
    for (int part = 0; part < 2; ++part) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        values[part][i] = __fmul_rn(values[part][i], rescale[out_acc_row(i, lane)]);
      }
    }
    float tile_values[2][8] = {};
    float (&pv)[2][8] = PromotePv ? tile_values : values;
    __syncwarp();
#pragma unroll 1
    for (int j = 0; j < Tile; j += 4) {
      const uint2 a = *reinterpret_cast<const uint2*>(&probs[frag][j]);
#pragma unroll
      for (int part = 0; part < 2; ++part) {
        const int d = part * 256 + pv_col + frag;
        const uint32_t b0 = pack_halves(keys[j][d], keys[j + 1][d]);
        const uint32_t b1 = pack_halves(keys[j + 2][d], keys[j + 3][d]);
        if constexpr (PromotePvStep) {
          float step[8] = {};
          HY4_F7_OUT_MMA(step, a.x, a.y, b0, b1);
#pragma unroll
          for (int i = 0; i < 8; ++i) values[part][i] = __fadd_rn(values[part][i], step[i]);
        } else {
          HY4_F7_OUT_MMA(pv[part], a.x, a.y, b0, b1);
        }
      }
    }
    if constexpr (PromotePv) {
#pragma unroll
      for (int part = 0; part < 2; ++part) {
#pragma unroll
        for (int i = 0; i < 8; ++i) values[part][i] = __fadd_rn(values[part][i], tile_values[part][i]);
      }
    }
    __syncthreads();
  }

  if (h < 8 && n == 0) {
    inv_denominator[h] = length || p.sinks ? __frcp_rn(total) : 1.f;
    denominator[h] = length || p.sinks ? total : 1.f;
    if (p.lse && head < p.heads) {
      p.lse[static_cast<int64_t>(row) * p.heads + head] = maximum + logf(total);
    }
  }
  __syncthreads();
#pragma unroll
  for (int part = 0; part < 2; ++part) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int relative_head = out_acc_row(i, lane);
      const int output_head = head_base + relative_head;
      if (output_head < p.heads) {
        const int64_t index = (static_cast<int64_t>(row) * p.heads + output_head) *
                              kKvRank + part * 256 + pv_col + out_acc_col(i, lane);
        float v;
        if constexpr (DivFull) {
          asm("div.full.f32 %0, %1, %2;" : "=f"(v) : "f"(values[part][i]), "f"(denominator[relative_head]));
        } else {
          v = values[part][i] * inv_denominator[relative_head];
        }
        if (p.partial) p.partial[index] = v;
        else p.out[index] = __float2half_rn(v);
      }
    }
  }
}

void paged_prefill_sm70_meta(
    const torch::Tensor& q, const torch::Tensor& cache, const torch::Tensor& slots,
    const torch::Tensor& counts, const std::optional<torch::Tensor>& sinks,
    const torch::Tensor& out, const std::optional<torch::Tensor>& lse, double /*scale*/,
    int64_t mode) {
  TORCH_CHECK((mode >= 0 && mode <= 8) || mode == 12,
              "paged_prefill_sm70 mode must be 0-8 or 12");
  const int rows = q.size(0);
  const int heads = q.size(1);
  const bool fp8 = cache.scalar_type() == at::kByte;
  TORCH_CHECK(q.scalar_type() == at::kHalf && q.size(2) == kKvADim && q.stride(2) == 1);
  TORCH_CHECK(fp8 ? cache.size(2) == 656 :
              cache.scalar_type() == at::kHalf && cache.size(2) == kKvADim);
  TORCH_CHECK(cache.stride(2) == 1 && slots.stride(1) == 1 && counts.is_contiguous());
  TORCH_CHECK(slots.size(0) == rows && counts.numel() == rows);
  TORCH_CHECK(out.is_contiguous() && out.sizes() == at::IntArrayRef({rows, heads, kKvRank}));
  TORCH_CHECK(lse.has_value() != sinks.has_value());
  TORCH_CHECK(!lse || (lse->is_contiguous() && lse->sizes() == at::IntArrayRef({rows, heads})));
  TORCH_CHECK(!sinks || (sinks->is_contiguous() && sinks->numel() == heads));
}

void paged_prefill_sm70(
    const torch::Tensor& q, const torch::Tensor& cache, const torch::Tensor& slots,
    const torch::Tensor& counts, const std::optional<torch::Tensor>& sinks,
    const torch::Tensor& out, const std::optional<torch::Tensor>& lse, double scale,
    int64_t mode) {
  paged_prefill_sm70_meta(q, cache, slots, counts, sinks, out, lse, scale, mode);
  const int rows = q.size(0);
  const int heads = q.size(1);
  const bool fp8 = cache.scalar_type() == at::kByte;
  if (!rows || !heads) return;
  const c10::cuda::CUDAGuard guard(q.device());
  const PrefillParams p{
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      static_cast<const char*>(cache.const_data_ptr()), slots.data_ptr<int32_t>(),
      counts.data_ptr<int32_t>(), sinks ? sinks->data_ptr<float>() : nullptr,
      lse ? nullptr : reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      lse ? out.data_ptr<float>() : nullptr, lse ? lse->data_ptr<float>() : nullptr,
      q.stride(0), q.stride(1), cache.stride(0) * static_cast<int64_t>(cache.element_size()),
      cache.stride(1) * static_cast<int64_t>(cache.element_size()), slots.stride(0),
      heads, static_cast<int>(cache.size(1)), static_cast<float>(scale)};
  const dim3 grid(rows, (heads + 7) / 8);
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto fp8_tag, auto mode_tag) {
    hy4_paged_prefill<decltype(fp8_tag)::value, decltype(mode_tag)::value>
        <<<grid, 256, 0, stream>>>(p);
  };
  auto by_mode = [&](auto fp8_tag) {
    switch (mode) {
      case 0: launch(fp8_tag, std::integral_constant<int, 0>{}); break;
      case 1: launch(fp8_tag, std::integral_constant<int, 1>{}); break;
      case 2: launch(fp8_tag, std::integral_constant<int, 2>{}); break;
      case 3: launch(fp8_tag, std::integral_constant<int, 3>{}); break;
      case 4: launch(fp8_tag, std::integral_constant<int, 4>{}); break;
      case 5: launch(fp8_tag, std::integral_constant<int, 5>{}); break;
      case 6: launch(fp8_tag, std::integral_constant<int, 6>{}); break;
      case 7: launch(fp8_tag, std::integral_constant<int, 7>{}); break;
      case 8: launch(fp8_tag, std::integral_constant<int, 8>{}); break;
      default: launch(fp8_tag, std::integral_constant<int, 12>{}); break;
    }
  };
  if (fp8) by_mode(std::true_type{});
  else by_mode(std::false_type{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// sm_70 Rule 1: o_latent was written in Phase 0 before grid.sync() -> MUST use __ldcg.
// sm_70 Rule 2: red MUST be volatile float* so nvcc 12.8 preserves LDS after __syncthreads().
template <int RB>
__device__ __forceinline__ void mma_tile_16_k512_cg(
    const half* __restrict__ X,
    const half* __restrict__ W_tile,
    int m,
    volatile float* red) {
  constexpr int kRows = RB * 8;
  constexpr int K = kKvRank;  // 512
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int wn = warp & 1;
  const int wk = warp >> 1;
  const int qp = (lane >> 2) & 3;
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0);

  // K=512 split evenly across kWarpsK=4 warps -> 128 elems/warp = 2 full 64-elem steps.
  const int k_begin = wk * 128;
  const int k_end = k_begin + 128;

  const half* wrow = W_tile + static_cast<size_t>(wn * 8 + r8) * K;
  const half* xrow[RB];
  bool row_ok[RB];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    const int r = rb * 8 + r8;
    row_ok[rb] = r < m;
    xrow[rb] = X + static_cast<size_t>(row_ok[rb] ? r : 0) * K;
  }

  float acc[RB][8];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[rb][i] = 0.f;
  }

  uint4 b_lo = __ldcs(reinterpret_cast<const uint4*>(wrow + k_begin + qp * 8));
  uint4 b_hi = __ldcs(reinterpret_cast<const uint4*>(wrow + k_begin + 32 + qp * 8));

#pragma unroll 1
  for (int e = k_begin; e < k_end; e += 64) {
    const int lo_k = e + qp * 8;
    const int hi_k = lo_k + 32;
    uint4 n_lo = out_zero4();
    uint4 n_hi = out_zero4();
    if (e + 64 < k_end) {
      n_lo = __ldcs(reinterpret_cast<const uint4*>(wrow + lo_k + 64));
      n_hi = __ldcs(reinterpret_cast<const uint4*>(wrow + hi_k + 64));
    }
#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
      // sm_70 Rule 1: __ldcg bypasses non-coherent per-SM L1 cache after grid.sync().
      const uint4 a_lo =
          row_ok[rb] ? __ldcg(reinterpret_cast<const uint4*>(xrow[rb] + lo_k)) : out_zero4();
      const uint4 a_hi =
          row_ok[rb] ? __ldcg(reinterpret_cast<const uint4*>(xrow[rb] + hi_k)) : out_zero4();
      HY4_F7_OUT_MMA(acc[rb], a_lo.x, a_lo.y, b_lo.x, b_lo.y);
      HY4_F7_OUT_MMA(acc[rb], a_lo.z, a_lo.w, b_lo.z, b_lo.w);
      HY4_F7_OUT_MMA(acc[rb], a_hi.x, a_hi.y, b_hi.x, b_hi.y);
      HY4_F7_OUT_MMA(acc[rb], a_hi.z, a_hi.w, b_hi.z, b_hi.w);
    }
    b_lo = n_lo;
    b_hi = n_hi;
  }

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
        const int row = rb * 8 + out_acc_row(i, lane);
        const int col = wn * 8 + out_acc_col(i, lane);
        red[(wk * kRows + row) * kTileN + col] = acc[rb][i];
      }
    }
  }
  __syncthreads();
}

template <int RB, bool W8>
__global__ void __launch_bounds__(kThreads, (RB == 1 ? 4 : 2))
    hy4_attn_merge_v_gate_coop_kernel(const MergeVGateParams p) {
  constexpr int kRows = RB * 8;
  // sm_70 Rule 2: volatile shared memory for cross-warp reductions & split-K merge weights.
  __shared__ volatile float red[kWarpsK * kRows * kTileN];
  cg::grid_group grid = cg::this_grid();

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int gdim = gridDim.x;
  const int m = p.m;
  const int H = p.heads;

  // =========================================================================
  // PHASE 0: Merge K4 splits [0..ns) + learnable sink -> head-major o_latent[H, M, 512]
  //   Work items: M * H * 2 (each CTA reduces 256 of 512 latent dims = 1 elem/thread)
  // =========================================================================
  const int phase0_items = m * H * 2;
  for (int item = bid; item < phase0_items; item += gdim) {
    const int half_d = item & 1;
    const int rh = item >> 1;
    const int h = rh % H;
    const int row = rh / H;
    const int d = half_d * 256 + tid;

    const int length = p.counts[row];
    const bool is_short = p.short_splits && length <= kShortKeys;
    const int ns = is_short ? (length + kShortTile - 1) / kShortTile
                            : compute_num_splits(length, p.max_splits);
    half* dst = p.o_latent + (static_cast<size_t>(h) * m + row) * kKvRank;

    if (ns == 0) {
      dst[d] = __float2half_rn(0.f);
    } else {
      const int64_t base_slot = (static_cast<int64_t>(row) * H + h) * p.ws_splits;
      if (tid < ns) {
        red[tid] = p.state[(base_slot + tid) * 2 + 0];       // m[s]
        red[kShortSplits + tid] = p.state[(base_slot + tid) * 2 + 1];  // total[s]
      }
      __syncthreads();
      const float sink = __ldg(p.sinks + h);
      if (is_short) {
        if (tid == 0) {
          float final_m = sink;
          for (int s = 0; s < ns; ++s) final_m = fmaxf(final_m, red[s]);
          red[3 * kShortSplits] = final_m;
        }
        __syncthreads();
        const float final_m = red[3 * kShortSplits];
        if (tid < ns) red[2 * kShortSplits + tid] = expf(red[tid] - final_m);
        __syncthreads();
        if (tid == 0) {
          float denom = expf(sink - final_m);
          for (int s = 0; s < ns; ++s) {
            denom = __fadd_rn(denom, __fmul_rn(red[kShortSplits + s], red[2 * kShortSplits + s]));
          }
          red[3 * kShortSplits] = 1.0f / denom;
        }
      } else {
        if (tid == 0) {
          float final_m = sink;
          for (int s = 0; s < ns; ++s) {
            final_m = fmaxf(final_m, red[s]);
          }
          float denom = expf(sink - final_m);
          for (int s = 0; s < ns; ++s) {
            const float w = expf(red[s] - final_m);
            red[2 * kShortSplits + s] = w;
            denom = __fadd_rn(denom, __fmul_rn(red[kShortSplits + s], w));
          }
          red[3 * kShortSplits] = 1.0f / denom;
        }
      }
      __syncthreads();

      const float inv_denom = red[3 * kShortSplits];
      const float* acc_base = p.acc + base_slot * kKvRank + d;
      float num = 0.f;
      for (int s = 0; s < ns; ++s) {
        num = __fadd_rn(num, __fmul_rn(acc_base[static_cast<size_t>(s) * kKvRank], red[2 * kShortSplits + s]));
      }
      dst[d] = __float2half_rn(num * inv_denom);
    }
    __syncthreads();
  }
  grid.sync();

  // =========================================================================
  // PHASE 1: Absorbed W_uv BMM (H x 256 x 512) + elementwise * gate_sig -> out[M, H*256]
  //   256 tiles (SWA H=16) or 128 tiles (Sparse H=8) -> fits in 1 wave at RB=1 (320 CTAs).
  // =========================================================================
  const int phase1_tiles = H * 16;
  const int out_dim = H * kVHead;
  for (int tile = bid; tile < phase1_tiles; tile += gdim) {
    const int head = tile >> 4;
    const int n0 = (tile & 15) * kTileN;
    const half* x_head = p.o_latent + static_cast<size_t>(head) * m * kKvRank;
    if constexpr (W8) {
      const auto* w = p.uv8 + head * hy4_w8::matrix_bytes(kVHead, kKvRank);
      hy4_w8::slice16<RB, true, kWarpsK>(
          x_head, w, m, kVHead, kKvRank, n0, 0, kKvRank, red);
    } else {
      const half* w_ptr = p.w_uv + static_cast<size_t>(head) * p.w_uv_head_stride
                          + static_cast<size_t>(n0) * kKvRank;
      mma_tile_16_k512_cg<RB>(x_head, w_ptr, m, red);
    }

    for (int idx = tid; idx < m * kTileN; idx += kThreads) {
      const int row = idx >> 4;
      const int c = idx & 15;
      float v = red[row * kTileN + c];
#pragma unroll
      for (int q = 1; q < kWarpsK; ++q) {
        v = __fadd_rn(v, red[(q * kRows + row) * kTileN + c]);
      }
      const half v_h = __float2half_rn(v);
      const size_t out_off = static_cast<size_t>(row) * out_dim + head * kVHead + n0 + c;
      const half g_h = __ldg(p.gate_sig + out_off);
      p.out[out_off] = __hmul(v_h, g_h);
    }
    __syncthreads();
  }
}

template <int RB, bool W8>
void launch_coop_merge_v_gate(MergeVGateParams p, cudaStream_t stream) {
  static int cached_blocks_by_dev[16] = {0};
  int dev = 0;
  cudaGetDevice(&dev);
  int blocks = (dev >= 0 && dev < 16) ? cached_blocks_by_dev[dev] : 0;
  if (blocks == 0) {
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    int max_bpsm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_bpsm, (const void*)hy4_attn_merge_v_gate_coop_kernel<RB, W8>, kThreads, 0);
    TORCH_CHECK(err == cudaSuccess && max_bpsm > 0,
                "cudaOccupancyMaxActiveBlocksPerMultiprocessor(K5) failed: ",
                cudaGetErrorString(err));
    blocks = max_bpsm * sm_count;
    if (dev >= 0 && dev < 16) cached_blocks_by_dev[dev] = blocks;
  }
  void* args[] = {&p};
  cudaError_t err = cudaLaunchCooperativeKernel(
      (const void*)hy4_attn_merge_v_gate_coop_kernel<RB, W8>,
      dim3(blocks),
      dim3(kThreads),
      args,
      0,
      stream);
  TORCH_CHECK(err == cudaSuccess,
              "cudaLaunchCooperativeKernel(K5) failed (blocks=", blocks, "): ",
              cudaGetErrorString(err));
}


// Four 32-lane K warps reproduce the legacy W8 reduction order for one
// eight-column tile. The short and legacy readers can both feed this merge.
constexpr int kNarrowMergeThreads = 128;
constexpr int kNarrowMergeCols = 8;

__global__ void __launch_bounds__(kNarrowMergeThreads, 6)
    hy4_attn_merge_v_gate_w8_narrow_kernel(const MergeVGateParams p) {
  __shared__ volatile float weights[kShortSplits];
  __shared__ volatile float totals[kShortSplits];
  __shared__ volatile float inv_denom;
  __shared__ volatile float red[kWarpsK * 8 * kNarrowMergeCols];
  const auto grid = cg::this_grid();
  const int tid = threadIdx.x;
  const int h = p.heads;
  constexpr int blocks_per_latent = kKvRank / kNarrowMergeThreads;

  for (int item = blockIdx.x; item < p.m * h * blocks_per_latent; item += gridDim.x) {
    const int rh = item / blocks_per_latent;
    const int head = rh % h;
    const int row = rh / h;
    const int d = (item % blocks_per_latent) * kNarrowMergeThreads + tid;
    const int length = p.counts[row];
    const int ns = p.short_splits && length <= kShortKeys
        ? (length + kShortTile - 1) / kShortTile
        : compute_num_splits(length, p.max_splits);
    half* dst = p.o_latent + (static_cast<size_t>(head) * p.m + row) * kKvRank;
    if (ns == 0) {
      dst[d] = __float2half_rn(0.0f);
      continue;
    }

    const int64_t base_slot = static_cast<int64_t>(rh) * p.ws_splits;
    if (tid < 32) {
      const float sink = __ldg(p.sinks + head);
      float first = tid < ns ? p.state[2 * (base_slot + tid)] : sink;
      float second = tid + 32 < ns ? p.state[2 * (base_slot + tid + 32)] : sink;
      float maximum = fmaxf(sink, fmaxf(first, second));
      for (int offset = 16; offset > 0; offset /= 2) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
      }
      maximum = __shfl_sync(0xffffffffu, maximum, 0);
      if (tid < ns) {
        weights[tid] = expf(first - maximum);
        totals[tid] = p.state[2 * (base_slot + tid) + 1];
      }
      if (tid + 32 < ns) {
        weights[tid + 32] = expf(second - maximum);
        totals[tid + 32] = p.state[2 * (base_slot + tid + 32) + 1];
      }
      __syncwarp();
      if (tid == 0) {
        float denom = expf(sink - maximum);
        for (int s = 0; s < ns; ++s) {
          denom = __fadd_rn(denom, __fmul_rn(totals[s], weights[s]));
        }
        inv_denom = 1.0f / denom;
      }
    }
    __syncthreads();

    float num = 0.0f;
    const float* acc = p.acc + base_slot * kKvRank + d;
    for (int s = 0; s < ns; ++s) {
      num = __fadd_rn(num, __fmul_rn(acc[s * kKvRank], weights[s]));
    }
    dst[d] = __float2half_rn(num * inv_denom);
    // Scratch can only be reused once every thread has consumed it.
    if (item + gridDim.x < p.m * h * blocks_per_latent) __syncthreads();
  }
  grid.sync();

  constexpr int tiles_per_head = kVHead / kNarrowMergeCols;
  for (int tile = blockIdx.x; tile < h * tiles_per_head; tile += gridDim.x) {
    const int head = tile / tiles_per_head;
    const int n0 = (tile % tiles_per_head) * kNarrowMergeCols;
    const auto* w = p.uv8 + head * hy4_w8::matrix_bytes(kVHead, kKvRank);
    const half* x = p.o_latent + static_cast<size_t>(head) * p.m * kKvRank;
    hy4_w8::slice16<1, true, kWarpsK, 1>(
        x, w, p.m, kVHead, kKvRank, n0, 0, kKvRank, red);
    if (tid < p.m * kNarrowMergeCols) {
      const int row = tid / kNarrowMergeCols;
      const int col = tid % kNarrowMergeCols;
      float v = red[tid];
#pragma unroll
      for (int warp = 1; warp < kWarpsK; ++warp) {
        v = __fadd_rn(v, red[warp * 8 * kNarrowMergeCols + tid]);
      }
      const int out_off = (row * h + head) * kVHead + n0 + col;
      p.out[out_off] = __hmul(__float2half_rn(v), __ldg(p.gate_sig + out_off));
    }
    if (tile + gridDim.x < h * tiles_per_head) __syncthreads();
  }
}

void launch_w8_narrow_merge(MergeVGateParams p, cudaStream_t stream) {
  static int cached_blocks_by_dev[16] = {0};
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  int blocks = dev < 16 ? cached_blocks_by_dev[dev] : 0;
  if (blocks == 0) {
    int sms = 0;
    int blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, hy4_attn_merge_v_gate_w8_narrow_kernel, kNarrowMergeThreads, 0));
    blocks = blocks_per_sm * sms;
    if (dev < 16) cached_blocks_by_dev[dev] = blocks;
  }
  const int items = p.heads * std::max(p.m * (kKvRank / kNarrowMergeThreads),
                                      kVHead / kNarrowMergeCols);
  void* args[] = {&p};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      (const void*)hy4_attn_merge_v_gate_w8_narrow_kernel,
      dim3(std::min(items, blocks)), dim3(kNarrowMergeThreads),
      args, 0, stream));
}

void attn_merge_v_gate_sm70(
    const at::Tensor& state,
    const at::Tensor& acc,
    const at::Tensor& counts,
    const at::Tensor& sinks,
    const at::Tensor& w_uv,
    const at::Tensor& gate_sig,
    at::Tensor& o_latent_scratch,
    at::Tensor& out,
    int64_t max_splits,
    bool short_splits,
    bool narrow_w8) {
  TORCH_CHECK(state.is_cuda() && w_uv.is_cuda() && gate_sig.is_cuda(), "CUDA tensors required");
  const int m = static_cast<int>(gate_sig.size(0));
  const int heads = static_cast<int>(w_uv.size(0));
  const int ws_splits = static_cast<int>(state.size(2));
  TORCH_CHECK(m >= 1 && m <= 24, "m must be in [1, 24], got ", m);
  TORCH_CHECK(heads >= 1 && heads <= 32, "heads must be in [1, 32], got ", heads);
  const bool w8 = w_uv.scalar_type() == at::kByte;
  TORCH_CHECK(w_uv.dim() == 3 && w_uv.size(1) == kVHead);
  if (w8) {
    TORCH_CHECK(w_uv.is_contiguous() && w_uv.device() == state.device() &&
                w_uv.numel() == heads * hy4_w8::matrix_bytes(kVHead, kKvRank));
  } else {
    TORCH_CHECK(w_uv.size(2) == kKvRank &&
                w_uv.stride(2) == 1 && w_uv.stride(1) == kKvRank,
                "w_uv inner [256, 512] must be row-contiguous");
  }
  TORCH_CHECK(gate_sig.is_contiguous() && gate_sig.size(1) == heads * kVHead);
  TORCH_CHECK(out.is_contiguous() && out.sizes() == gate_sig.sizes());
  TORCH_CHECK(o_latent_scratch.is_contiguous() &&
              o_latent_scratch.numel() >= static_cast<int64_t>(heads) * m * kKvRank);
  TORCH_CHECK(max_splits >= 1 && max_splits <= ws_splits && max_splits <= kMaxSplitsCap);

  const c10::cuda::CUDAGuard guard(gate_sig.device());
  MergeVGateParams p{};
  p.state = state.const_data_ptr<float>();
  p.acc = acc.const_data_ptr<float>();
  p.counts = counts.const_data_ptr<int32_t>();
  p.sinks = sinks.const_data_ptr<float>();
  if (w8) {
    p.uv8 = w_uv.const_data_ptr<uint8_t>();
  } else {
    p.w_uv = reinterpret_cast<const half*>(w_uv.const_data_ptr<at::Half>());
  }
  p.gate_sig = reinterpret_cast<const half*>(gate_sig.const_data_ptr<at::Half>());
  p.o_latent = reinterpret_cast<half*>(o_latent_scratch.mutable_data_ptr<at::Half>());
  p.out = reinterpret_cast<half*>(out.mutable_data_ptr<at::Half>());
  p.w_uv_head_stride = w_uv.stride(0);
  p.m = m;
  p.heads = heads;
  p.ws_splits = ws_splits;
  p.max_splits = static_cast<int>(max_splits);
  p.short_splits = short_splits;
  TORCH_CHECK(!short_splits || (m == 1 && ws_splits >= kShortSplits));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (w8 && m <= 8 && narrow_w8) {
    launch_w8_narrow_merge(p, stream);
    return;
  }
  const int rb = (m + 7) / 8;
  const auto launch = [&](auto packed) {
    constexpr bool w8 = decltype(packed)::value;
    if (rb == 1) {
      launch_coop_merge_v_gate<1, w8>(p, stream);
    } else if (rb == 2) {
      launch_coop_merge_v_gate<2, w8>(p, stream);
    } else {
      launch_coop_merge_v_gate<3, w8>(p, stream);
    }
  };
  if (w8) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
}

void attn_merge_v_gate_sm70_meta(
    const at::Tensor& state,
    const at::Tensor& acc,
    const at::Tensor& counts,
    const at::Tensor& sinks,
    const at::Tensor& w_uv,
    const at::Tensor& gate_sig,
    at::Tensor& o_latent_scratch,
    at::Tensor& out,
    int64_t max_splits,
    bool short_splits,
    bool narrow_w8) {
  const int64_t m = gate_sig.size(0);
  const int64_t heads = w_uv.size(0);
  TORCH_CHECK(m >= 1 && m <= 24);
  TORCH_CHECK(heads >= 1 && heads <= 32);
  TORCH_CHECK(out.sizes() == at::IntArrayRef({m, heads * kVHead}));
  TORCH_CHECK(max_splits >= 1 && max_splits <= kMaxSplitsCap);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(hy4_f7_attn, m) {
  m.def(
      "paged_prefill_sm70(Tensor q, Tensor cache, Tensor slots, Tensor counts, "
      "Tensor? sinks, Tensor(a!) out, Tensor(b!)? lse, float scale, int mode=0) -> ()");
  m.def(
      "attn_splitk_sm70("
      "Tensor q, Tensor kv_cache, Tensor? draft_kv, Tensor indices, Tensor counts,"
      "Tensor(a!) state, Tensor(b!) acc, float scale, int max_splits, int query_width, bool short_splits=False, bool prmt_fp8=False, bool swa_tensor=False) -> ()");
  m.def(
      "attn_merge_v_gate_sm70("
      "Tensor state, Tensor acc, Tensor counts, Tensor sinks, Tensor w_uv, Tensor gate_sig,"
      "Tensor(a!) o_latent_scratch, Tensor(b!) out, int max_splits, bool short_splits=False, bool narrow_w8=False) -> ()");
}

TORCH_LIBRARY_IMPL(hy4_f7_attn, CUDA, m) {
  m.impl("paged_prefill_sm70", &paged_prefill_sm70);
  m.impl("attn_splitk_sm70", &attn_splitk_sm70);
  m.impl("attn_merge_v_gate_sm70", &attn_merge_v_gate_sm70);
}

TORCH_LIBRARY_IMPL(hy4_f7_attn, Meta, m) {
  m.impl("paged_prefill_sm70", &paged_prefill_sm70_meta);
  m.impl("attn_splitk_sm70", &attn_splitk_sm70_meta);
  m.impl("attn_merge_v_gate_sm70", &attn_merge_v_gate_sm70_meta);
}
