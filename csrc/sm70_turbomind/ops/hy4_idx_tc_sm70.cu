// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// IDXDEC (HY4_IDX_TC): SM70 IndexPool decode scoring and history selection.
//   idx_tc_scores: logits[row, g] = sum_h w[row, h] * relu(q[row, h] . key[g])
//     with m8n8k4 HMMA (FP16 inputs, FP32 accumulate). One thread owns one
//     FP16 key row (256 B in registers), so each CTA streams 128 keys once and
//     reuses them for every query row of a verify request (query_width rows).
//   idx_tc_select: per-row radix select of the top `history` live logits,
//     written as ascending group IDs (stream compaction keeps index order, so
//     no sort is needed). Ties at the threshold take the lowest group IDs.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>

namespace {

constexpr int kHeads = 16;
constexpr int kDim = 128;
constexpr int kQPitch = kDim + 8;  // halves; 16-byte skew keeps LDS.128 conflict-free
constexpr int kScoreThreads = 128;
constexpr int kGroupsPerCta = 128;
constexpr int kMaxWidth = 8;
constexpr int kSelectThreads = 1024;
constexpr int kSelectBatch = 8;

#define IDX_TC_MMA(C, A0, A1, B0, B1)                                \
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
  return (i & 1) | (lane & 2) | (i & 4);
}

template <typename SlotT>
__device__ __forceinline__ int live_groups(const int* lengths, const SlotT* slots, int row,
                                           int ratio, int group_topk, int world, int rank) {
  const int length = lengths[row];
  const bool active = slots[row] >= 0 && length > group_topk * ratio;
  int upper = active ? (length - 1) / ratio : 0;
  if (world > 1) upper = (upper + world - 1 - rank) / world;
  return upper;
}

struct ScoreParams {
  const half* q;
  const float* w;
  const half* pool;
  const int* table;
  const int* lengths;
  const void* slots;
  float* logits;
  int64_t stride_qt, stride_qh, stride_pool_p, stride_pool_g, stride_table, stride_logits;
  int groups, ratio, group_topk, page, pages_per_block, world, rank, width;
  bool fill;
};

template <typename SlotT>
__global__ __launch_bounds__(kScoreThreads, 4) void idx_tc_scores_kernel(ScoreParams p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  half* qs = reinterpret_cast<half*>(smem_raw);  // [width][16][kQPitch]
  __shared__ float ws[kMaxWidth][kHeads];
  __shared__ int uppers[kMaxWidth];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int width = p.width;
  const int request = blockIdx.x;
  const int row0 = request * width;
  const int start = blockIdx.y * kGroupsPerCta;
  const SlotT* slots = static_cast<const SlotT*>(p.slots);

  if (tid < width) {
    uppers[tid] = live_groups(p.lengths, slots, row0 + tid, p.ratio, p.group_topk, p.world, p.rank);
  }
  __syncthreads();
  int max_upper = 0;
  for (int r = 0; r < width; ++r) max_upper = max(max_upper, uppers[r]);

  if (start >= max_upper) {
    if (p.fill && (p.world == 1 || start < max_upper)) {
      for (int i = tid; i < width * kGroupsPerCta; i += kScoreThreads) {
        const int r = i / kGroupsPerCta;
        const int g = start + i % kGroupsPerCta;
        if (g < p.groups) p.logits[(row0 + r) * p.stride_logits + g] = -INFINITY;
      }
    }
    return;
  }

  // Stage Q (width x 16 x 128 FP16) and head weights.
  for (int i = tid; i < width * kHeads * (kDim / 8); i += kScoreThreads) {
    const int r = i / (kHeads * (kDim / 8));
    const int h = (i / (kDim / 8)) % kHeads;
    const int c = i % (kDim / 8);
    const uint4 v = __ldg(reinterpret_cast<const uint4*>(
        p.q + (row0 + r) * p.stride_qt + h * p.stride_qh + c * 8));
    *reinterpret_cast<uint4*>(qs + (r * kHeads + h) * kQPitch + c * 8) = v;
  }
  for (int i = tid; i < width * kHeads; i += kScoreThreads) {
    ws[i / kHeads][i % kHeads] = p.w[(row0 + i / kHeads) * kHeads + i % kHeads];
  }

  // One key row per thread: B-fragment column = (lane & 3) + 4 * (lane >= 16).
  const int quad = (lane >> 2) & 3;
  const int frag = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int tile = start + warp * 32 + quad * 8;
  const int g = tile + frag;
  uint4 k[kDim / 8];
  if (g < max_upper) {
    const int logical = g / p.page;
    const int64_t page =
        static_cast<int64_t>(p.table[request * p.stride_table + logical / p.pages_per_block]) *
            p.pages_per_block +
        logical % p.pages_per_block;
    const uint4* src = reinterpret_cast<const uint4*>(
        p.pool + page * p.stride_pool_p + (g % p.page) * p.stride_pool_g);
#pragma unroll
    for (int j = 0; j < kDim / 8; ++j) k[j] = __ldg(src + j);
  } else {
#pragma unroll
    for (int j = 0; j < kDim / 8; ++j) k[j] = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  __syncwarp();

#pragma unroll 1
  for (int r = 0; r < width; ++r) {
    float acc[2][8];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[mt][i] = 0.f;
    const half* qrow = qs + r * kHeads * kQPitch + frag * kQPitch;
#pragma unroll
    for (int j = 0; j < kDim / 8; ++j) {
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const uint4 a = *reinterpret_cast<const uint4*>(qrow + mt * 8 * kQPitch + j * 8);
        IDX_TC_MMA(acc[mt], a.x, a.y, k[j].x, k[j].y);
        IDX_TC_MMA(acc[mt], a.z, a.w, k[j].z, k[j].w);
      }
    }
    // Weighted ReLU head sum. Column c of the quad's 8x8 tile is spread over
    // rows (i & 2), lane bit 0 and lane bit 4 for each 8-head tile.
    float part[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int i = (c & 1) | ((c & 2) << 1);
      float s = 0.f;
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        s += fmaxf(acc[mt][i], 0.f) * ws[r][mt * 8 + acc_row(i, lane)];
        s += fmaxf(acc[mt][i | 2], 0.f) * ws[r][mt * 8 + acc_row(i | 2, lane)];
      }
      s += __shfl_xor_sync(0xffffffffu, s, 1);
      s += __shfl_xor_sync(0xffffffffu, s, 16);
      part[c] = s;
    }
    if ((lane & 17) == 0) {
      const int upper = uppers[r];
      float* out = p.logits + (row0 + r) * p.stride_logits;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int i = (c & 1) | ((c & 2) << 1);
        const int gg = tile + acc_col(i, lane);
        if (gg < p.groups && (p.fill || gg < upper)) out[gg] = gg < upper ? part[c] : -INFINITY;
      }
    }
  }
}

__device__ __forceinline__ uint32_t order_key(float v) {
  const uint32_t u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

struct SelectParams {
  const float* logits;
  const int* lengths;
  const void* slots;
  int64_t* out;
  int64_t stride_logits, stride_out;
  int ratio, group_topk, history, world, rank;
};

// Block-wide exclusive scan of one int per thread (1024 threads).
__device__ __forceinline__ int block_exclusive_scan(int v, int* warp_sums) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) warp_sums[warp] = x;
  __syncthreads();
  if (warp == 0) {
    int t = warp_sums[lane];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, t, o);
      if (lane >= o) t += y;
    }
    warp_sums[lane] = t;
  }
  __syncthreads();
  const int before = (warp ? warp_sums[warp - 1] : 0) + x - v;
  __syncthreads();
  return before;
}

template <typename SlotT>
__global__ __launch_bounds__(kSelectThreads, 1) void idx_tc_select_kernel(SelectParams p) {
  __shared__ int hist[256];
  __shared__ int warp_sums[32];
  __shared__ int found_bin, found_above;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int row = blockIdx.x;
  const int k = p.history;
  const SlotT* slots = static_cast<const SlotT*>(p.slots);
  const int upper = live_groups(p.lengths, slots, row, p.ratio, p.group_topk, p.world, p.rank);
  int64_t* out = p.out + row * p.stride_out;
  if (upper <= k) {
    // Inactive rows: decode_slots ignores the selection. Keep it defined.
    for (int i = tid; i < k; i += kSelectThreads) out[i] = i < upper ? i : 0;
    return;
  }
  const float* v = p.logits + row * p.stride_logits;

  uint32_t desired = 0, mask = 0;
  int remaining = k;
#pragma unroll 1
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (tid < 256) hist[tid] = 0;
    __syncthreads();
    // Batch the loads for memory-level parallelism, then aggregate equal bins
    // across the warp: scores share a few exponents, so plain atomics serialize.
#pragma unroll 1
    for (int base = 0; base < upper; base += kSelectThreads * kSelectBatch) {
      uint32_t keys[kSelectBatch];
#pragma unroll
      for (int u = 0; u < kSelectBatch; ++u) {
        const int i = base + u * kSelectThreads + tid;
        keys[u] = i < upper ? order_key(__ldg(v + i)) : 0u;
      }
#pragma unroll
      for (int u = 0; u < kSelectBatch; ++u) {
        const int i = base + u * kSelectThreads + tid;
        const bool hit = i < upper && (keys[u] & mask) == desired;
        const int bin = hit ? static_cast<int>((keys[u] >> shift) & 255u) : -1;
        const unsigned peers = __match_any_sync(0xffffffffu, bin);
        if (hit && lane == __ffs(peers) - 1) atomicAdd(&hist[bin], __popc(peers));
      }
    }
    __syncthreads();
    if (tid < 32) {
      // Lane L owns bins [8L, 8L + 8); count from the top bin down.
      int counts[8];
      int total = 0;
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        counts[b] = hist[tid * 8 + b];
        total += counts[b];
      }
      int suffix = total;  // inclusive sum over lanes >= tid
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_down_sync(0xffffffffu, suffix, o);
        if (tid + o < 32) suffix += y;
      }
      int above = suffix - total;
#pragma unroll
      for (int b = 7; b >= 0; --b) {
        if (above < remaining && above + counts[b] >= remaining) {
          found_bin = tid * 8 + b;
          found_above = above;
        }
        above += counts[b];
      }
    }
    __syncthreads();
    desired |= static_cast<uint32_t>(found_bin) << shift;
    mask |= 255u << shift;
    remaining -= found_above;
    __syncthreads();
  }

  // `desired` is the k-th largest key; take `remaining` of its ties.
  // Contiguous per-thread chunks (multiples of 4, so float4 loads stay aligned)
  // keep the output in index order with one block scan per counter.
  const int chunk = ((upper + kSelectThreads - 1) / kSelectThreads + 3) & ~3;
  const int begin = min(tid * chunk, upper);
  const int end = min(begin + chunk, upper);
  int gt = 0, eq = 0;
#pragma unroll 4
  for (int i = begin; i < end; i += 4) {
    const float4 x = __ldg(reinterpret_cast<const float4*>(v + i));
    const float xs[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const uint32_t key = order_key(xs[u]);
      const bool live = i + u < end;
      gt += live && key > desired;
      eq += live && key == desired;
    }
  }
  const int gt_before = block_exclusive_scan(gt, warp_sums);
  int eq_seen = block_exclusive_scan(eq, warp_sums);
  int pos = gt_before + min(eq_seen, remaining);
  if (gt == 0 && (eq == 0 || eq_seen >= remaining)) return;
  for (int i = begin; i < end; i += 4) {
    const float4 x = __ldg(reinterpret_cast<const float4*>(v + i));
    const float xs[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const uint32_t key = order_key(xs[u]);
      if (i + u >= end) break;
      if (key > desired) {
        out[pos++] = i + u;
      } else if (key == desired) {
        if (eq_seen < remaining) out[pos++] = i + u;
        ++eq_seen;
      }
    }
  }
}

void check_rows(const at::Tensor& lengths, const at::Tensor& slots, int64_t rows) {
  // Graph-padded rows may lie past the live metadata, as in the Triton path;
  // padded rows read negative slots and never touch the pool.
  (void)rows;
  TORCH_CHECK(lengths.is_cuda() && lengths.scalar_type() == at::kInt);
  TORCH_CHECK(slots.is_cuda() &&
              (slots.scalar_type() == at::kLong || slots.scalar_type() == at::kInt));
}

void idx_tc_scores(const at::Tensor& q, const at::Tensor& w, const at::Tensor& pool,
                   const at::Tensor& table, at::Tensor& logits, const at::Tensor& lengths,
                   const at::Tensor& slots, int64_t ratio, int64_t group_topk,
                   int64_t pages_per_block, int64_t world, int64_t rank, int64_t width,
                   bool fill) {
  const int64_t rows = q.size(0);
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf && q.dim() == 3 &&
              q.size(1) == kHeads && q.size(2) == kDim && q.stride(2) == 1 &&
              q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0 &&
              reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0);
  TORCH_CHECK(w.scalar_type() == at::kFloat && w.is_contiguous() && w.size(0) == rows &&
              w.size(1) == kHeads);
  TORCH_CHECK(pool.scalar_type() == at::kHalf && pool.dim() == 3 && pool.size(2) == kDim &&
              pool.stride(2) == 1 && pool.stride(1) % 8 == 0 && pool.stride(0) % 8 == 0 &&
              reinterpret_cast<uintptr_t>(pool.data_ptr()) % 16 == 0);
  TORCH_CHECK(table.scalar_type() == at::kInt && table.dim() == 2 && table.stride(1) == 1);
  TORCH_CHECK(logits.scalar_type() == at::kFloat && logits.dim() == 2 && logits.stride(1) == 1 &&
              logits.size(0) >= rows);
  TORCH_CHECK(width >= 1 && width <= kMaxWidth && rows % width == 0);
  check_rows(lengths, slots, rows);
  if (rows == 0) return;

  ScoreParams p{};
  p.q = reinterpret_cast<const half*>(q.const_data_ptr<at::Half>());
  p.w = w.const_data_ptr<float>();
  p.pool = reinterpret_cast<const half*>(pool.const_data_ptr<at::Half>());
  p.table = table.const_data_ptr<int>();
  p.lengths = lengths.const_data_ptr<int>();
  p.slots = slots.const_data_ptr();
  p.logits = logits.mutable_data_ptr<float>();
  p.stride_qt = q.stride(0);
  p.stride_qh = q.stride(1);
  p.stride_pool_p = pool.stride(0);
  p.stride_pool_g = pool.stride(1);
  p.stride_table = table.stride(0);
  p.stride_logits = logits.stride(0);
  p.groups = static_cast<int>(logits.size(1));
  p.ratio = static_cast<int>(ratio);
  p.group_topk = static_cast<int>(group_topk);
  p.page = static_cast<int>(pool.size(1));
  p.pages_per_block = static_cast<int>(pages_per_block);
  p.world = static_cast<int>(world);
  p.rank = static_cast<int>(rank);
  p.width = static_cast<int>(width);
  p.fill = fill;

  const c10::cuda::CUDAGuard guard(q.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(rows / width),
                  static_cast<unsigned>((p.groups + kGroupsPerCta - 1) / kGroupsPerCta));
  const size_t smem = static_cast<size_t>(width) * kHeads * kQPitch * sizeof(half);
  if (slots.scalar_type() == at::kLong) {
    idx_tc_scores_kernel<int64_t><<<grid, kScoreThreads, smem, stream>>>(p);
  } else {
    idx_tc_scores_kernel<int32_t><<<grid, kScoreThreads, smem, stream>>>(p);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void idx_tc_select(const at::Tensor& logits, const at::Tensor& lengths, const at::Tensor& slots,
                   at::Tensor& out, int64_t ratio, int64_t group_topk, int64_t world,
                   int64_t rank) {
  const int64_t rows = out.size(0);
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat && logits.dim() == 2 &&
              logits.stride(1) == 1 && logits.size(0) >= rows);
  TORCH_CHECK(out.scalar_type() == at::kLong && out.dim() == 2 && out.stride(1) == 1);
  TORCH_CHECK(logits.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0);
  check_rows(lengths, slots, rows);
  if (rows == 0) return;
  SelectParams p{};
  p.logits = logits.const_data_ptr<float>();
  p.lengths = lengths.const_data_ptr<int>();
  p.slots = slots.const_data_ptr();
  p.out = out.mutable_data_ptr<int64_t>();
  p.stride_logits = logits.stride(0);
  p.stride_out = out.stride(0);
  p.ratio = static_cast<int>(ratio);
  p.group_topk = static_cast<int>(group_topk);
  p.history = static_cast<int>(out.size(1));
  p.world = static_cast<int>(world);
  p.rank = static_cast<int>(rank);
  const c10::cuda::CUDAGuard guard(logits.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (slots.scalar_type() == at::kLong) {
    idx_tc_select_kernel<int64_t><<<rows, kSelectThreads, 0, stream>>>(p);
  } else {
    idx_tc_select_kernel<int32_t><<<rows, kSelectThreads, 0, stream>>>(p);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void idx_tc_scores_meta(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, at::Tensor&, const at::Tensor&, const at::Tensor&,
                        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool) {}
void idx_tc_select_meta(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
                        int64_t, int64_t, int64_t, int64_t) {}

}  // namespace

TORCH_LIBRARY(hy4_idx_tc, m) {
  m.def(
      "scores(Tensor q, Tensor w, Tensor pool, Tensor table, Tensor(a!) logits,"
      "Tensor lengths, Tensor slots, int ratio, int group_topk, int pages_per_block,"
      "int world, int rank, int width, bool fill) -> ()");
  m.def(
      "select(Tensor logits, Tensor lengths, Tensor slots, Tensor(a!) out,"
      "int ratio, int group_topk, int world, int rank) -> ()");
}

TORCH_LIBRARY_IMPL(hy4_idx_tc, CUDA, m) {
  m.impl("scores", &idx_tc_scores);
  m.impl("select", &idx_tc_select);
}

TORCH_LIBRARY_IMPL(hy4_idx_tc, Meta, m) {
  m.impl("scores", &idx_tc_scores_meta);
  m.impl("select", &idx_tc_select_meta);
}
