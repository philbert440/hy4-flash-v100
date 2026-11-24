// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Volta MMA tiling adapted from the Hy4 F7 attention prototype. Each
// output row always uses the same K partition and reduction order.

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/library.h>

namespace {
constexpr int kThreads = 256;
constexpr int kWarpsK = 4;
constexpr int kTileN = 16;
constexpr int kHidden = 3072;
constexpr int kLatent = 1536;
constexpr int kShared = 384;

#define HY4_F7_MMA(C, A0, A1, B0, B1)                                \
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
__device__ __forceinline__ uint4 ld_x(const half* p) {
  return __ldcg(reinterpret_cast<const uint4*>(p));
}
// Compute 16 output columns of X[m, K] @ W[16, K]^T into shared memory red[kWarpsK][RB*8][16].
// K must be a multiple of 64 (3072, 1024, 192).
template <int RB>
__device__ __forceinline__ void mma_tile_16(
    const half* __restrict__ X,
    const half* __restrict__ W_tile,
    int m,
    int K, int stride,
    volatile float* red) {
  constexpr int kRows = RB * 8;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int wn = warp & 1;
  const int wk = warp >> 1;
  const int qp = (lane >> 2) & 3;
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0);

  const int steps16 = K >> 4;
  const int k_begin = (wk * steps16 / kWarpsK) * 16;
  const int k_end = ((wk + 1) * steps16 / kWarpsK) * 16;

  const half* wrow = W_tile + static_cast<size_t>(wn * 8 + r8) * stride;
  const half* xrow[RB];
  bool row_ok[RB];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    const int r = rb * 8 + r8;
    row_ok[rb] = r < m;
    xrow[rb] = X + static_cast<size_t>(row_ok[rb] ? r : 0) * stride;
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
      const uint4 a_lo = (row_ok[rb] && lo_in) ? ld_x(xrow[rb] + lo_k) : zero4();
      const uint4 a_hi = (row_ok[rb] && hi_in) ? ld_x(xrow[rb] + hi_k) : zero4();
      HY4_F7_MMA(acc[rb], a_lo.x, a_lo.y, b_lo.x, b_lo.y);
      HY4_F7_MMA(acc[rb], a_lo.z, a_lo.w, b_lo.z, b_lo.w);
      HY4_F7_MMA(acc[rb], a_hi.x, a_hi.y, b_hi.x, b_hi.y);
      HY4_F7_MMA(acc[rb], a_hi.z, a_hi.w, b_hi.z, b_hi.w);
    }
    b_lo = n_lo;
    b_hi = n_hi;
  }

  // Deterministic quadpair reduction q0 + q1 + q2 + q3.
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

template <int RB>
__device__ __forceinline__ float sum_tile(volatile float* red, int i) {
  constexpr int stride = RB * 8 * kTileN;
  float value = red[i];
#pragma unroll
  for (int k = 1; k < kWarpsK; k++) {
    value = __fadd_rn(value, red[i + k * stride]);
  }
  return value;
}

template <int RB>
__global__ void hy4_f7_moe_in(const half* x, const half* down,
                              const half* gate_up, half* latent,
                              half* shared, int m) {
  __shared__ volatile float red[kWarpsK * RB * 8 * kTileN];
  constexpr int latent_tiles = kLatent / kTileN;
  const int tile = blockIdx.x;
  const bool routed = tile < latent_tiles;
  const int col = (routed ? tile : tile - latent_tiles) * kTileN;
  const int i = threadIdx.x;
  const int row = i / kTileN;
  const int out_col = col + i % kTileN;
  const bool valid = row < RB * 8 && row < m;
  const half* weight = routed ? down : gate_up;
  mma_tile_16<RB>(x, weight + col * kHidden, m, kHidden, kHidden, red);
  float value = valid ? __half2float(__float2half_rn(sum_tile<RB>(red, i))) : 0.f;
  // At RB=3 the epilogue above covers 16 of 24 rows. Handle the
  // remaining eight with a second output per thread.
  const int j = i + kThreads;
  const int extra_row = j / kTileN;
  const bool extra = extra_row < RB * 8 && extra_row < m;
  float extra_value =
      extra ? __half2float(__float2half_rn(sum_tile<RB>(red, j))) : 0.f;
  if (routed) {
    if (valid) latent[row * kLatent + out_col] = __float2half_rn(value);
    if (extra) latent[extra_row * kLatent + out_col] = __float2half_rn(extra_value);
    return;
  }
  __syncthreads();
  mma_tile_16<RB>(x, weight + (kShared + col) * kHidden, m, kHidden, kHidden, red);
  if (valid) {
    float up = __half2float(__float2half_rn(sum_tile<RB>(red, i)));
    shared[row * kShared + out_col] =
        __float2half_rn((value / (1.f + expf(-value))) * up);
  }
  if (extra) {
    float up = __half2float(__float2half_rn(sum_tile<RB>(red, j)));
    shared[extra_row * kShared + out_col] =
        __float2half_rn((extra_value / (1.f + expf(-extra_value))) * up);
  }
}

template <int RB>
__global__ void hy4_f7_moe_out(const half* latent, const half* shared,
                               const half* up, const half* down, half* out,
                               int m) {
  __shared__ volatile float red[kWarpsK * RB * 8 * kTileN];
  const int col = blockIdx.x * kTileN;
  const int i = threadIdx.x;
  const int j = i + kThreads;
  const int row = i / kTileN;
  const int extra_row = j / kTileN;
  const int out_col = col + i % kTileN;
  const bool valid = row < RB * 8 && row < m;
  const bool extra = extra_row < RB * 8 && extra_row < m;
  mma_tile_16<RB>(latent, up + col * kLatent, m, kLatent, kLatent, red);
  float value = valid ? __half2float(__float2half_rn(sum_tile<RB>(red, i))) : 0.f;
  float extra_value =
      extra ? __half2float(__float2half_rn(sum_tile<RB>(red, j))) : 0.f;
  __syncthreads();
  mma_tile_16<RB>(shared, down + col * kShared, m, kShared, kShared, red);
  if (valid) {
    float other = __half2float(__float2half_rn(sum_tile<RB>(red, i)));
    out[row * kHidden + out_col] = __float2half_rn(other + value);
  }
  if (extra) {
    float other = __half2float(__float2half_rn(sum_tile<RB>(red, j)));
    out[extra_row * kHidden + out_col] = __float2half_rn(other + extra_value);
  }
}

bool half_matrix(const at::Tensor& t, const at::Tensor& x, int rows, int cols) {
  return t.device() == x.device() && t.scalar_type() == at::kHalf &&
         t.is_contiguous() && t.dim() == 2 && t.size(0) == rows &&
         t.size(1) == cols && t.storage_offset() % 8 == 0;
}

const half* input(const at::Tensor& t) {
  return reinterpret_cast<const half*>(t.const_data_ptr<at::Half>());
}

half* output(at::Tensor& t) {
  return reinterpret_cast<half*>(t.mutable_data_ptr<at::Half>());
}

auto alloc_in(const at::Tensor& x, const at::Tensor&, const at::Tensor&) {
  return std::make_tuple(at::empty({x.size(0), kLatent}, x.options()),
                         at::empty({x.size(0), kShared}, x.options()));
}

at::Tensor alloc_out(const at::Tensor& x, const at::Tensor&,
                     const at::Tensor&, const at::Tensor&) {
  return at::empty({x.size(0), kHidden}, x.options());
}

template <typename F>
void dispatch_rows(int m, F launch) {
  if (m <= 8) launch(std::integral_constant<int, 1>{});
  else if (m <= 16) launch(std::integral_constant<int, 2>{});
  else launch(std::integral_constant<int, 3>{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<at::Tensor, at::Tensor> moe_in(const at::Tensor& x,
                                       const at::Tensor& down,
                                       const at::Tensor& gate_up) {
  const int m = x.size(0);
  TORCH_CHECK(x.is_cuda() && m > 0 && m <= 24 &&
              half_matrix(x, x, m, kHidden) &&
              half_matrix(down, x, kLatent, kHidden) &&
              half_matrix(gate_up, x, 2 * kShared, kHidden),
              "F7 MoE input projection tensor contract");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto [latent, shared] = alloc_in(x, down, gate_up);
  dispatch_rows(m, [&](auto rb) {
    hy4_f7_moe_in<decltype(rb)::value>
        <<<(kLatent + kShared) / kTileN, kThreads, 0, stream>>>(
            input(x), input(down), input(gate_up),
            output(latent), output(shared), m);
  });
  return {latent, shared};
}

at::Tensor moe_out(const at::Tensor& latent, const at::Tensor& shared,
                   const at::Tensor& up, const at::Tensor& down) {
  const int m = latent.size(0);
  TORCH_CHECK(latent.is_cuda() && m > 0 && m <= 24 &&
              half_matrix(latent, latent, m, kLatent) &&
              half_matrix(shared, latent, m, kShared) &&
              half_matrix(up, latent, kHidden, kLatent) &&
              half_matrix(down, latent, kHidden, kShared),
              "F7 MoE output projection tensor contract");
  const c10::cuda::CUDAGuard guard(latent.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto out = alloc_out(latent, shared, up, down);
  dispatch_rows(m, [&](auto rb) {
    hy4_f7_moe_out<decltype(rb)::value><<<kHidden / kTileN, kThreads, 0, stream>>>(
        input(latent), input(shared), input(up), input(down), output(out), m);
  });
  return out;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(hy4_sm70, m) {
  m.def("f7_moe_in(Tensor x, Tensor down, Tensor gate_up) -> (Tensor, Tensor)");
  m.def("f7_moe_out(Tensor latent, Tensor shared, Tensor up, Tensor down) -> Tensor");
}

TORCH_LIBRARY_IMPL(hy4_sm70, CUDA, m) {
  m.impl("f7_moe_in", &moe_in);
  m.impl("f7_moe_out", &moe_out);
}

TORCH_LIBRARY_IMPL(hy4_sm70, Meta, m) {
  m.impl("f7_moe_in", &alloc_in);
  m.impl("f7_moe_out", &alloc_out);
}
