// SPDX-License-Identifier: Apache-2.0
// Single-segment Volta skinny GEMM for Hy4's TP4 dense projections.
// Quadpair tiling follows ReVolta's fp16_skinny_gemm_sm70.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/library.h>

namespace {
__device__ __forceinline__ void mma(float (&c)[8], uint32_t a0, uint32_t a1,
                                    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]),
        "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

template <bool Weight>
__device__ __forceinline__ uint4 load8(const half* base, int col, int end) {
  if (col >= end) return make_uint4(0, 0, 0, 0);
  const auto* ptr = reinterpret_cast<const uint4*>(base + col);
  if constexpr (Weight) {
    return __ldcs(ptr);
  } else {
    return __ldg(ptr);
  }
}

template <int KWarps>
__global__ void __launch_bounds__(32 * KWarps)
    fp16_tiled(const half* x, const half* w, half* y, float* partials,
              int n, int k, int m) {
  constexpr int kTileN = 8;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int quadpair = (lane >> 2) & 3;
  const int idx8 = (lane & 3) + ((lane & 16) >> 2);
  const int c0 = kTileN * blockIdx.x;
  const int c = c0 + idx8;
  const int parts = gridDim.y * KWarps;
  const int chunk = blockIdx.y * KWarps + warp;
  const int k_start = (chunk * (k / 64) / parts) * 64;
  const int k_end = ((chunk + 1) * (k / 64) / parts) * 64;
  const half* weight = w + static_cast<int64_t>(c) * k;
  const half* input = x + (idx8 < m ? idx8 * k : 0);
  const int offset = quadpair * 8;
  float accum[8] = {};
  uint4 b0 = load8<true>(weight, k_start + offset, k_end);
  uint4 b1 = load8<true>(weight, k_start + offset + 32, k_end);
  for (int pos = k_start; pos < k_end; pos += 64) {
    const uint4 b0_next = load8<true>(weight, pos + offset + 64, k_end);
    const uint4 b1_next = load8<true>(weight, pos + offset + 96, k_end);
    const uint4 a0 = load8<false>(input, pos + offset, idx8 < m ? k_end : 0);
    const uint4 a1 = load8<false>(input, pos + offset + 32, idx8 < m ? k_end : 0);
    mma(accum, a0.x, a0.y, b0.x, b0.y);
    mma(accum, a0.z, a0.w, b0.z, b0.w);
    mma(accum, a1.x, a1.y, b1.x, b1.y);
    mma(accum, a1.z, a1.w, b1.z, b1.w);
    b0 = b0_next;
    b1 = b1_next;
  }

  __shared__ volatile float shared[KWarps][4][kTileN];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    float sum = __shfl_sync(0xffffffff, accum[i], lane & ~12);
#pragma unroll
    for (int q = 1; q < 4; ++q) {
      sum += __shfl_sync(0xffffffff, accum[i], (lane & ~12) | (q << 2));
    }
    if (row < m && quadpair == 0) {
      shared[warp][row][col] = sum;
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < m * kTileN; i += blockDim.x) {
    const int row = i / kTileN;
    const int col = i % kTileN;
    float sum = shared[0][row][col];
#pragma unroll
    for (int wk = 1; wk < KWarps; ++wk) sum += shared[wk][row][col];
    if (gridDim.y == 1) y[row * n + c0 + col] = __float2half(sum);
    else partials[(blockIdx.y * m + row) * n + c0 + col] = sum;
  }
}

__global__ void finish(half* y, const float* partials, int size, int splits) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;
  float sum = partials[idx];
  for (int s = 1; s < splits; ++s) sum += partials[s * size + idx];
  y[idx] = __float2half(sum);
}

bool supported_projection(int n, int k) {
  return (k == 3072 && (n == 768 || n == 1536)) ||
         (n == 3072 && (k == 384 || k == 1536 || k == 2048 || k == 4096));
}

at::Tensor skinny_projection(const at::Tensor& x, const at::Tensor& weight) {
  TORCH_CHECK(x.is_cuda() && x.device() == weight.device() &&
              x.scalar_type() == at::kHalf && weight.scalar_type() == at::kHalf &&
              x.dim() == 2 && weight.dim() == 2 && x.is_contiguous() &&
              weight.is_contiguous() && x.storage_offset() % 8 == 0 &&
              weight.storage_offset() % 8 == 0 && x.size(0) >= 1 &&
              x.size(0) <= 4 && x.size(1) == weight.size(1) &&
              supported_projection(weight.size(0), weight.size(1)),
              "Unsupported Hy4 skinny projection tensor contract.");
  const c10::cuda::CUDAGuard guard(x.device());
  const int m = x.size(0);
  const int n = weight.size(0);
  const int k = weight.size(1);
  // SWA o_proj reads the largest matrix. Two K warps gave as much bandwidth
  // as four there. Split shared gate/up into two K tiles to fill more SMs.
  const int splits = n == 768 ? 2 : 1;
  const dim3 grid(n / 8, splits);
  auto out = at::empty({m, n}, x.options());
  at::Tensor workspace;
  float* partials = nullptr;
  if (splits > 1) {
    workspace = at::empty({splits, m, n}, x.options().dtype(at::kFloat));
    partials = workspace.mutable_data_ptr<float>();
  }
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const half*>(x.const_data_ptr<at::Half>());
  const auto* wp = reinterpret_cast<const half*>(weight.const_data_ptr<at::Half>());
  auto* yp = reinterpret_cast<half*>(out.mutable_data_ptr<at::Half>());
  if (k == 4096) {
    fp16_tiled<2><<<grid, 64, 0, stream>>>(xp, wp, yp, partials, n, k, m);
  } else {
    fp16_tiled<4><<<grid, 128, 0, stream>>>(xp, wp, yp, partials, n, k, m);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (splits > 1) {
    finish<<<(m * n + 255) / 256, 256, 0, stream>>>(yp, partials, m * n, splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(hy4_sm70, m) {
  m.def("skinny_projection(Tensor x, Tensor weight) -> Tensor");
  m.impl("skinny_projection", c10::DispatchKey::CUDA, &skinny_projection);
  m.impl("skinny_projection", c10::DispatchKey::Meta,
         [](const at::Tensor& x, const at::Tensor& weight) {
           return at::empty({x.size(0), weight.size(0)}, x.options());
         });
}
