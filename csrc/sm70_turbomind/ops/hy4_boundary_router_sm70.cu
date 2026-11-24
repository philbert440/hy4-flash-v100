// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/library.h>

namespace {
constexpr int kHidden = 3072;
constexpr int kThreads = 128;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

bool matches(const at::Tensor& t, const at::Tensor& ref, at::ScalarType dtype,
             at::IntArrayRef shape) {
  return t.device() == ref.device() && t.is_contiguous() &&
         t.scalar_type() == dtype && t.sizes() == shape;
}

struct alignas(8) Half4 {
  half2 lo;
  half2 hi;
};

__device__ __forceinline__ float4 to_float4(float4 weight) {
  return weight;
}

__device__ __forceinline__ float4 to_float4(Half4 weight) {
  const float2 lo = __half22float2(weight.lo);
  const float2 hi = __half22float2(weight.hi);
  return make_float4(lo.x, lo.y, hi.x, hi.y);
}

// Each warp reduces one output expert, reusing its weights over the
// small batch. The input conversion is exact and happens in registers.
template <int Rows, typename Weight>
__global__ void boundary_router_gemv(const half* x, const Weight* weight,
                                     float* logits) {
  constexpr int kExperts = 512;
  const int expert = blockIdx.x * (kThreads / 32) + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  float sums[Rows]{};
  for (int i = lane; i < kHidden / 4; i += 32) {
    const float4 w = to_float4(weight[expert * (kHidden / 4) + i]);
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
      const auto* input = reinterpret_cast<const half2*>(x + row * kHidden);
      const float2 lo = __half22float2(input[2 * i]);
      const float2 hi = __half22float2(input[2 * i + 1]);
      sums[row] = fmaf(w.x, lo.x, sums[row]);
      sums[row] = fmaf(w.y, lo.y, sums[row]);
      sums[row] = fmaf(w.z, hi.x, sums[row]);
      sums[row] = fmaf(w.w, hi.y, sums[row]);
    }
  }
#pragma unroll
  for (int row = 0; row < Rows; ++row) {
    const float sum = warp_sum(sums[row]);
    if (lane == 0) logits[row * kExperts + expert] = sum;
  }
}

at::Tensor alloc_router_logits(const at::Tensor& x, const at::Tensor&) {
  return at::empty({x.size(0), 512}, x.options().dtype(at::kFloat));
}

template <typename Weight>
void launch_router(const half* in, const Weight* w, float* out, int rows) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int blocks = 512 / (kThreads / 32);
  switch (rows) {
    case 1:
      boundary_router_gemv<1, Weight><<<blocks, kThreads, 0, stream>>>(in, w, out);
      break;
    case 2:
      boundary_router_gemv<2, Weight><<<blocks, kThreads, 0, stream>>>(in, w, out);
      break;
    case 3:
      boundary_router_gemv<3, Weight><<<blocks, kThreads, 0, stream>>>(in, w, out);
      break;
    case 4:
      boundary_router_gemv<4, Weight><<<blocks, kThreads, 0, stream>>>(in, w, out);
      break;
    // Five to eight rows: the DFly verify under HY4_DREV2_WIDE. Each row keeps
    // the one-row FMA order, so a verify row equals the plain decode row.
    case 5:
      boundary_router_gemv<5, Weight><<<blocks, kThreads, 0, stream>>>(in, w, out);
      break;
    case 6:
      boundary_router_gemv<6, Weight><<<blocks, kThreads, 0, stream>>>(in, w, out);
      break;
    case 7:
      boundary_router_gemv<7, Weight><<<blocks, kThreads, 0, stream>>>(in, w, out);
      break;
    case 8:
      boundary_router_gemv<8, Weight><<<blocks, kThreads, 0, stream>>>(in, w, out);
      break;
  }
}

at::Tensor router_input(const at::Tensor& x, const at::Tensor& weight) {
  TORCH_CHECK(x.dim() == 2, "boundary router expects a matrix");
  const int rows = x.size(0);
  TORCH_CHECK(x.is_cuda() && rows >= 1 && rows <= 8 &&
                  matches(x, x, at::kHalf, {rows, kHidden}) &&
                  (matches(weight, x, at::kFloat, {512, kHidden}) ||
                   matches(weight, x, at::kHalf, {512, kHidden})) &&
                  x.storage_offset() % 2 == 0 &&
                  weight.storage_offset() % 4 == 0,
              "boundary router tensor contract");
  const c10::cuda::CUDAGuard guard(x.device());
  auto logits = alloc_router_logits(x, weight);
  const auto* in = reinterpret_cast<const half*>(x.const_data_ptr<at::Half>());
  auto* out = logits.mutable_data_ptr<float>();
  if (weight.scalar_type() == at::kHalf) {
    launch_router(in, reinterpret_cast<const Half4*>(weight.const_data_ptr<at::Half>()),
                  out, rows);
  } else {
    launch_router(in, reinterpret_cast<const float4*>(weight.const_data_ptr<float>()),
                  out, rows);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return logits;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(hy4_sm70, m) {
  m.def("boundary_router_input(Tensor x, Tensor weight) -> Tensor");
}
TORCH_LIBRARY_IMPL(hy4_sm70, CUDA, m) {
  m.impl("boundary_router_input", &router_input);
}
TORCH_LIBRARY_IMPL(hy4_sm70, Meta, m) {
  m.impl("boundary_router_input", &alloc_router_logits);
}
