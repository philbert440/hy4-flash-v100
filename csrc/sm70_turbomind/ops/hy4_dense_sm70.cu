// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/library.h>

namespace {

template <typename T>
__device__ float as_float(T value) {
  return static_cast<float>(value);
}
template <>
__device__ float as_float(half value) {
  return __half2float(value);
}

// One output channel per warp. All input rows reuse the same vector loads.
template <int M, typename T, int WarpsPerChannel>
__global__ void skinny(const T* x, const half* w, T* y, int n, int k) {
  constexpr int kThreadsPerChannel = 32 * WarpsPerChannel;
  constexpr int kChannels = 128 / kThreadsPerChannel;
  const int channel = blockIdx.x * kChannels + threadIdx.x / kThreadsPerChannel;
  if (channel >= n) return;
  const int lane = threadIdx.x % kThreadsPerChannel;
  float sums[M][2] = {};
  const auto* weight = reinterpret_cast<const uint4*>(w + int64_t(channel) * k);
  for (int chunk = lane; chunk < k / 8; chunk += kThreadsPerChannel) {
    const uint4 packed = __ldg(weight + chunk);
    const auto* values = reinterpret_cast<const half*>(&packed);
#pragma unroll
    for (int row = 0; row < M; ++row) {
      const T* input = x + row * k + chunk * 8;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        sums[row][i % 2] = fmaf(as_float(input[i]), as_float(values[i]),
                                sums[row][i % 2]);
      }
    }
  }
  __shared__ float partials[M][4];
#pragma unroll
  for (int row = 0; row < M; ++row) {
    float sum = sums[row][0] + sums[row][1];
#pragma unroll
    for (int shift = 16; shift > 0; shift /= 2) {
      sum += __shfl_down_sync(0xffffffffu, sum, shift);
    }
    if constexpr (WarpsPerChannel > 1) {
      if (lane % 32 == 0) partials[row][threadIdx.x / 32] = sum;
    } else {
      if (lane == 0) y[row * n + channel] = static_cast<T>(sum);
    }
  }
  if constexpr (WarpsPerChannel > 1) {
    __syncthreads();
    if (lane < M) {
      float sum = partials[lane][0];
#pragma unroll
      for (int i = 1; i < WarpsPerChannel; ++i) sum += partials[lane][i];
      y[lane * n + channel] = static_cast<T>(sum);
    }
  }
}

template <typename T, int WarpsPerChannel>
void launch(const T* x, const half* w, T* out, int m, int n, int k,
            cudaStream_t stream) {
  const int blocks = (n * WarpsPerChannel + 3) / 4;
#define LAUNCH_ROWS(rows)                                                        \
  case rows:                                                                     \
    skinny<rows, T, WarpsPerChannel><<<blocks, 128, 0, stream>>>(x, w, out, n, k); \
    break
  switch (m) {
    LAUNCH_ROWS(1);
    LAUNCH_ROWS(2);
    LAUNCH_ROWS(3);
    LAUNCH_ROWS(4);
  }
#undef LAUNCH_ROWS
}

at::Tensor dense_gemv(const at::Tensor& x, const at::Tensor& weight) {
  TORCH_CHECK(x.is_cuda() && x.device() == weight.device(),
              "Hy4 dense GEMV requires tensors on the same CUDA device.");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() &&
              x.dim() == 2 && weight.dim() == 2 &&
              x.size(0) >= 1 && x.size(0) <= 4 &&
              x.size(1) > 0 && x.size(1) == weight.size(1) && x.size(1) % 8 == 0 &&
              weight.size(0) > 0 && weight.storage_offset() % 8 == 0 &&
              weight.scalar_type() == at::kHalf &&
              x.scalar_type() == at::kFloat,
              "Unsupported Hy4 dense GEMV shape, layout or dtype.");
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty({x.size(0), weight.size(0)}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* w = reinterpret_cast<const half*>(weight.const_data_ptr<at::Half>());
  launch<float, 4>(x.const_data_ptr<float>(), w, out.data_ptr<float>(),
                   x.size(0), weight.size(0), x.size(1), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(hy4_sm70, m) {
  m.def("dense_gemv(Tensor x, Tensor weight) -> Tensor");
  m.impl("dense_gemv", c10::DispatchKey::CUDA, &dense_gemv);
}
