// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "hy4_w8_sm70.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <type_traits>

namespace {

using namespace hy4_w8;

// One warp reduces a weight group and writes its chunks directly into the
// streaming layout. The loader needs no FP16 or FP32 staging matrices.
template <int GroupSize>
__global__ void pack_groups(
    const half* input, uint8_t* output, int n, int k,
    int64_t batch_stride, int64_t row_stride, int64_t col_stride) {
  const int n_pad = padded_n(n);
  const int count = groups(k, GroupSize);
  const int group_row = blockIdx.x * 4 + threadIdx.x / 32;
  if (group_row >= n_pad * count) return;
  const int group = group_row % count;
  const int row = group_row / count;
  const int lane = threadIdx.x & 31;
  const int col = group * GroupSize + lane * 4;
  const int chunk = col / 16;

  float values[4] = {};
  float maximum = 0.0f;
  if (lane < GroupSize / 4 && row < n && col < k) {
    const half* src = input + blockIdx.y * batch_stride + row * row_stride + col * col_stride;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      values[i] = __half2float(src[i * col_stride]);
      maximum = fmaxf(maximum, fabsf(values[i]));
    }
  }
#pragma unroll
  for (int stride = 16; stride > 0; stride >>= 1) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, stride));
  }
  const half scale = __float2half_rn(__fdiv_rn(maximum, 127.0f));
  const half nonzero = __half_as_ushort(scale) ? scale : __ushort_as_half(1);
  const float scale_f = __half2float(nonzero);
  uint8_t* matrix = output + blockIdx.y * matrix_bytes(n, k, GroupSize);
  if (lane == 0) {
    auto* scales = reinterpret_cast<half*>(matrix + n_pad * k);
    scales[(row / 8 * count + group) * 8 + row % 8] = nonzero;
  }
  if (lane < GroupSize / 4 && col < k) {
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int q = __float2int_rn(__fdiv_rn(values[i], scale_f));
      packed |= static_cast<uint32_t>(min(max(q, -127), 127) + 128) << (8 * i);
    }
    auto* dest = reinterpret_cast<uint32_t*>(matrix);
    dest[((row / 8 * (k / 16) + chunk) * 8 + row % 8) * 4 + lane % 4] = packed;
  }
}

template <int GroupSize>
at::Tensor pack_fp16_group(const at::Tensor& weight) {
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == at::kHalf &&
              (weight.dim() == 2 || weight.dim() == 3) &&
              weight.size(-2) > 0 && weight.size(-1) > 0 &&
              weight.size(-1) % 16 == 0,
              "Unsupported Hy4 W8 pack shape or dtype");
  const c10::cuda::CUDAGuard guard(weight.device());
  auto shape = weight.sizes().vec();
  const int n = weight.size(-2);
  const int k = weight.size(-1);
  const int batch = weight.dim() == 3 ? weight.size(0) : 1;
  shape[shape.size() - 2] = padded_n(n);
  shape.back() = k + 2 * groups(k, GroupSize);
  auto out = at::empty(shape, weight.options().dtype(at::kByte));
  if (batch > 0) {
    const int count = padded_n(n) * groups(k, GroupSize);
    const auto stream = at::cuda::getCurrentCUDAStream();
    pack_groups<GroupSize><<<dim3((count + 3) / 4, batch), 128, 0, stream>>>(
        reinterpret_cast<const half*>(weight.const_data_ptr<at::Half>()),
        out.mutable_data_ptr<uint8_t>(), n, k,
        weight.dim() == 3 ? weight.stride(0) : 0, weight.stride(-2), weight.stride(-1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

at::Tensor pack_fp16_grouped(const at::Tensor& weight, int64_t group_size) {
  switch (group_size) {
    case 32: return pack_fp16_group<32>(weight);
    case 64: return pack_fp16_group<64>(weight);
    case 128: return pack_fp16_group<128>(weight);
    default: TORCH_CHECK(false, "Unsupported Hy4 W8 group size");
  }
}

template <bool SplitK, int GroupSize, typename Output>
__global__ void __launch_bounds__(128) w8_project(
    const half* input, const uint8_t* packed, Output* output, float* partial,
    int m, int n, int k) {
  const int batch = blockIdx.z;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int quad = (lane >> 2) & 3;
  const int tile = blockIdx.x * 4 + quad;
  const int col_tiles = padded_n(n) / 8;
  const int split = blockIdx.y * 4 + warp;
  const int chunks = k / 16;
  const int splits = SplitK ? gridDim.y * 4 : 4;
  const int begin = chunks * split / splits;
  const int end = chunks * (split + 1) / splits;
  float acc[1][8] = {};
  warp_project<1, false, false, GroupSize>(
      input + batch * m * k, packed + batch * matrix_bytes(n, k, GroupSize),
      m, n, k, min(tile, col_tiles - 1), begin, end, end - begin, acc);

  __shared__ float red[4][8][32];
  if (tile < col_tiles) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      red[warp][acc_row(i, lane)][quad * 8 + acc_col(i, lane)] = acc[0][i];
    }
  }
  __syncthreads();
  const int local_col = threadIdx.x & 31;
  const int col = blockIdx.x * 32 + local_col;
  if (col >= n) return;
  for (int row = warp; row < m; row += 4) {
    float sum = red[0][row][local_col] + red[1][row][local_col]
              + red[2][row][local_col] + red[3][row][local_col];
    const int offset = (batch * m + row) * n + col;
    if constexpr (!SplitK) {
      output[offset] = static_cast<Output>(sum);
    } else {
      partial[blockIdx.y * gridDim.z * m * n + offset] = sum;
    }
  }
}

template <typename Output>
__global__ void finish(const float* partial, Output* output, int count, int splits) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  float sum = partial[idx];
  for (int s = 1; s < splits; ++s) sum += partial[s * count + idx];
  output[idx] = static_cast<Output>(sum);
}

template <int GroupSize, bool FloatOutput = false>
at::Tensor projection_group(
    const at::Tensor& x, const at::Tensor& w, int64_t n, int64_t override_splits) {
  TORCH_CHECK(x.dim() == 2 || x.dim() == 3);
  const int b = x.dim() == 3 ? x.size(0) : 1;
  const int m = x.size(-2);
  const int k = x.size(-1);
  TORCH_CHECK(x.is_cuda() && x.device() == w.device() &&
              x.scalar_type() == at::kHalf && w.scalar_type() == at::kByte &&
              x.is_contiguous() && w.is_contiguous() &&
              x.storage_offset() % 8 == 0 && w.storage_offset() % 16 == 0 &&
              m >= 1 && m <= 8 && n > 0 && k > 0 && k % 16 == 0 &&
              w.numel() == b * matrix_bytes(n, k, GroupSize),
              "Unsupported Hy4 W8 projection");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto options = x.options().dtype(FloatOutput ? at::kFloat : at::kHalf);
  auto out = x.dim() == 3 ? at::empty({b, m, n}, options)
                          : at::empty({m, n}, options);
  const auto* in = reinterpret_cast<const half*>(x.const_data_ptr<at::Half>());
  const auto* packed = w.const_data_ptr<uint8_t>();
  using Output = std::conditional_t<FloatOutput, float, half>;
  auto* result = reinterpret_cast<Output*>(out.mutable_data_ptr());
  const int tiles = (n + 31) / 32;
  TORCH_CHECK(override_splits >= 0 && override_splits <= 16);
  // Use the same split count for all small batches, including DFly verify.
  int splits = static_cast<int>(override_splits);
  if (splits == 0) {
    if (k >= 1536 || (k >= 1024 && tiles * b < 80)) {
      splits = 8;
    } else {
      splits = k >= 512 ? 4 : 1;
    }
  }
  if (splits > 1) {
    const int count = b * m * n;
    auto scratch = at::empty({splits, count}, x.options().dtype(at::kFloat));
    auto* partial = scratch.mutable_data_ptr<float>();
    w8_project<true, GroupSize, Output><<<dim3(tiles, splits, b), 128, 0, stream>>>(
        in, packed, result, partial, m, n, k);
    finish<Output><<<(count + 127) / 128, 128, 0, stream>>>(partial, result, count, splits);
  } else {
    w8_project<false, GroupSize, Output><<<dim3(tiles, 1, b), 128, 0, stream>>>(
        in, packed, result, nullptr, m, n, k);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor projection_grouped(
    const at::Tensor& x, const at::Tensor& w, int64_t n,
    int64_t group_size, int64_t override_splits) {
  switch (group_size) {
    case 32: return projection_group<32>(x, w, n, override_splits);
    case 64: return projection_group<64>(x, w, n, override_splits);
    case 128: return projection_group<128>(x, w, n, override_splits);
    default: TORCH_CHECK(false, "Unsupported Hy4 W8 group size");
  }
}

at::Tensor head_projection(
    const at::Tensor& x, const at::Tensor& w, int64_t n, int64_t group_size) {
  // The vocab shard supplies ample N tiles, so skip the split-K staging.
  switch (group_size) {
    case 64: return projection_group<64, true>(x, w, n, 1);
    case 128: return projection_group<128, true>(x, w, n, 1);
    default: TORCH_CHECK(false, "Unsupported Hy4 W8 head group size");
  }
}

at::Tensor head_projection_meta(
    const at::Tensor& x, const at::Tensor&, int64_t n, int64_t) {
  auto shape = x.sizes().vec();
  shape.back() = n;
  return at::empty(shape, x.options().dtype(at::kFloat));
}

at::Tensor projection_meta(const at::Tensor& x, const at::Tensor&, int64_t n, int64_t) {
  auto shape = x.sizes().vec();
  shape.back() = n;
  return at::empty(shape, x.options());
}

at::Tensor projection_meta_grouped(
    const at::Tensor& x, const at::Tensor& w, int64_t n, int64_t, int64_t splits) {
  return projection_meta(x, w, n, splits);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(hy4_sm70, m) {
  m.def("w8_head_projection(Tensor x, Tensor packed, int n, int group_size) -> Tensor");
  m.impl("w8_head_projection", c10::DispatchKey::CUDA, &head_projection);
  m.impl("w8_head_projection", c10::DispatchKey::Meta, &head_projection_meta);
  m.def("w8_pack_fp16(Tensor weight) -> Tensor");
  m.impl("w8_pack_fp16", c10::DispatchKey::CUDA, &pack_fp16_group<128>);
  m.def("w8_projection(Tensor x, Tensor packed, int n, int splits=0) -> Tensor");
  m.impl("w8_projection", c10::DispatchKey::CUDA, &projection_group<128>);
  m.impl("w8_projection", c10::DispatchKey::Meta, &projection_meta);
  m.def("w8_pack_fp16_grouped(Tensor weight, int group_size) -> Tensor");
  m.impl("w8_pack_fp16_grouped", c10::DispatchKey::CUDA, &pack_fp16_grouped);
  m.def("w8_projection_grouped(Tensor x, Tensor packed, int n, int group_size, int splits=0) -> Tensor");
  m.impl("w8_projection_grouped", c10::DispatchKey::CUDA, &projection_grouped);
  m.impl("w8_projection_grouped", c10::DispatchKey::Meta, &projection_meta_grouped);
}
