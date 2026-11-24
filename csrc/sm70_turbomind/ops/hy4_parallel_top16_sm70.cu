// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Sigmoid, validity and normalization follow csrc/moe/grouped_topk_kernels.cu
// (adapted from NVIDIA TensorRT-LLM noAuxTcKernels.cu, Apache-2.0).

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <math_constants.h>
#include <torch/library.h>

namespace {

namespace cg = cooperative_groups;
constexpr int kExperts = 512;
constexpr int kTopK = 16;
constexpr unsigned kFullWarp = 0xffffffff;

__device__ float sigmoid(float x) {
  return 0.5f * tanhf(0.5f * x) + 0.5f;
}

// Finite scores and +inf compare in FP32 order. Equal scores sort by
// ascending expert ID, with both signed zeros using the same score key.
// A zero key stands for an unfilled baseline WarpSelect slot (expert 0).
__device__ uint64_t rank_key(float score, int expert) {
  if (!(score > -CUDART_INF_F)) return 0;
  const unsigned bits = score == 0.f ? 0 : __float_as_uint(score);
  const unsigned ordered =
      (bits & 0x80000000U) ? ~bits : (bits ^ 0x80000000U);
  return (static_cast<uint64_t>(ordered) << 9) | (kExperts - 1 - expert);
}

__device__ uint64_t select_partner(uint64_t key, int step, bool take_max) {
  const uint64_t other = __shfl_xor_sync(kFullWarp, key, step);
  return take_max ? max(key, other) : min(key, other);
}

__global__ void hy4_parallel_top16_kernel(const float* logits,
                                         const float* bias, float* weights,
                                         int32_t* ids, bool renorm,
                                         float scale) {
  const int expert = threadIdx.x;
  const int lane = expert % 32;
  const int warp = expert / 32;
  auto tile = cg::tiled_partition<32>(cg::this_thread_block());
  logits += blockIdx.x * kExperts;
  weights += blockIdx.x * kTopK;
  ids += blockIdx.x * kTopK;
  const float raw = logits[expert];
  const float corrected = sigmoid(raw) + bias[expert];

  __shared__ uint64_t candidates[kExperts / 2];
  __shared__ bool group_valid;
  // For typical scores the sole group is necessarily selected. For irregular
  // group-score inputs, recompute its exact top-two sum, including +/-inf
  // logits before masking them out of the expert selection below.
  if (__syncthreads_or(!(corrected > -0x1.fffffep126f) ||
                       !isfinite(corrected))) {
    if (expert == 0) {
      float first = -CUDART_INF_F;
      float second = -CUDART_INF_F;
      for (int i = 0; i < kExperts; ++i) {
        const float val = sigmoid(logits[i]) + bias[i];
        if (val > first) {
          second = first;
          first = val;
        } else if (val > second) {
          second = val;
        }
      }
      group_valid = first + second > -CUDART_INF_F;
    }
    __syncthreads();
    if (!group_valid) {
      if (expert < kTopK) {
        weights[expert] = 1.f / kTopK;
        ids[expert] = expert;
      }
      return;
    }
  }

  uint64_t key = isfinite(raw) ? rank_key(corrected, expert) : 0;
#pragma unroll
  for (int width = 2; width <= 32; width *= 2) {
#pragma unroll
    for (int step = width / 2; step; step /= 2) {
      key = select_partner(key, step, ((lane & width) == 0) == ((lane & step) == 0));
    }
  }
  if (lane < kTopK) candidates[warp * kTopK + lane] = key;
  __syncthreads();

  // Pair two 16-element sorted lists into a bitonic sequence, keeping
  // only its best half. Merge siblings in place in the left list.
#pragma unroll
  for (int span = 1; span < kExperts / 32; span *= 2) {
    if (warp % (2 * span) == 0) {
      key = lane < kTopK ? candidates[warp * kTopK + lane]
                         : candidates[(warp + span) * kTopK + 31 - lane];
#pragma unroll
      for (int step = kTopK; step; step /= 2) {
        key = select_partner(key, step, (lane & step) == 0);
      }
      if (lane < kTopK && span < kExperts / 64)
        candidates[warp * kTopK + lane] = key;
    }
    if (span < kExperts / 64) __syncthreads();
  }

  if (warp == 0) {
    const int idx = key == 0 ? 0 : kExperts - 1 - (key & (kExperts - 1));
    const float unbiased = lane < kTopK ? sigmoid(logits[idx]) : 0.f;
    if (renorm) {
      scale /= 1e-20f + cg::reduce(tile, unbiased, cg::plus<float>());
    }
    if (lane < kTopK) {
      ids[lane] = idx;
      weights[lane] = unbiased * scale;
    }
  }
}

std::tuple<at::Tensor, at::Tensor> hy4_parallel_top16(
    const at::Tensor& logits, const at::Tensor& bias, bool renorm,
    double scale) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat &&
                  logits.is_contiguous() && logits.dim() == 2 &&
                  logits.size(1) == kExperts && logits.size(0) > 0 &&
                  logits.size(0) <= 32,
              "hy4_parallel_top16 requires contiguous CUDA FP32 [1..32, 512]");
  TORCH_CHECK(bias.device() == logits.device() &&
                  bias.scalar_type() == at::kFloat && bias.is_contiguous() &&
                  bias.dim() == 1 && bias.size(0) == kExperts,
              "hy4_parallel_top16 requires contiguous CUDA FP32 bias [512]");
  const c10::cuda::CUDAGuard guard(logits.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0,
              "hy4_parallel_top16 requires SM70");
  auto weights = at::empty({logits.size(0), kTopK}, logits.options());
  auto ids = at::empty({logits.size(0), kTopK}, logits.options().dtype(at::kInt));
  hy4_parallel_top16_kernel<<<logits.size(0), kExperts, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
      logits.const_data_ptr<float>(), bias.const_data_ptr<float>(),
      weights.mutable_data_ptr<float>(), ids.mutable_data_ptr<int32_t>(),
      renorm, static_cast<float>(scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {weights, ids};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(hy4_sm70, m) {
  m.def("parallel_top16(Tensor logits, Tensor bias, bool renorm, float scale)"
        " -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(hy4_sm70, CUDA, m) {
  m.impl("parallel_top16", &hy4_parallel_top16);
}

TORCH_LIBRARY_IMPL(hy4_sm70, Meta, m) {
  m.impl("parallel_top16", [](const at::Tensor& logits, const at::Tensor& bias,
                              bool renorm, double scale) {
    const auto shape = {logits.size(0), static_cast<int64_t>(kTopK)};
    return std::make_tuple(at::empty(shape, logits.options()),
                           at::empty(shape, logits.options().dtype(at::kInt)));
  });
}
