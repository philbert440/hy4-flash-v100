// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty_like.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "../../custom_all_reduce.cuh"

namespace {
using namespace vllm;
using Pack = packed_t<half>::P;
using Acc = packed_t<half>::A;

constexpr int kHidden = 3072;
constexpr int kThreads = kHidden / Pack::size;
constexpr int kPeers = kSm70Tp4PushAllreduceWorldSize;
constexpr int kLogicalThreads = kSm70Tp4PushAllreduceThreads;
static_assert(kThreads % kLogicalThreads == 0);
constexpr int kSegments = kThreads / kLogicalThreads;

__global__ void boundary_ar_rms_norm(RankData peers, int rank,
                                      const Pack* input, const Pack* other,
                                      const Pack* weight,
                                      Pack* output, float epsilon) {
  const int tid = threadIdx.x;
  const int offset = blockIdx.x * kThreads + tid;
  const int block = blockIdx.x * kSegments + tid / kLogicalThreads;
  auto* storage = const_cast<char*>(static_cast<const char*>(peers.ptrs[rank]));
  auto* epochs = reinterpret_cast<uint32_t*>(storage);

  // A row covers three blocks of the ordinary push kernel. Keep their
  // independent epoch indices and the original mapping from packs to slots.
  const uint32_t epoch = epochs[block];
  const int epoch_offset =
      epoch * kPeers * (kSm70Tp4PushAllreduceMaxBytes / sizeof(Pack));
  Pack local = input[offset];
  if (other) packed_assign_add(local, other[offset]);
  Pack peer_values[kPeers];
  sm70_tp4_push_transfer(peers, storage, rank, epoch_offset, offset,
                         local, peer_values);
  const Pack reduced = sm70_push_reduce<Pack, kPeers, Acc>(peer_values);
  sm70_tp4_push_clear<Pack, kPeers>(storage, epoch_offset, offset);

  float squares = 0.f;
#pragma unroll
  for (int i = 0; i < Pack::size; ++i) {
    const float value = __half2float(reduced.data[i]);
    squares = fmaf(value, value, squares);
  }
  using Reduction = cub::BlockReduce<float, kThreads>;
  __shared__ typename Reduction::TempStorage scratch;
  __shared__ float rrms;
  const float sum = Reduction(scratch).Sum(squares);
  if (tid % kLogicalThreads == 0) {
    epochs[block] = (epoch + 1) % kSm70Tp4PushAllreduceEpochs;
  }
  if (tid == 0) rrms = rsqrtf(sum / kHidden + epsilon);
  __syncthreads();

  const Pack scale = weight[tid];
  Pack norm;
#pragma unroll
  for (int i = 0; i < Pack::size; ++i) {
    const float value = __half2float(reduced.data[i]);
    norm.data[i] = __float2half_rn(value * rrms * __half2float(scale.data[i]));
  }
  output[offset] = norm;
}

at::Tensor alloc_ar_norm(const at::Tensor& input,
                         const std::optional<at::Tensor>&, const at::Tensor&, double,
                         at::IntArrayRef, int64_t) {
  return at::empty_like(input);
}

at::Tensor ar_norm(const at::Tensor& input,
                   const std::optional<at::Tensor>& other,
                   const at::Tensor& weight,
                   double epsilon, at::IntArrayRef peers, int64_t rank) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kHalf &&
                  input.dim() == 2 && input.size(1) == kHidden &&
                  input.size(0) >= 1 && input.size(0) <= 8 &&
                  input.is_contiguous() && input.storage_offset() % Pack::size == 0 &&
                  weight.device() == input.device() && weight.scalar_type() == at::kHalf &&
                  weight.dim() == 1 && weight.size(0) == kHidden &&
                  weight.is_contiguous() && weight.storage_offset() % Pack::size == 0 &&
                  (!other || (other->device() == input.device() &&
                              other->scalar_type() == at::kHalf &&
                              other->sizes() == input.sizes() &&
                              other->is_contiguous() &&
                              other->storage_offset() % Pack::size == 0)) &&
                  peers.size() == kPeers && rank >= 0 && rank < kPeers,
              "Hy4 boundary all-reduce norm tensor contract");
  const c10::cuda::CUDAGuard guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  cudaStreamCaptureStatus capture;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
  TORCH_CHECK(capture == cudaStreamCaptureStatusActive &&
                  custom_allreduce_current_device_is_sm70(),
              "Hy4 boundary all-reduce norm requires captured SM70 TP4 push");
  RankData buffers{};
  for (int i = 0; i < kPeers; ++i) {
    buffers.ptrs[i] = reinterpret_cast<const void*>(peers[i]);
  }
  auto output = at::empty_like(input);
  boundary_ar_rms_norm<<<input.size(0), kThreads, 0, stream>>>(
      buffers, static_cast<int>(rank),
      reinterpret_cast<const Pack*>(input.const_data_ptr<at::Half>()),
      other ? reinterpret_cast<const Pack*>(other->const_data_ptr<at::Half>()) : nullptr,
      reinterpret_cast<const Pack*>(weight.const_data_ptr<at::Half>()),
      reinterpret_cast<Pack*>(output.mutable_data_ptr<at::Half>()),
      static_cast<float>(epsilon));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(hy4_sm70, m) {
  m.def("boundary_ar_norm(Tensor input, Tensor? other, Tensor weight, float epsilon,"
        " int[] peers, int rank) -> Tensor");
}
TORCH_LIBRARY_IMPL(hy4_sm70, CUDA, m) {
  m.impl("boundary_ar_norm", &ar_norm);
}
TORCH_LIBRARY_IMPL(hy4_sm70, Meta, m) {
  m.impl("boundary_ar_norm", &alloc_ar_norm);
}
