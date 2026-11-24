// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include <cstring>
#include <torch/library.h>

namespace {
constexpr int kHidden = 3072;
constexpr int kChannels = 4;
constexpr int kWidth = kHidden * kChannels;
constexpr int kGates = 2 * kChannels;
constexpr int kThreads = 128;
constexpr int kChunks = kWidth / kThreads;
constexpr int kFields = kGates + 1;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ float load_half_cg(const half* addr) {
  return __half2float(__ushort_as_half(
      __ldcg(reinterpret_cast<const unsigned short*>(addr))));
}

__device__ __forceinline__ float gate(float value, float rms, float scale,
                                      float bias) {
  const float logit = __fadd_rn(__fmul_rn(__fmul_rn(value, rms), scale), bias);
  return __fdividef(1.f, 1.f + __expf(-logit));
}

struct Bridge {
  const half* residual;
  const half* previous;
  const float* previous_post;
  const float* weight;
  const float* scale;
  const float* base;
  half* updated;
  half* reduced;
  float* post;
  float* partial;
  int rows;
  float norm_eps;
  float hc_eps;
  float magnitude;
  const half* rms_weight;
  float rms_eps;
};

struct alignas(16) Half8 {
  half data[8];
};

template <int Threads = kThreads>
__device__ __forceinline__ void mix_and_rms(Bridge p, int row,
                                             volatile float scratch[kFields][4]) {
  // Keep the 384 logical vectors and stock variance grouping. Map
  // one or three vectors to each physical thread, depending on CTA size.
  constexpr int kWarps = Threads / 32;
  constexpr int kVectors = kHidden / 8;
  constexpr int kParts = kVectors / Threads;
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int warp = tid / 32;
  __shared__ float warp_variance[kParts * kWarps];
  __shared__ float inverse_rms;
  Half8 values[kParts];
#pragma unroll
  for (int part = 0; part < kParts; ++part) {
    const int vector = tid + part * Threads;
    float mix[8]{};
#pragma unroll
    for (int channel = 0; channel < kChannels; ++channel) {
      const auto* source = reinterpret_cast<const uint4*>(p.updated) +
                           (row * kChannels + channel) * kVectors + vector;
      const uint4 packed = __ldcg(source);
      const auto* activation = reinterpret_cast<const half*>(&packed);
      const float g = scratch[channel + 1][1];
#pragma unroll
      for (int i = 0; i < 8; ++i)
        mix[i] = __fadd_rn(mix[i], __fmul_rn(__half2float(activation[i]), g));
    }
    float variance = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      values[part].data[i] = __float2half_rn(mix[i]);
      const float x = __half2float(values[part].data[i]);
      variance = fmaf(x, x, variance);
    }
    if constexpr (Threads != kThreads) __syncwarp();
    // CUB BlockReduce uses ascending shuffle offsets and adds warp
    // aggregates in warp-index order for the stock C1/C4 norm.
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2)
      variance += __shfl_down_sync(0xffffffff, variance, offset);
    if (lane == 0) warp_variance[part * kWarps + warp] = variance;
  }
  __syncthreads();
  if (tid == 0) {
    float sum = warp_variance[0];
#pragma unroll
    for (int i = 1; i < kParts * kWarps; ++i) sum += warp_variance[i];
    inverse_rms = rsqrtf(sum / kHidden + p.rms_eps);
  }
  __syncthreads();
  const auto* gamma = reinterpret_cast<const Half8*>(p.rms_weight);
  auto* output = reinterpret_cast<Half8*>(p.reduced) + row * kVectors;
#pragma unroll
  for (int part = 0; part < kParts; ++part) {
    const int vector = tid + part * Threads;
    const Half8 weight = gamma[vector];
    Half8 normalized;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      normalized.data[i] = __float2half_rn(
          __half2float(values[part].data[i]) * inverse_rms *
          __half2float(weight.data[i]));
    output[vector] = normalized;
  }
}

template <bool HasPrevious, bool Normalize>
__global__ void hy4_f7_hc_bridge(Bridge p) {
  const auto grid = cooperative_groups::this_grid();
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int warp = tid / 32;
  __shared__ volatile float scratch[kFields][kThreads / 32];

  // One coalesced 128-column slice computes contributions to all eight
  // gates and the norm. Materialize the original post gate's FP16 cast.
  for (int tile = blockIdx.x; tile < p.rows * kChunks; tile += gridDim.x) {
    const int row = tile / kChunks;
    const int col = (tile % kChunks) * kThreads + tid;
    const int idx = row * kWidth + col;
    float x = __half2float(p.residual[idx]);
    if constexpr (HasPrevious) {
      const float y = __half2float(p.previous[row * kHidden + col % kHidden]);
      const float g = p.previous_post[row * kChannels + col / kHidden];
      const half rounded = __float2half_rn(__fadd_rn(x, __fmul_rn(g, y)));
      x = __half2float(rounded);
    }
    p.updated[idx] = __float2half_rn(x);
    float sq = warp_sum(__fmul_rn(x, x));
    if (lane == 0) scratch[0][warp] = sq;
#pragma unroll
    for (int field = 1; field < kFields; ++field) {
      float dot = warp_sum(__fmul_rn(x, p.weight[(field - 1) * kWidth + col]));
      if (lane == 0) scratch[field][warp] = dot;
    }
    __syncthreads();
    if (tid < kFields) {
      float total = scratch[tid][0];
#pragma unroll
      for (int w = 1; w < kThreads / 32; ++w) total += scratch[tid][w];
      p.partial[tid * (p.rows * kChunks) + tile] = total;
    }
    __syncthreads();
  }

  grid.sync();

  // Repeat the inexpensive final gate reduction for each 128-column
  // output tile. All cross-phase reads use L2-coherent loads on Volta.
  constexpr int kTiles = Normalize ? 1 : kHidden / kThreads;
  for (int tile = blockIdx.x; tile < p.rows * kTiles; tile += gridDim.x) {
    const int row = tile / kTiles;
    const int col = (tile % kTiles) * kThreads + tid;
    for (int field = warp; field < kFields; field += kThreads / 32) {
      const float* values = p.partial + (field * p.rows + row) * kChunks;
      float total = 0.f;
#pragma unroll
      for (int part = 0; part < kChunks; part += 32)
        total += __ldcg(values + part + lane);
      total = warp_sum(total);
      if (lane == 0) scratch[field][0] = total;
    }
    __syncthreads();
    const float rms = rsqrtf(scratch[0][0] / kWidth + p.norm_eps);
    if (tile % kTiles == 0 && tid < kChannels) {
      const float g = gate(scratch[1 + kChannels + tid][0], rms, p.scale[1],
                           p.base[kChannels + tid]);
      p.post[row * kChannels + tid] =
          __fadd_rn(__fmul_rn(p.magnitude, g), p.hc_eps);
    }
    if (tid < kChannels) {
      scratch[tid + 1][1] =
          gate(scratch[tid + 1][0], rms, p.scale[0], p.base[tid]) + p.hc_eps;
    }
    __syncthreads();
    if constexpr (Normalize) {
      mix_and_rms(p, row, scratch);
    } else {
      const half* mixed = p.updated;
      float result = 0.f;
  #pragma unroll
      for (int c = 0; c < kChannels; ++c) {
        const float x = load_half_cg(mixed + row * kWidth + c * kHidden + col);
        result = __fadd_rn(result, __fmul_rn(x, scratch[c + 1][1]));
      }
      p.reduced[row * kHidden + col] = __float2half_rn(result);
    }
    __syncthreads();
  }
}

template <bool HasPrevious>
__global__ void hy4_bridge_slice(Bridge p) {
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int warp = tid / 32;
  const int tile = blockIdx.x;
  __shared__ volatile float scratch[kFields][kThreads / 32];

  const int row = tile / kChunks;
  const int col = (tile % kChunks) * kThreads + tid;
  const int idx = row * kWidth + col;
  float x = __half2float(p.residual[idx]);
  if constexpr (HasPrevious) {
    const float y = __half2float(p.previous[row * kHidden + col % kHidden]);
    const float g = p.previous_post[row * kChannels + col / kHidden];
    const half rounded = __float2half_rn(__fadd_rn(x, __fmul_rn(g, y)));
    x = __half2float(rounded);
  }
  p.updated[idx] = __float2half_rn(x);
  float products[kFields];
  products[0] = __fmul_rn(x, x);
#pragma unroll
  for (int field = 1; field < kFields; ++field)
    products[field] = __fmul_rn(x, p.weight[(field - 1) * kWidth + col]);
  __syncwarp();
#pragma unroll
  for (int field = 0; field < kFields; ++field)
    products[field] = warp_sum(products[field]);
  if (lane == 0) {
#pragma unroll
    for (int field = 0; field < kFields; ++field)
      scratch[field][warp] = products[field];
  }
  __syncthreads();
  if (tid < kFields) {
    float total = scratch[tid][0];
#pragma unroll
    for (int w = 1; w < kThreads / 32; ++w) total += scratch[tid][w];
    p.partial[tid * (p.rows * kChunks) + tile] = total;
  }
}

template <bool Normalize, int Threads = kThreads>
__global__ void hy4_bridge_mix(Bridge p) {
  constexpr int kTiles = Normalize ? 1 : kHidden / kThreads;
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int warp = tid / 32;
  const int tile = blockIdx.x;
  __shared__ volatile float scratch[kFields][kThreads / 32];

  const int row = tile / kTiles;
  const int col = (tile % kTiles) * kThreads + tid;
  for (int field = warp; field < kFields; field += Threads / 32) {
    const float* values = p.partial + (field * p.rows + row) * kChunks;
    float total = 0.f;
#pragma unroll
    for (int part = 0; part < kChunks; part += 32)
      total += __ldcg(values + part + lane);
    __syncwarp();
    total = warp_sum(total);
    if (lane == 0) scratch[field][0] = total;
  }
  __syncthreads();
  const float rms = rsqrtf(scratch[0][0] / kWidth + p.norm_eps);
  if (tile % kTiles == 0 && tid < kChannels) {
    const float g = gate(scratch[1 + kChannels + tid][0], rms, p.scale[1],
                         p.base[kChannels + tid]);
    p.post[row * kChannels + tid] =
        __fadd_rn(__fmul_rn(p.magnitude, g), p.hc_eps);
  }
  if (tid < kChannels) {
    scratch[tid + 1][1] =
        gate(scratch[tid + 1][0], rms, p.scale[0], p.base[tid]) + p.hc_eps;
  }
  __syncthreads();
  if constexpr (Normalize) {
    mix_and_rms<Threads>(p, row, scratch);
  } else {
    const half* mixed = p.updated;
    float result = 0.f;
#pragma unroll
    for (int c = 0; c < kChannels; ++c) {
      const float x = load_half_cg(mixed + row * kWidth + c * kHidden + col);
      result = __fadd_rn(result, __fmul_rn(x, scratch[c + 1][1]));
    }
    p.reduced[row * kHidden + col] = __float2half_rn(result);
  }
}

bool matches(const at::Tensor& t, const at::Tensor& ref, at::ScalarType dtype,
             at::IntArrayRef shape) {
  return t.device() == ref.device() && t.is_contiguous() &&
         t.scalar_type() == dtype && t.sizes() == shape;
}

using Result = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

Result alloc_bridge(const at::Tensor& residual,
                    const std::optional<at::Tensor>&,
                    const std::optional<at::Tensor>&, const at::Tensor&,
                    const at::Tensor&, const at::Tensor&, double, double, double) {
  const auto rows = residual.size(0);
  return {at::empty({rows, kHidden}, residual.options()),
          at::empty({rows, kChannels}, residual.options().dtype(at::kFloat)),
          at::empty(residual.sizes(), residual.options())};
}

template <bool HasPrevious, bool Normalize>
void launch(Bridge p, int sms, bool split_big_mix) {
  if (split_big_mix) {
    constexpr int kMixThreads = Normalize ? kHidden / 8 : kThreads;
    constexpr int kTiles = Normalize ? 1 : kHidden / kThreads;
    const auto stream = at::cuda::getCurrentCUDAStream();
    hy4_bridge_slice<HasPrevious><<<p.rows * kChunks, kThreads, 0, stream>>>(p);
    hy4_bridge_mix<Normalize, kMixThreads>
        <<<p.rows * kTiles, kMixThreads, 0, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  auto kernel = hy4_f7_hc_bridge<HasPrevious, Normalize>;
  int active = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, kernel, kThreads, 0));
  TORCH_CHECK(active > 0, "F7 iHC bridge could not establish cooperative residency");
  void* args[] = {&p};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(kernel), std::min(sms * active, p.rows * kChunks),
      kThreads, args, 0, at::cuda::getCurrentCUDAStream()));
}

template <bool Normalize>
Result run_bridge(const at::Tensor& residual,
                 const std::optional<at::Tensor>& previous,
                 const std::optional<at::Tensor>& previous_post,
                 const at::Tensor& weight, const at::Tensor& scale,
                 const at::Tensor& base, double magnitude, double hc_eps,
                 double norm_eps, const at::Tensor& rms_weight, double rms_eps,
                 std::optional<bool> split_big_mix) {
  const int rows = residual.size(0);
  TORCH_CHECK(residual.is_cuda() && rows > 0 && rows <= 24 &&
              matches(residual, residual, at::kHalf, {rows, kChannels, kHidden}) &&
              matches(weight, residual, at::kFloat, {kGates, kWidth}) &&
              matches(scale, residual, at::kFloat, {2}) &&
              matches(base, residual, at::kFloat, {kGates}) &&
              previous.has_value() == previous_post.has_value() &&
              (!previous ||
               (matches(*previous, residual, at::kHalf, {rows, kHidden}) &&
                matches(*previous_post, residual, at::kFloat, {rows, kChannels}))),
              "F7 iHC bridge tensor contract");
  if constexpr (Normalize) {
    TORCH_CHECK(rows <= 8 && matches(rms_weight, residual, at::kHalf, {kHidden}) &&
                    rms_weight.storage_offset() % 8 == 0,
                "Hy4 fused bridge RMSNorm tensor contract");
  }
  const c10::cuda::CUDAGuard guard(residual.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0 && props->cooperativeLaunch,
              "F7 iHC bridge requires SM70 cooperative launch");
  auto [reduced, post, updated] =
      alloc_bridge(residual, previous, previous_post, weight, scale, base,
                   magnitude, hc_eps, norm_eps);
  auto partial = at::empty({kFields, rows, kChunks}, weight.options());
  Bridge params{
      reinterpret_cast<const half*>(residual.const_data_ptr<at::Half>()),
      previous ? reinterpret_cast<const half*>(previous->const_data_ptr<at::Half>())
               : nullptr,
      previous_post ? previous_post->const_data_ptr<float>() : nullptr,
      weight.const_data_ptr<float>(), scale.const_data_ptr<float>(),
      base.const_data_ptr<float>(),
      reinterpret_cast<half*>(updated.mutable_data_ptr<at::Half>()),
      reinterpret_cast<half*>(reduced.mutable_data_ptr<at::Half>()),
      post.mutable_data_ptr<float>(), partial.mutable_data_ptr<float>(), rows,
      static_cast<float>(norm_eps), static_cast<float>(hc_eps),
      static_cast<float>(magnitude),
      Normalize ? reinterpret_cast<const half*>(rms_weight.const_data_ptr<at::Half>())
                : nullptr,
      static_cast<float>(rms_eps)};
  const char* split = std::getenv("HY4_BRIDGE_SPLIT_BIG_MIX");
  const bool use_split =
      split_big_mix.value_or(split != nullptr && std::strcmp(split, "1") == 0);
  if (previous) launch<true, Normalize>(params, props->multiProcessorCount, use_split);
  else launch<false, Normalize>(params, props->multiProcessorCount, use_split);
  return {reduced, post, updated};
}

Result hc_bridge(const at::Tensor& residual,
                 const std::optional<at::Tensor>& previous,
                 const std::optional<at::Tensor>& previous_post,
                 const at::Tensor& weight, const at::Tensor& scale,
                 const at::Tensor& base, double magnitude, double hc_eps,
                 double norm_eps, std::optional<bool> split_big_mix) {
  return run_bridge<false>(residual, previous, previous_post, weight, scale, base,
                           magnitude, hc_eps, norm_eps, {}, 0., split_big_mix);
}

Result alloc_bridge_rms(const at::Tensor& residual,
                        const std::optional<at::Tensor>& previous,
                        const std::optional<at::Tensor>& previous_post,
                        const at::Tensor& weight, const at::Tensor& scale,
                        const at::Tensor& base, double magnitude, double hc_eps,
                        double norm_eps, const at::Tensor&, double,
                        std::optional<bool>) {
  return alloc_bridge(residual, previous, previous_post, weight, scale, base,
                      magnitude, hc_eps, norm_eps);
}

Result alloc_bridge_optional_split(
    const at::Tensor& residual, const std::optional<at::Tensor>& previous,
    const std::optional<at::Tensor>& previous_post, const at::Tensor& weight,
    const at::Tensor& scale, const at::Tensor& base, double magnitude,
    double hc_eps, double norm_eps, std::optional<bool>) {
  return alloc_bridge(residual, previous, previous_post, weight, scale, base,
                      magnitude, hc_eps, norm_eps);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(hy4_sm70, m) {
  m.def("nodes_hc_rms(Tensor residual, Tensor? previous, Tensor? previous_post,"
        " Tensor weight, Tensor scale, Tensor base, float magnitude, float hc_eps,"
        " float norm_eps, Tensor rms_weight, float rms_eps, bool? split_big_mix=None) -> (Tensor, Tensor, Tensor)");
  m.def("f7_hc_bridge(Tensor residual, Tensor? previous, Tensor? previous_post,"
        " Tensor weight, Tensor scale, Tensor base, float magnitude, float hc_eps,"
        " float norm_eps, bool? split_big_mix=None) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(hy4_sm70, CUDA, m) {
  m.impl("nodes_hc_rms", &run_bridge<true>);
  m.impl("f7_hc_bridge", &hc_bridge);
}

TORCH_LIBRARY_IMPL(hy4_sm70, Meta, m) {
  m.impl("nodes_hc_rms", &alloc_bridge_rms);
  m.impl("f7_hc_bridge", &alloc_bridge_optional_split);
}
