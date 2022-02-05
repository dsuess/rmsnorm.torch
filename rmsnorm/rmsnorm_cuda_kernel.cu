#include <vector>
#include <iostream>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cublas_helpers.h"

__global__ void varianceScaleInput(half *inputs, half *variance, half *channel_weights, int num_elems, int num_channels)
{
  for (int j = blockIdx.x * blockDim.x + threadIdx.x;
       j < num_elems;
       j += blockDim.x * gridDim.x)
  {
    half *inputs_j = inputs + j * num_channels;
    half sigma = hrsqrt(variance[j]);

    for (int i = blockIdx.y * blockDim.y + threadIdx.y;
         i < num_channels;
         i += blockDim.y * gridDim.y)
    {
      const auto fac = __hfma(channel_weights[i], sigma, __float2half(0.0));
      inputs_j[i] = __hfma(inputs_j[i], fac, __float2half(0.0));
    }
  }
}

__global__ void varianceScaleInput(float *inputs, float *variance, float *channel_weights, int num_elems, int num_channels)
{
  for (int j = blockIdx.x * blockDim.x + threadIdx.x;
       j < num_elems;
       j += blockDim.x * gridDim.x)
  {
    float *inputs_j = inputs + j * num_channels;
    float sigma = rsqrt(variance[j]);

    for (int i = blockIdx.y * blockDim.y + threadIdx.y;
         i < num_channels;
         i += blockDim.y * gridDim.y)
    {
      inputs_j[i] *= channel_weights[i] * sigma;
    }
  }
}

torch::Tensor rmsnorm_cuda_forward(
    torch::Tensor inputs,
    torch::Tensor weights)
{
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cublasSetStream(handle, stream);

  const auto batch_size = inputs.size(0);
  const auto seq_len = inputs.size(1);
  const auto embed_dim = inputs.size(2);
  const auto vector_step = batch_size * seq_len;
  const float alpha = (1.0 / embed_dim);
  const float beta = (1e-6);

  if (inputs.dtype() == torch::kFloat16)
  {
    // TODO experiment with CUDA-allocated strided memory
    // TODO Experiment with half2
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(inputs.device()).requires_grad(false);
    auto channel_variance = torch::zeros({batch_size, seq_len, 1}, options);

    TORCH_CUDABLAS_CHECK_WORKAROUND(
        cublasGemmStridedBatchedEx(
            handle,                                           // handle
            CUBLAS_OP_T,                                      // transa
            CUBLAS_OP_N,                                      // transb
            1,                                                // m
            1,                                                // n
            embed_dim,                                        // k
            static_cast<const void *>(&alpha),                // alpha
            static_cast<const void *>(inputs.data_ptr()),     // A
            CUDA_R_16F,                                       // dtype(A);
            embed_dim,                                        // lda
            embed_dim,                                        // strideA
            static_cast<const void *>(inputs.data_ptr()),     // B
            CUDA_R_16F,                                       // dtype(B)
            embed_dim,                                        // ldb
            embed_dim,                                        // strideB
            static_cast<const void *>(&beta),                 // beta
            static_cast<void *>(channel_variance.data_ptr()), // C
            CUDA_R_16F,                                       //dtype(C)
            1,                                                // ldc
            1,                                                // strideC
            vector_step,                                      // batchCount
            CUBLAS_COMPUTE_32F,                               // computeType
            CUBLAS_GEMM_DEFAULT_TENSOR_OP)                    // algo
    );
    const dim3 block_dim(16, 64, 1);
    const dim3 grid_dim(16, 16, 1);

    varianceScaleInput<<<grid_dim, block_dim>>>(
        static_cast<half *>(inputs.data_ptr()),
        static_cast<half *>(channel_variance.data_ptr()),
        static_cast<half *>(weights.data_ptr()),
        vector_step,
        embed_dim);
  }
  else if (inputs.dtype() == torch::kFloat32)
  {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(inputs.device()).requires_grad(false);
    auto channel_variance = torch::zeros({batch_size, seq_len, 1}, options);

    TORCH_CUDABLAS_CHECK_WORKAROUND(
        cublasGemmStridedBatchedEx(
            handle,                                           // handle
            CUBLAS_OP_T,                                      // transa
            CUBLAS_OP_N,                                      // transb
            1,                                                // m
            1,                                                // n
            embed_dim,                                        // k
            static_cast<const void *>(&alpha),                // alpha
            static_cast<const void *>(inputs.data_ptr()),     // A
            CUDA_R_32F,                                       // dtype(A);
            embed_dim,                                        // lda
            embed_dim,                                        // strideA
            static_cast<const void *>(inputs.data_ptr()),     // B
            CUDA_R_32F,                                       // dtype(B)
            embed_dim,                                        // ldb
            embed_dim,                                        // strideB
            static_cast<const void *>(&beta),                 // beta
            static_cast<void *>(channel_variance.data_ptr()), // C
            CUDA_R_32F,                                       //dtype(C)
            1,                                                // ldc
            1,                                                // strideC
            vector_step,                                      // batchCount
            CUBLAS_COMPUTE_32F,                               // computeType
            CUBLAS_GEMM_DEFAULT_TENSOR_OP)                    // algo
    );

    const dim3 block_dim(16, 64, 1);
    const dim3 grid_dim(16, 16, 1);

    varianceScaleInput<<<grid_dim, block_dim>>>(
        static_cast<float *>(inputs.data_ptr()),
        static_cast<float *>(channel_variance.data_ptr()),
        static_cast<float *>(weights.data_ptr()),
        vector_step,
        embed_dim);
  }
  else
  {
    throw std::runtime_error("Passed unsupported dtype.");
  }
  return inputs;
}