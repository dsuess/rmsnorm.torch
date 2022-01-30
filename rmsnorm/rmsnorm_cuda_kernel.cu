#include <vector>
#include <iostream>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cublas_helpers.h"

__global__ void varianceScaleInput(float *inputs, float *variance, float *channel_weights, int num_elems, int num_channels)
{
  for (long j = blockIdx.x * blockDim.x + threadIdx.x;
       j < num_elems;
       j += blockDim.x * gridDim.x)
  {
    float *row_d_matrix = inputs + j * num_channels;
    float sigma = rsqrt(variance[j] + 1e-6);

    for (long i = blockIdx.y * blockDim.y + threadIdx.y;
         i < num_channels;
         i += blockDim.y * gridDim.y)
    {
      row_d_matrix[i] *= channel_weights[i] * sigma;
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
  const float alpha = 1.0 / embed_dim;
  const float beta = 0.0;

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(at::kCUDA).requires_grad(false);
  auto channel_variance = torch::zeros({batch_size, seq_len, 1}, options);

  // TORCH_CUDABLAS_CHECK_WORKAROUND(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
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
          CUBLAS_GEMM_DEFAULT)                              // algo
  );

  varianceScaleInput<<<1, 1>>>(
      static_cast<float *>(inputs.data_ptr()),
      static_cast<float *>(channel_variance.data_ptr()),
      static_cast<float *>(weights.data_ptr()),
      vector_step,
      embed_dim);
  return inputs;
}