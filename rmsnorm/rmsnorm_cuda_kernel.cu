 #include <vector>
#include <iostream>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cublas_helpers.h"


torch::Tensor rmsnorm_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights)
{
  // There is no reason to use more than one stream as every kernel is
  // sequentially dependent
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cublasSetStream(handle, stream);

  const auto batch_size = input.size(0);
  const auto seq_len = input.size(1);
  const auto embed_dim = input.size(2);
  const auto vector_step = batch_size * seq_len;
  const float alpha = 1.0 / embed_dim;
  const float beta = 0.0;

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(at::kCUDA).requires_grad(false);
  auto channel_variance = torch::zeros({batch_size, seq_len}, options);

  TORCH_CUDABLAS_CHECK_WORKAROUND(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  TORCH_CUDABLAS_CHECK_WORKAROUND(
    cublasGemmStridedBatchedEx(
      handle,  // handle
      CUBLAS_OP_T,  // transa
      CUBLAS_OP_N,  // transb
      1,  // m
      1, // n
      embed_dim, // k
      static_cast<const void *>(&alpha), // alpha
      static_cast<const void *>(input.data_ptr()), // A
      CUDA_R_32F, // dtype(A)
      embed_dim,  // lda
      1,          // strideA
      static_cast<const void *>(input.data_ptr()),  // B
      CUDA_R_32F, // dtype(B)
      embed_dim, // ldb
      1, // strideB
      static_cast<const void *>(&beta),  // beta
      static_cast<void *>(channel_variance.data_ptr()), // C
      CUDA_R_32F, //dtype(C)
      1, // ldc
      1, // strideC
      vector_step, // batchCount
      CUBLAS_COMPUTE_32F, // computeType
      CUBLAS_GEMM_DEFAULT) // algo
  );

  return input;
}