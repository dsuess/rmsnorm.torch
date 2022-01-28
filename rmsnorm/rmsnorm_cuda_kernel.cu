#include <vector>
#include <iostream>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

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
  const float alpha = 1.0;
  const float beta = 0.0;

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(at::kCUDA).requires_grad(false);
  auto channel_variance = torch::zeros({batch_size, seq_len}, options);

  // TODO THis should use TORCH_CUDABLAS_CHECK
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  // Input Linear Fwd
  auto status = cublasDotEx(
      handle, embed_dim,
      static_cast<void *>(input.data_ptr()), CUDA_R_16F, vector_step,
      static_cast<void *>(input.data_ptr()), CUDA_R_16F, vector_step,
      static_cast<void *>(channel_variance.data_ptr()),
      CUDA_R_32F, CUDA_R_32F);

  std::cout << status << std::endl;
  std::cout << channel_variance << std::endl;

  // switch (input.scalar_type())
  // {
  // case at::ScalarType::Float:
  //   //to sth
  //   break;
  // case at::ScalarType::Half:
  //   //to sth
  //   break;
  // default:
  //   cudaFreeAsync(channel_var, stream);
  //   throw std:out:rououttuntime_error("Input-dtype not supported");
  // }

  //cudaFreeAsync(channel_var, stream);
  return input;
}