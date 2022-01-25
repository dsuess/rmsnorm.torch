#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

const int MAX_BLOCK_SIZE = 512;

__host__ __forceinline__ int prev_power_of_two(unsigned int n)
{
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return n - (n >> 1);
}

torch::Tensor rmsnorm_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights)
{
  const auto batch_size = input.size(0);
  const auto seq_len = input.size(1);
  const auto num_channels = input.size(2);
  auto stream = at::cuda::getCurrentCUDAStream();

  // Allocate batch-variance
  float *channel_var;
  {
    const auto size = sizeof(float) * batch_size * seq_len;
    cudaMallocAsync((void **)&channel_var, size, stream);
  }

  auto out = torch::empty_like(input);
  int block_x = max(32, min(MAX_BLOCK_SIZE, prev_power_of_two(num_channels) / 4));
  int block_y = max(1, min(MAX_BLOCK_SIZE / block_x, prev_power_of_two(batch_size) / 4));
  const dim3 block(block_x, block_y);

  switch (input.scalar_type())
  {
  case at::ScalarType::Float:
    //to sth
    break;
  case at::ScalarType::Half:
    //to sth
    break;
  default:
    cudaFreeAsync(channel_var, stream);
    throw std::runtime_error("Input-dtype not supported");
  }

  cudaFreeAsync(channel_var, stream);
  return out;
}