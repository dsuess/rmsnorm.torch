#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


torch::Tensor rmsnorm_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights) {
  return input;
}