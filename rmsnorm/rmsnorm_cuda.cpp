#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor rmsnorm_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight);

// C++ interface

// TODO Add channels-last kernel
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous, channels-first")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor rmsnorm_forward(
    torch::Tensor input,
    torch::Tensor weights)
{
  CHECK_INPUT(input);
  CHECK_INPUT(weights);

  return rmsnorm_cuda_forward(input, weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward", &rmsnorm_forward, "rmsnorm forward (CUDA)");
}