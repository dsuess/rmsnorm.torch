from pathlib import Path

import torch
from torch import nn
from torch.utils.cpp_extension import load

basepath = Path(__file__).parent
rmsnorm_cuda = load(name='rmsnorm_cuda', sources=[basepath / 'rmsnorm_cuda.cpp', basepath / 'rmsnorm_cuda_kernel.cu'])


class RMSNorm(nn.Module):
    # https://github.com/huggingface/transformers/blob/3fc221d077a789bbb0c69bf81cff3d976604ed46/src/transformers/models/t5/modeling_t5.py#L237
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class RMSNorm2(RMSNorm):
    def forward(self, hidden_states):
        return rmsnorm_cuda.forward(hidden_states, self.weight)
