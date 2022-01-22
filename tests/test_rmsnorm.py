import torch
import pytest
from rmsnorm import RMSNorm
from typing import Callable, Tuple


@pytest.fixture
def device() -> str:
    return "cuda"


@pytest.mark.parametrize("dtype_str", ["float16", "float32"])
@pytest.mark.parametrize("input_shape", [(16, 64, 512)])
def test_t5_layer_norm(
    dtype_str: str, input_shape: Tuple[int, int, int], device: str, benchmark: Callable
):
    dtype = getattr(torch, dtype_str)
    x = torch.randn(*input_shape, dtype=dtype).to(device)
    _, _, hidden_size = input_shape
    module = RMSNorm(hidden_size=hidden_size).to(dtype).to(device)

    benchmark(lambda: module(x))
