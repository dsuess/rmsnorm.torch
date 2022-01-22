from typing import Callable, Tuple, Type

import numpy as np
import pytest
import torch

from rmsnorm import RMSNorm, RMSNorm2


@pytest.fixture
def device() -> str:
    return "cuda"


RMS_IMPL = [RMSNorm, RMSNorm2]


@pytest.mark.parametrize("dtype_str", ["float16", "float32"])
@pytest.mark.parametrize("input_shape", [(16, 64, 512)])
@pytest.mark.parametrize("impl", RMS_IMPL)
def test_rmsnorm_benchmark(
    dtype_str: str,
    input_shape: Tuple[int, int, int],
    impl: Type[torch.nn.Module],
    device: str,
    benchmark: Callable,
):
    dtype = getattr(torch, dtype_str)
    x = 2 * torch.randn(*input_shape, dtype=dtype).to(device)
    _, _, hidden_size = input_shape
    module = impl(hidden_size=hidden_size).to(dtype).to(device)

    with torch.no_grad():
        benchmark(lambda: module(x))


@pytest.mark.parametrize("dtype_str", ["float16", "float32"])
@pytest.mark.parametrize("input_shape", [(16, 64, 512)])
@pytest.mark.parametrize("impl", RMS_IMPL)
def test_rmsnorm(
    dtype_str: str,
    input_shape: Tuple[int, int, int],
    impl: Type[torch.nn.Module],
    device: str,
):
    dtype = getattr(torch, dtype_str)
    x = 2 * torch.randn(*input_shape, dtype=dtype).to(device)
    *shape, hidden_size = input_shape
    with torch.no_grad():
        module = impl(hidden_size=hidden_size).to(dtype).to(device)
        y = module(x).cpu().numpy()

    np.testing.assert_array_almost_equal(np.std(y, axis=-1), np.ones(shape), decimal=2)
