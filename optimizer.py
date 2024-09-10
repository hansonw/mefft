from typing import Iterable, Type

import torch
from torch.optim.optimizer import Optimizer
from bitsandbytes.optim import PagedAdamW8bit
from numba.cuda import current_context
from numba.cuda.cudadrv import devicearray, driver
from numba.cuda.api_util import prepare_shape_strides_dtype
import numpy as np
import torch

dtype_map: dict[torch.dtype, np.dtype] = {
    torch.bfloat16: np.dtype("float16"),
    torch.float16: np.dtype("float16"),
    torch.float32: np.dtype("float32"),
    torch.long: np.dtype("int64"),
    torch.uint8: np.dtype("uint8"),
    torch.int8: np.dtype("int8"),
}


def mapped_tensor(shape: tuple[int, ...], dtype: torch.dtype):
    np_dtype = dtype_map[dtype]
    shape, strides, _ = prepare_shape_strides_dtype(
        shape, strides=None, dtype=np_dtype, order="C"
    )
    bytesize = driver.memory_size_from_info(shape, strides, np_dtype.itemsize)
    buffer = current_context().memhostalloc(bytesize, mapped=True)
    ndary = devicearray.DeviceNDArray(
        shape=shape, dtype=np_dtype, gpu_data=buffer, strides=strides
    )
    return torch.as_tensor(ndary, device="cuda", dtype=dtype)


class MappedAdamW8bit(PagedAdamW8bit):
    @torch.no_grad()
    def step(self, closure=None):
        """
        Same as the parent implementation, but **without prefetch/synchronize calls**.
        We will do this globally in `InterleavedOptimizer.step`
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    self.init_state(group, p, gindex, pindex)
                self.update_step(group, p, gindex, pindex)

        return loss

    def get_state_buffer(self, p, dtype=torch.float32):
        return mapped_tensor(p.shape, dtype)


class InterleavedOptimizer(Optimizer):
    """
    PyTorch optimizer that interleaves step() with the backward pass.
    IMPORTANT: Not compatible with gradient accumulation (yet).
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        optimizer_class: Type[Optimizer] = MappedAdamW8bit,
        **kwargs,
    ) -> None:
        self.optim_dict = dict()

        params_list = list(params)
        super().__init__(params_list, {})

        def backward_hook(p_cuda):
            if p_cuda.grad is not None:
                optim = self.optim_dict[p_cuda]
                optim.step()
                p_cuda.grad = None

        for p_cuda in params_list:
            if p_cuda.requires_grad:
                p_cuda.register_post_accumulate_grad_hook(backward_hook)
                self.optim_dict[p_cuda] = optimizer_class([p_cuda], **kwargs)

    @property
    def param_groups(self):
        return sum([optim.param_groups for optim in self.optim_dict.values()], [])

    @param_groups.setter
    def param_groups(self, _param_groups):
        # Setting param_groups is stubbed out
        return

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        # Make sure all parameters are updated.
        torch.cuda.synchronize()
        return loss

    def zero_grad(self, set_to_none=True):
        assert set_to_none, "Expected set_to_none=True"
        for p_cuda in self.optim_dict.keys():
            p_cuda.grad = None

    def state_dict(self):
        return [optim.state_dict() for optim in self.optim_dict.values()]

    def load_state_dict(self, state_dict):
        for optim, optim_state_dict in zip(self.optim_dict.values(), state_dict):
            optim.load_state_dict(optim_state_dict)
