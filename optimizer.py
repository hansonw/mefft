# pyright: reportMissingImports=false
from collections import abc, defaultdict
from itertools import chain
import torch
import numpy as np
from torch.optim.optimizer import Optimizer
from typing import Any, Iterable, Type

try:
    # To enable running on Modal
    from bitsandbytes.optim import PagedAdamW8bit
except ImportError:
    PagedAdamW8bit: Any = object


dtype_map: dict[torch.dtype, np.dtype] = {
    torch.float16: np.dtype("float16"),
    torch.float32: np.dtype("float32"),
    torch.long: np.dtype("int64"),
    torch.uint8: np.dtype("uint8"),
    torch.int8: np.dtype("int8"),
}


def mapped_tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """
    A hacky approach for allocating mapped CUDA memory (lives on CPU, but accessible on GPU)
    and then creating a PyTorch tensor from it.

    This could be streamlined with C code that calls cudaHostAlloc and then torch::from_blob.
    """
    from numba.cuda import current_context
    from numba.cuda.cudadrv import devicearray, driver
    from numba.cuda.api_util import prepare_shape_strides_dtype

    if dtype == torch.bfloat16:
        # bfloat16 is not supported by numpy, so allocate float16 and reinterpret it
        return mapped_tensor(shape, torch.float16).view(dtype=torch.bfloat16)

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
    """
    Modification to PagedAdamW8bit that uses mapped memory instead of page memory.
    The paged memory implementation triggers a lot of CUDA synchronizations/page faults.
    """

    @torch.no_grad()
    def step(self, closure=None):
        """
        Same as the parent implementation, but **without prefetch/synchronize calls**.
        It's not needed for mapped memory.
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

    def load_state_dict(self, state_dict):
        """
        This is generally just copied from the base implementation, except that it doesn't
        correctly load state into mapped memory (TODO: upstream START CHANGE -> END CHANGE)
        """

        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict["param_groups"]

        if len(groups) != len(saved_groups):
            raise ValueError(
                "loaded state dict has a different number of parameter groups"
            )
        param_lens = (len(g["params"]) for g in groups)
        saved_lens = (len(g["params"]) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                "loaded state dict contains a parameter group that doesn't match the size of optimizer's group",
            )

        # Update the state
        id_map = {
            old_id: p
            for old_id, p in zip(
                chain.from_iterable(g["params"] for g in saved_groups),
                chain.from_iterable(g["params"] for g in groups),
            )
        }

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if param.is_floating_point() and value.dtype != torch.uint8:
                    value = value.to(param.dtype)
                return value
            elif isinstance(value, dict):
                for k, v in value.items():
                    if k in self.non_castable_tensor_keys:
                        # START CHANGE: make sure the tensor gets loaded into mapped memory
                        buffer = self.get_state_buffer(v, v.dtype)
                        buffer.copy_(v)
                        value[k] = buffer
                        # END CHANGE
                    else:
                        value[k] = cast(param, v)

                return value
            elif isinstance(value, abc.Iterable):
                return type(value)(cast(param, v) for v in value)  # type: ignore
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)  # type: ignore
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group["params"] = group["params"]
            return new_group

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({"state": self.state, "param_groups": param_groups})

    def zero_grad(self, set_to_none=True):
        # Force set_to_none=False to avoid clearing mapped gradients
        super().zero_grad(set_to_none=False)


def initialize_mapped_gradients(model: torch.nn.Module):
    """
    Offloads all model gradients to CPU as mapped tensors.
    Accumulated gradients will be transparently saved back to CPU.
    """
    for p in model.parameters():
        if p.requires_grad:
            p.grad = mapped_tensor(p.shape, p.dtype)


class InterleavedOptimizer(Optimizer):
    """
    PyTorch optimizer that interleaves step() with the backward pass.
    IMPORTANT: Not compatible with gradient accumulation.
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
