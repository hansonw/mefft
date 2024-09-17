import os
import torch


class OffloadCheckpointer(torch.autograd.Function):
    """
    Uses pinned memory to efficiently offload only the input tensor to CPU.
    Note that the backward pass will run the full forward pass again.
    """

    # Use a separate stream for offloading copies
    stream = torch.cuda.Stream()

    @staticmethod
    def forward(ctx, x: torch.Tensor, module: torch.nn.Module, *args) -> torch.Tensor:
        # Save the input tensor to CPU (pinned memory)
        if x.requires_grad:
            saved_x = torch.empty(
                x.size(),
                dtype=x.dtype,
                layout=x.layout,
                pin_memory=not os.environ.get("DISABLE_CHECKPOINT_MEMORY_PINNING"),
            )
            with torch.cuda.stream(OffloadCheckpointer.stream):
                saved_x.copy_(x, non_blocking=True)
            ctx.save_for_backward(saved_x)

        # Forward pass
        with torch.no_grad():
            output = module(x, *args)

        ctx.module = module
        ctx.args = args
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if len(ctx.saved_tensors) == 0:
            return (None, None) + (None,) * len(ctx.args)

        x: torch.Tensor = ctx.saved_tensors[0]
        OffloadCheckpointer.stream.synchronize()
        x = x.cuda(non_blocking=True).detach()
        x.requires_grad = True

        with torch.enable_grad():
            output = ctx.module(x, *ctx.args)

        torch.autograd.backward(output, grad_output)
        return (x.grad, None) + (None,) * len(ctx.args)
