import torch
import triton
import triton.language as tl


# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576 https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
BLOCK_SIZE = 65536 // 2


@triton.jit
def liger_cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore_ptr,
    ignore_index,
    label_smoothing: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This kernel computes both cross entropy loss and the gradient of the input.
    We only consider hard label + mean reduction for now. Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.

    Parameters:
    X_ptr: Pointer to input tensor.
    X_stride (int): The stride of the input tensor.
    Y_ptr: Pointer to target tensor.
    Y_stride (int): The stride of the target tensor.
    loss_ptr: Pointer to tensor to store the loss.
    loss_stride (int): The stride of the loss tensor.
    n_cols (int): The number of columns in the input tensor.
    n_non_ignore_ptr: Pointer to the number of non-ignored elements in the batch.
    ignore_index (int): The index to ignore in the target.
    label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
    BLOCK_SIZE (int): The block size for Triton operations.
    """

    # https://github.com/triton-lang/triton/issues/1058
    # If B*T*V is too large, program_id * stride will overflow out of int32, so we convert to int64
    program_id = tl.program_id(0).to(tl.int64)

    # 1. Load Y_ptr first because if the target is ignore_index, we can return right away
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    # 2. locate the start index
    X_ptr += program_id * X_stride

    if y == ignore_index:
        # set all X_ptr as 0
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    loss_ptr += program_id * loss_stride

    # Online softmax: 2 loads + 1 store (compared with 3 loads + 1 store for the safe softmax)
    # Refer to Algorithm 3 in the paper: https://arxiv.org/pdf/1805.02867

    # 3. [Online softmax] first pass: find max + sum
    m = float("-inf")  # m is the max value. use the notation from the paper
    d = 0.0  # d is the sum. use the notation from the paper
    y = tl.load(Y_ptr)
    ori_X_y = tl.load(X_ptr + y).cast(
        tl.float32
    )  # we need to store the original value of X_y for the loss calculation

    # Label smoothing is a general case of normal cross entropy
    # See the full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issue-2503665310
    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")
        ).cast(tl.float32)
        block_max = tl.max(X_block)
        if label_smoothing > 0:
            # scale X beforehand to avoid overflow
            scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block, 0.0))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    # 4. [Online softmax] second pass: calculate the gradients
    # dx_y = (softmax(x_y) - 1) / N
    # dx_i = softmax(x_i) / N, i != y
    # N is the number of non ignored elements in the batch
    # For label smoothing:
    # dx_i = (softmax(x_y) - label_smoothing / V) / N, V = n_cols, i != y
    # dx_y = (softmax(x_y) - label_smoothing / V - (1 - label_smoothing)) / N
    #      = dx_i - (1 - label_smoothing) / N
    n_non_ignore = tl.load(n_non_ignore_ptr)
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")
        ).cast(tl.float32)
        X_block = (tl.exp(X_block - m) / d - eps) / (n_non_ignore)
        tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)

    # We need tl.debug_barrier() to ensure the new result of X_ptr is written as mentioned in
    # https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/ops/cross_entropy.py#L34
    tl.debug_barrier()

    # 5. Calculate the loss

    # loss = log (softmax(X_y)) = log ((e ^ (X_y - max(X)) / sum(e ^ (X - max(X))))
    #      = (X_y - max(X)) - log(sum(e ^ (X - max(X))))
    # sum(e ^ (X - max(X))) must >= 1 because the max term is e ^ 0 = 1
    # So we can safely calculate log (softmax(X_y)) without overflow
    loss = -(ori_X_y - m - tl.log(d))

    # Orginal loss = H(q, p),  with label smoothing regularization = H(q', p) and (label_smoothing / V) = eps
    # H(q', p) = (1 - label_smoothing) * H(q, p) + label_smoothing * H(u, p)
    #          = (1 - label_smoothing) * H(q, p) + eps * sum(logsoftmax(x_i))
    # By using m (global max of xi) and d (sum of e^(xi-m)), we can simplify as:
    #          = (1 - label_smoothing) * H(q, p) + (-sum(x_i * eps) + label_smoothing * (m + logd))
    # Refer to H(q', p) in section 7 of the paper: https://arxiv.org/pdf/1512.00567
    # pytorch: https://github.com/pytorch/pytorch/blob/2981534f54d49fa3a9755c9b0855e7929c2527f0/aten/src/ATen/native/LossNLL.cpp#L516
    # See full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issuecomment-2333753087
    if label_smoothing > 0:
        smooth_loss = scaled_x_sum + label_smoothing * (m + tl.log(d))
        loss = loss * (1 - label_smoothing) + smooth_loss

    # 6. Specially handle the i==y case where `dx_y = (softmax(x_y) - (1 - label_smoothing) / N`
    X_y = tl.load(X_ptr + y).cast(tl.float32)
    X_y += -(1 - label_smoothing) / (n_non_ignore)

    tl.store(loss_ptr, loss)
    tl.store(X_ptr + y, X_y)


def fused_linear_cross_entropy_forward(
    _input, weight, target, bias=None, ignore_index=-100, softcap=0.0
):
    device = _input.device

    # inputs have shape: BT x H
    # materialized activations will have shape: BT x V
    # the increase in memory = BT x V
    # reduction can be achieved by partitioning the number of tokens BT into smaller chunks.
    # for ex: if we were to achieve the same memory consumption as BT x H, then the chunk size should be:
    BT, H = _input.shape
    V = weight.shape[0]

    # Choose the max. chunk size that keeps the C * V logit matrix < 1GB @ float16
    chunk_size = 64 * triton.cdiv(536_870_912 // V, 64)
    num_chunks = triton.cdiv(BT, chunk_size)  # (BT + chunk_size - 1) // chunk_size

    grad_weight = (
        torch.zeros_like(weight, device=device) if weight.requires_grad else None
    )
    grad_input = torch.zeros_like(_input, device=device)
    grad_bias = torch.zeros_like(bias, device=device) if bias is not None else None
    # we use fp32 for loss accumulator
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    # Allocate logits_chunk once and re-use it across iterations
    logits_chunk = torch.zeros(chunk_size, V, dtype=_input.dtype, device=device)

    total_n_non_ignore = (target != ignore_index).sum()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]  # chunk_size x H

        if end_idx - start_idx != logits_chunk.shape[0]:
            logits_chunk.resize_(end_idx - start_idx, V)
        torch.matmul(_input_chunk, weight.t(), out=logits_chunk)
        if bias is not None:
            logits_chunk += bias
        if softcap > 0:
            # TODO: Fuse this into the triton kernel. Note it's not really the bottleneck
            tanh_chunk = torch.tanh(logits_chunk / softcap)
            logits_chunk = tanh_chunk * softcap
        target_chunk = target[start_idx:end_idx].contiguous()  # chunk_size,

        n_rows = logits_chunk.shape[0]

        # unreduced loss
        loss_1d_slice = loss_1d[start_idx:end_idx]  # chunk_size,

        # Here we calculate the gradient of logits_chunk in place so we can save memory.
        n_non_ignore = (target_chunk != ignore_index).sum()
        liger_cross_entropy_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),  # always 1
            loss_ptr=loss_1d_slice,
            loss_stride=loss_1d_slice.stride(-1),  # always 1
            n_cols=V,
            n_non_ignore_ptr=n_non_ignore,
            ignore_index=ignore_index,
            label_smoothing=0.0,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

        if softcap > 0:
            # d/dx t * tanh(x/t) = 1 - tanh(x/t)^2
            logits_chunk *= 1 - tanh_chunk.pow(2)

        # gradient of logits_chunk is computed in-place by the above triton kernel and is of shape: chunk_size x V
        # thus grad_input[start_idx: end_idx] should be of shape: chunk_size x H
        # additionally, since we are chunking the inputs, observe that the loss and gradients are calculated only
        # on `n_non_ignore` tokens. However, the gradient of the input should be calculated for all tokens.
        # Thus, we need an additional scaling factor of (n_non_ignore/total_n_non_ignore) to scale the gradients.
        grad_input_chunk = grad_input[start_idx:end_idx]
        torch.addmm(
            input=grad_input_chunk,
            mat1=logits_chunk,
            mat2=weight,
            out=grad_input_chunk,
            alpha=n_non_ignore / total_n_non_ignore,
            beta=0.0,  # grad_input_chunk is already 0
        )

        if grad_weight is not None:
            torch.addmm(
                input=grad_weight,
                mat1=logits_chunk.t(),
                mat2=_input_chunk,
                out=grad_weight,
                alpha=n_non_ignore / total_n_non_ignore,
                beta=1.0,
            )

        if grad_bias is not None:
            torch.add(
                input=grad_bias,
                other=logits_chunk.sum(dim=0),
                out=grad_bias,
                alpha=n_non_ignore / total_n_non_ignore,
            )

    loss = torch.sum(loss_1d) / total_n_non_ignore
    return loss, grad_input, grad_weight, grad_bias


class FusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _input, weight, target, bias=None, ignore_index=-100, softcap=0.0):
        """
        Fusing the last linear layer with cross-entropy loss
            Reference: https://github.com/mgmalek/efficient_cross_entropy

        Handle the forward and backward pass of the final linear layer via cross-entropy loss by avoiding
        the materialization of the large logits tensor. Since Cross Entropy Loss is the last layer, we can
        compute the gradient at the forward pass. By doing so, we don't have to store the _input and target
        for the backward pass.

        _input: (B*T, H) where B is batch size, T is sequence length, H is hidden dimension.
        target: (B*T) where each value is in [0, V-1]
        weight: (V, H) where V is the number of classes
        bias: (V) where V is the number of classes
        ignore_index: the index to ignore in the target
        """
        loss, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_forward(
            _input, weight, target, bias, ignore_index, softcap
        )
        # downcast to dtype and store for backward
        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if grad_bias is not None else None,
        )
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (grad_input, grad_weight, grad_bias) = ctx.saved_tensors
        # Assume grad_output is always 1.0 (training code directly used loss.backward())
        # grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
        #     grad_output, grad_input, grad_weight, grad_bias
        # )
        return (grad_input, grad_weight, None, grad_bias, None, None)
