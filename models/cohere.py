# fmt: off
from typing import Optional

import torch
import torch.nn as nn
from .kernels import FusedLinearCrossEntropyFunction
from transformers.models.cohere.modeling_cohere import (
    CohereConfig,
    CohereRotaryEmbedding,
    CohereForCausalLM as OriginalCohereForCausalLM,
    CoherePreTrainedModel,
)


class CohereCheckpointedDecoder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, *args) -> torch.Tensor:
        ctx.args = args
        (attention_mask, position_ids, input_layernorm, self_attn, mlp) = args

        # Save the input tensor to CPU (pinned memory)
        if x.requires_grad:
            saved_x = torch.empty(
                x.size(), dtype=x.dtype, layout=x.layout, pin_memory=True
            )
            saved_x.copy_(x, non_blocking=True)
            ctx.save_for_backward(saved_x)

        # Forward pass
        with torch.no_grad():
            x_n = input_layernorm(x)
            x += self_attn(x_n, attention_mask, position_ids)
            return x + mlp(x_n)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if len(ctx.saved_tensors) == 0:
            return (None,) * (len(ctx.args) + 1)

        x: torch.Tensor = ctx.saved_tensors[0]
        x = x.cuda(non_blocking=True).detach()
        x.requires_grad = True
        (attention_mask, position_ids, input_layernorm, self_attn, mlp) = ctx.args

        # Backward pass
        # NOTE: this is more memory-efficient than the generic checkpointing implementation
        # because Cohere's decoder implementation allows gradients to flow independently.
        # Thus we can run backward() against (residual, attn, mlp) separately to avoid keeping
        # activations around for all three channels.
        x.grad = grad_output.detach()  # residual grad pass-through
        with torch.enable_grad():
            # Do backprop for self-attn/mlp separately to reduce peak memory -
            # total gradient will be accumulated in x.grad
            torch.autograd.backward(
                self_attn(input_layernorm(x), attention_mask, position_ids), grad_output
            )
            torch.autograd.backward(mlp(input_layernorm(x)), grad_output)

        return (x.grad,) + (None,) * len(ctx.args)


def rotate_half(x):
    # WARNING: Cohere's rotate_half is older/different than Llama's
    # [0, 1, 2, 3, 4, 5] -> [-1, 0, -3, 2, -5, 4]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Attention(nn.Module):
    def __init__(self, config: CohereConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = CohereRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,  # type: ignore
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # "ungroup" key-value heads
        kv_heads = self.num_key_value_heads
        k = k.view(bsz, q_len, kv_heads, 1, self.head_dim)
        v = v.view(bsz, q_len, kv_heads, 1, self.head_dim)
        if self.num_key_value_groups > 1:
            k = k.expand(bsz, q_len, kv_heads, self.num_key_value_groups, self.head_dim)
            v = v.expand(bsz, q_len, kv_heads, self.num_key_value_groups, self.head_dim)

        # RoPE + SDPA requires shape [b, n_h, s, h_d]
        # -> [b, s, n_h, h_d]
        q = q.reshape(bsz, q_len, -1, self.head_dim)
        k = k.reshape(bsz, q_len, -1, self.head_dim)
        v = v.reshape(bsz, q_len, -1, self.head_dim)

        # -> [b, n_h, s, h_d]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(hidden_states, position_ids)  # note: input is only used for type/device
        q, k = apply_rotary_pos_emb(q, k, cos.to(dtype=q.dtype), sin.to(dtype=q.dtype))

        # TODO: Actually handle attention_mask (may be 2D)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output)


class CohereDecoderLayer(nn.Module):
    def __init__(self, config: CohereConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Attention(config)
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

        self.mlp = LigerSwiGLUMLP(config)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        return CohereCheckpointedDecoder.apply(
            hidden_states,
            attention_mask,
            position_ids,
            self.input_layernorm,
            self.self_attn,
            self.mlp,
        )


class CohereModel(nn.Module):
    def __init__(self, config: CohereConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([CohereDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, attention_mask, position_ids)
        return self.norm(hidden_states)


class CohereForCausalLM(OriginalCohereForCausalLM):
    def __init__(self, config: CohereConfig):
        super(CoherePreTrainedModel, self).__init__(config)

        self.model = CohereModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.logit_scale = config.logit_scale
        self.tie_word_embeddings = config.tie_word_embeddings
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        if labels is not None:
            # Each hidden state is for the *next* label, so shift labels right
            shift_hidden_states = (
                hidden_states[..., :-1, :]
                .contiguous()
                .view(-1, self.config.hidden_size)
            )
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = FusedLinearCrossEntropyFunction.apply(
                (shift_hidden_states * self.logit_scale).to(
                    self.lm_head.weight.device, non_blocking=True
                ),
                self.lm_head.weight,
                shift_labels.to(self.lm_head.weight.device, non_blocking=True),
            )
            # Logits are not needed for training/evaluation
            return (loss, None)  # type: ignore
        else:
            return (self.lm_head(hidden_states) * self.logit_scale,)
