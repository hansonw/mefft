# fmt: off
from typing import Optional

import torch
import torch.nn as nn
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from transformers import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding, MistralForCausalLM as OriginalMistralForCausalLM, MistralPreTrainedModel

from .kernels import FusedLinearCrossEntropyFunction


class Attention(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = MistralRotaryEmbedding(
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
        q, k = liger_rotary_pos_emb(q, k, cos, sin)  # type: ignore

        # TODO: Actually handle attention_mask (may be 2D)
        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output)


class CheckpointedDecoder(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        attention_mask,
        position_ids,
        self_attn,
        mlp,
        input_layernorm,
        post_attention_layernorm,
    ) -> torch.Tensor:
        # Save the input tensor to CPU (pinned memory)
        if x.requires_grad:
            saved_x = torch.empty(x.size(), dtype=x.dtype, layout=x.layout, pin_memory=True)
            saved_x.copy_(x, non_blocking=True)
            ctx.save_for_backward(saved_x)

        # Forward pass
        with torch.no_grad():
            output = x + self_attn(input_layernorm(x), attention_mask, position_ids)
            output = output + mlp(post_attention_layernorm(output))

        ctx.args = (attention_mask, position_ids, self_attn, mlp, input_layernorm, post_attention_layernorm)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if len(ctx.saved_tensors) == 0:
            return (None,) * (1 + len(ctx.args))

        attention_mask, position_ids, self_attn, mlp, input_layernorm, post_attention_layernorm = ctx.args

        x: torch.Tensor = ctx.saved_tensors[0]
        x = x.cuda(non_blocking=True).detach()
        x.requires_grad = True

        with torch.enable_grad():
            # Forward pass (again)
            # TODO: if we save the output post-attention, maybe we could do
            # output2 = output1 + mlp
            # backward(output2, dY)
            # output1 = x + self_attn (can happen in parallel with backward?)
            # backward(output1, output2.grad)
            output = x + self_attn(input_layernorm(x), attention_mask, position_ids)
            output = output + mlp(post_attention_layernorm(output))

        torch.autograd.backward(output, grad_output)
        return (x.grad,) + (None,) * (len(ctx.args))


class DecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
        self.mlp = LigerSwiGLUMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        # The intermediate actions inside the decoder are huge!
        return CheckpointedDecoder.apply(
            x,
            attention_mask,
            position_ids,
            self.self_attn,
            self.mlp,
            self.input_layernorm,
            self.post_attention_layernorm,
        )


class MistralModel(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, attention_mask, position_ids)
        return self.norm(hidden_states)


class MistralForCausalLM(OriginalMistralForCausalLM):
    def __init__(self, config):
        super(MistralPreTrainedModel, self).__init__(config)
        self.model = MistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        if labels is not None:
            # Each hidden state is for the *next* label, so shift labels right
            shift_hidden_states = hidden_states[..., :-1, :].contiguous().view(-1, self.config.hidden_size)
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = FusedLinearCrossEntropyFunction.apply(shift_hidden_states, self.lm_head.weight, shift_labels)
            # Logits are not needed for training/evaluation
            return (loss, None)  # type: ignore
        else:
            return (self.lm_head(hidden_states),)
