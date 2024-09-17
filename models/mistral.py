# fmt: off
from typing import Optional

import torch
import torch.nn as nn
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from transformers import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding, MistralForCausalLM as OriginalMistralForCausalLM, MistralPreTrainedModel

from .kernels import FusedLinearCrossEntropyFunction
from .utils import OffloadCheckpointer


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


class MistralDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = LigerSwiGLUMLP(config)
        self.input_layernorm = LigerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LigerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _self_attn(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        return x + self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x), attention_mask, position_ids)[0]
        )

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        output = OffloadCheckpointer.apply(x, self._self_attn, attention_mask, position_ids)
        return OffloadCheckpointer.apply(output, self._mlp)


class MistralModel(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([MistralDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = LigerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask,
                position_ids,
            )
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
