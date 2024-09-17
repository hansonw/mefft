from typing import Optional

import torch
import torch.nn as nn
from liger_kernel.transformers import LigerSwiGLUMLP, LigerRMSNorm, liger_rotary_pos_emb
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention as OriginalQwen2Attention,
    Qwen2Config,
    Qwen2ForCausalLM as OriginalQwen2ForCausalLM,
    Qwen2PreTrainedModel,
)

from .kernels import FusedLinearCrossEntropyFunction
from .utils import OffloadCheckpointer


# Same as OriginalQwen2Attention, but simplify and use liger_rotary_pos_emb
class Qwen2Attention(OriginalQwen2Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = liger_rotary_pos_emb(query_states, key_states, cos, sin)

        if attention_mask is not None:
            seq_len = attention_mask.shape[1]
            key_states = key_states[:, :, :seq_len]
            value_states = value_states[:, :, :seq_len]

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,  # type: ignore
            q_len,
            softmax_scale=self.scaling,
            is_causal=True,
            sliding_window=sliding_window,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        return self.o_proj(attn_output)


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = LigerSwiGLUMLP(config)
        self.input_layernorm = LigerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LigerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _self_attn(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        return x + self.self_attn(self.input_layernorm(x), attention_mask, position_ids)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.post_attention_layernorm(x))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        hidden_states = OffloadCheckpointer.apply(hidden_states, self._self_attn, attention_mask, position_ids)
        return OffloadCheckpointer.apply(hidden_states, self._mlp)


class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LigerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(0, hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, attention_mask, position_ids)
        return self.norm(hidden_states)


class Qwen2ForCausalLM(OriginalQwen2ForCausalLM):
    def __init__(self, config: Qwen2Config):
        super(Qwen2PreTrainedModel, self).__init__(config)
        self.model = Qwen2Model(config)
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
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        if labels is not None:
            # Each hidden state is for the *next* label, so shift labels right
            shift_hidden_states = hidden_states[..., :-1, :].contiguous().view(-1, self.config.hidden_size)
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = FusedLinearCrossEntropyFunction.apply(
                shift_hidden_states,
                self.lm_head.weight,
                shift_labels,
            )
            # Logits are not needed for training/evaluation
            return (loss, None)  # type: ignore
        else:
            return (self.lm_head(hidden_states),)
