from typing import Optional

import torch
import torch.nn as nn
from liger_kernel.transformers.geglu import LigerGEGLUMLP
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2Config,
    Gemma2RotaryEmbedding,
    Gemma2ForCausalLM as OriginalGemma2ForCausalLM,
    Gemma2PreTrainedModel,
)

from .kernels import FusedLinearCrossEntropyFunction
from .utils import OffloadCheckpointer


class Gemma2RMSNorm(LigerRMSNorm):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(dim, eps=eps, offset=1.0, casting_mode="gemma")


class Gemma2Attention(nn.Module):
    def __init__(self, config: Gemma2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.scaling = config.query_pre_attn_scalar**-0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )
        self.rotary_emb = Gemma2RotaryEmbedding(self.head_dim, base=config.rope_theta)
        self.sliding_window = config.sliding_window if not bool(layer_idx % 2) else None

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
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = liger_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if attention_mask is not None:
            seq_len = attention_mask.shape[1]
            key_states = key_states[:, :, :seq_len]
            value_states = value_states[:, :, :seq_len]

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,  # type: ignore
            q_len,
            softmax_scale=self.scaling,
            is_causal=True,
            sliding_window=self.sliding_window,
            softcap=self.config.attn_logit_softcapping,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        return self.o_proj(attn_output)


class Gemma2DecoderLayer(nn.Module):
    def __init__(self, config: Gemma2Config, layer_idx: int):
        super().__init__()
        self.self_attn = Gemma2Attention(config=config, layer_idx=layer_idx)
        self.mlp = LigerGEGLUMLP(config)
        self.input_layernorm = Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def _self_attn(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        return x + self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x), attention_mask, position_ids)
        )

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(x))
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        hidden_states = OffloadCheckpointer.apply(
            hidden_states, self._self_attn, attention_mask, position_ids
        )
        return OffloadCheckpointer.apply(hidden_states, self._mlp)


class Gemma2Model(nn.Module):
    def __init__(self, config: Gemma2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                Gemma2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(
                0, hidden_states.shape[1], device=hidden_states.device
            ).unsqueeze(0)
        # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, attention_mask, position_ids)
        return self.norm(hidden_states)


class Gemma2ForCausalLM(OriginalGemma2ForCausalLM):
    def __init__(self, config: Gemma2Config):
        super(Gemma2PreTrainedModel, self).__init__(config)
        self.model = Gemma2Model(config)
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
            shift_hidden_states = (
                hidden_states[..., :-1, :]
                .contiguous()
                .view(-1, self.config.hidden_size)
            )
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = FusedLinearCrossEntropyFunction.apply(
                shift_hidden_states,
                self.lm_head.weight,
                shift_labels,
                None,  # bias
                -100,  # ignore_index
                self.config.final_logit_softcapping,
            )
            # Logits are not needed for training/evaluation
            return (loss, None)  # type: ignore
        else:
            logits = self.lm_head(hidden_states)
            logits = self.config.final_logit_softcapping * torch.tanh(
                logits / self.config.final_logit_softcapping
            )
            return (logits,)
