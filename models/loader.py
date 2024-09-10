import torch
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

from .cohere import CohereForCausalLM
from .mistral import MistralForCausalLM


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"  # for better numerical stability

    if model_name.startswith("Cohere"):
        model = CohereForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        # Freeze input/output embeddings
        model.lm_head.requires_grad_(False)
        model.model.embed_tokens.requires_grad_(False)
        model.enable_input_require_grads()

        response_template = "<|CHATBOT_TOKEN|>"
    elif model_name.startswith("mistralai"):
        model = MistralForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        # Freeze input/output embeddings
        model.lm_head.requires_grad_(False)
        model.model.embed_tokens.requires_grad_(False)
        model.enable_input_require_grads()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<unk>"

        response_template = "[/INST]"
    else:
        raise ValueError(f"Model {model_name} not supported")

    return (
        model,
        tokenizer,
        DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer,
        ),
    )
