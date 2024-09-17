import torch
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM


def load_model(model_name: str, max_seq_length: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"  # for better numerical stability

    if model_name.startswith("mistralai"):
        from .mistral import MistralForCausalLM

        model = MistralForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )

        if tokenizer.pad_token is None:
            print("NOTICE: Setting Mistral pad_token to <unk>")
            tokenizer.pad_token = "<unk>"

        if isinstance(tokenizer.chat_template, str):
            # Patch a bug with the Mistral tokenizer + system prompts
            print("NOTICE: Patching Mistral tokenizer system prompts")
            tokenizer.chat_template = tokenizer.chat_template.replace(
                "loop.last", "loop.first"
            )

        response_template = "[/INST]"
    elif "gemma-2" in model_name:
        from .gemma2 import Gemma2ForCausalLM

        # (Hacky) dynamic RoPE theta scaling
        if max_seq_length > 8192:
            print("NOTICE: scaling Gemma2 rope_theta by", max_seq_length / 8192)
            rope_theta = 10000.0 * max_seq_length / 8192
        else:
            rope_theta = 10000.0

        model = Gemma2ForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            rope_theta=rope_theta,
        )

        # Patch tokenizer to support chat template
        print("NOTICE: Patching Gemma2 tokenizer to support system messages")
        tokenizer.chat_template = """{{ bos_token }}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
{%- endif %}
{%- for message in loop_messages %}
    {%- if message['role'] == 'user' %}
        {%- set role = 'user' %}
        {%- if loop.first and system_message is defined %}
            {%- set content = system_message + '\n\n' + message['content'] %}
        {%- else %}
            {%- set content = message['content'] %}
        {%- endif %}
    {%- else %}
        {%- set role = 'model' %}
        {%- set content = message['content'] %}
    {%- endif %}
    {{- '<start_of_turn>' + role + '\n' + (content | trim) + '<end_of_turn>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}{{'<start_of_turn>model\n'}}{%- endif %}"""

        response_template = "<start_of_turn>model\n"
    else:
        raise ValueError(f"Model {model_name} not supported")

    # Freeze input/output embeddings
    model.get_input_embeddings().requires_grad_(False)
    model.get_output_embeddings().requires_grad_(False)
    model.enable_input_require_grads()

    return (
        model,
        tokenizer,
        DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer,
        ),
    )
