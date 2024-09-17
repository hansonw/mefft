import modal

from train import train

app_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "software-properties-common", "git", "gcc", "curl", "sudo", "htop", "nvtop"
    )
    .run_commands(
        "pip install torch --index-url https://download.pytorch.org/whl/cu124"
    )
    .pip_install(
        "bitsandbytes",
        "datasets",
        "hf-transfer",
        "huggingface_hub",
        "numba",
        "peft",
        "sentencepiece",
        "transformers",
        "trl",
        "wandb",
        # For flash-attn
        "packaging",
        "wheel",
    )
    .run_commands("pip install flash-attn --no-build-isolation")
    .run_commands("pip install git+https://github.com/hansonw/Liger-Kernel.git")
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE="/pretrained",
            HF_HUB_ENABLE_HF_TRANSFER="1",
        )
    )
)

app = modal.App(
    "mefft",
    image=app_image,
    # Should contain HF_TOKEN and WANDB_API_KEY; optionally WANDB_PROJECT
    secrets=[modal.Secret.from_dotenv()],
    volumes={
        "/pretrained": modal.Volume.from_name("pretrained-vol"),
        "/runs": modal.Volume.from_name("runs-vol"),
    },
)


@app.function(gpu="h100:1", timeout=86400)
async def main(
    model_name: str,
    train_path: str,
    eval_path: str,
    run_name: str,
    resume: bool = True,
    max_seq_length: int = 20000,
    learning_rate: float = 5e-6,
    micro_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
):
    assert train.callback
    train.callback(
        model_name=model_name,
        train_path=train_path,
        eval_path=eval_path,
        run_name=run_name,
        output_dir="/runs/" + run_name,
        resume=resume,
        max_seq_length=max_seq_length,
        learning_rate=learning_rate,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
