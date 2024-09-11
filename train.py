from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer
from models.loader import load_model
from optimizer import InterleavedOptimizer, MappedAdamW8bit, initialize_mapped_gradients
import click


@click.command()
@click.option("--model-name", required=True, help="Name of the model to train")
@click.option("--train-path", required=True, help="Path to the training data file")
@click.option("--eval-path", required=True, help="Path to the evaluation data file")
@click.option("--run-name", required=True, help="Name of the training run")
@click.option("--output-dir", required=True, help="Directory to save the output")
@click.option("--resume", is_flag=True, help="Resume training from the last checkpoint")
@click.option("--max-seq-length", default=20000, help="Maximum sequence length")
@click.option("--learning-rate", default=5e-6, help="Learning rate")
@click.option("--micro-batch-size", default=8, help="Batch size per forward pass")
@click.option(
    "--gradient-accumulation-steps",
    default=1,
    help="Number of gradient accumulation steps",
)
def train(
    model_name: str,
    train_path: str,
    eval_path: str,
    run_name: str,
    output_dir: str,
    resume: bool,
    max_seq_length: int,
    learning_rate: float,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
):
    print("Loading datasets...")
    train_data = load_dataset("json", data_files=train_path, split="train")
    assert isinstance(train_data, Dataset)
    eval_data = load_dataset(
        "json", data_files=eval_path, split="train", keep_in_memory=True
    ).sort("tokens")  # type: ignore

    print(f"Loading model {model_name}...")
    model, tokenizer, collator = load_model(model_name)

    if gradient_accumulation_steps > 1:
        print("Initializing offloaded gradients and optimizer...")
        initialize_mapped_gradients(model)
        optimizer = MappedAdamW8bit(model.parameters(), lr=learning_rate)
        # Make sure the mapped gradients do not get cleared!
        original_zero_grad = model.zero_grad
        model.zero_grad = lambda set_to_none=False: original_zero_grad(False)
    else:
        optimizer = InterleavedOptimizer(
            model.parameters(), MappedAdamW8bit, lr=learning_rate
        )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=collator,
        optimizers=(optimizer, None),  # type: ignore - scheduler will be initialized
        args=SFTConfig(
            # SFT-only
            max_seq_length=max_seq_length,
            # hyperparameters
            num_train_epochs=1.0,
            group_by_length=True,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine_with_min_lr",
            lr_scheduler_kwargs={"min_lr_rate": 0.01},
            neftune_noise_alpha=5,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=50,
            seed=42,
            # evaluation
            per_device_eval_batch_size=8,
            eval_strategy="steps",
            eval_steps=0.2,
            eval_on_start=False,
            # saving/logging
            logging_steps=1,
            run_name=run_name,
            output_dir=output_dir,
            save_strategy="steps",
            save_steps=0.25,
            report_to="wandb",
        ),
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume)  # type: ignore
    print(trainer.evaluate())


if __name__ == "__main__":
    train()
