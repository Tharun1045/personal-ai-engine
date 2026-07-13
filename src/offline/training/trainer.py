import sys
from loguru import logger
from src.offline.training.training_config import TrainingConfig


def train_model(config: TrainingConfig, hf_dataset_id: str) -> bool:
    """Train a model using Unsloth LoRA/QLoRA on supported platforms (Linux/WSL).

    Args:
        config: Training configuration.
        hf_dataset_id: HuggingFace dataset repo ID or local path.

    Returns:
        bool: True if training succeeded, False otherwise.
    """
    if sys.platform == "win32":
        logger.warning(
            "Unsloth training is not supported on native Windows. "
            "Please run in WSL, Linux, or a Colab notebook with GPU support."
        )
        return False

    try:
        # Lazy imports of GPU-dependent training libraries
        from trl import SFTTrainer  # type: ignore
        from transformers import TrainingArguments  # type: ignore
        from unsloth import FastLanguageModel  # type: ignore
        from datasets import load_dataset  # type: ignore

        logger.info(f"Initializing FastLanguageModel for model: {config.model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
        )

        logger.info("Setting up PEFT/LoRA modules...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=config.lora_target_modules,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.bias,
            use_gradient_checkpointing="unsloth",
            random_state=config.seed,
        )

        logger.info(f"Loading dataset: {hf_dataset_id}")
        dataset = load_dataset(hf_dataset_id, split="train")

        # Format dataset to standard Alpaca prompt template
        def format_prompts(examples):
            alpaca_prompt = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n{output}"
            )
            texts = []
            for instruction, output in zip(examples["instruction"], examples["output"]):
                text = alpaca_prompt.format(instruction=instruction, output=output)
                texts.append(text)
            return {"text": texts}

        dataset = dataset.map(format_prompts, batched=True)

        logger.info("Setting up SFTTrainer...")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=config.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=config.per_device_train_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                warmup_steps=5,
                max_steps=config.max_steps,
                learning_rate=config.learning_rate,
                fp16=not FastLanguageModel.is_bfloat16_supported(),
                bf16=FastLanguageModel.is_bfloat16_supported(),
                logging_steps=config.logging_steps,
                optim="adamw_8bit",
                weight_decay=config.weight_decay,
                lr_scheduler_type=config.lr_scheduler_type,
                seed=config.seed,
                output_dir=config.output_dir,
            ),
        )

        logger.info("Starting model training run...")
        trainer.train()

        logger.info(f"Model training complete. Saving adapter to: {config.output_dir}")
        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        return True

    except Exception as e:
        logger.error(f"Error during FastLanguageModel fine-tuning: {e}")
        raise e
