from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    model_name: str = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    # LoRA target modules
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"

    # Training Arguments
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: int = 60
    logging_steps: int = 1
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    output_dir: str = "models/outputs"
