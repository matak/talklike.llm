#!/usr/bin/env python3

"""
Konfigurace pro fine-tuning Babišova stylu
"""
from dataclasses import dataclass
from typing import List

@dataclass
class ColabConfig:
    """Konfigurace pro Google Colab fine-tuning"""
    
    # Model settings
    base_model: str = "microsoft/DialoGPT-medium"  # Malý model pro Colab
    model_name: str = "babis-dialogpt-colab"
    
    # Training settings (optimalizováno pro Colab GPU)
    learning_rate: float = 2e-4
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 0.3
    warmup_steps: int = 50
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    
    # LoRA settings
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    # Dataset settings
    max_seq_length: int = 512
    train_split: float = 0.9
    eval_split: float = 0.1
    
    # Output settings
    output_dir: str = "/content/babis_finetune"
    logging_dir: str = "/content/babis_finetune/logs"
    
    # Hardware settings (optimalizováno pro Colab)
    fp16: bool = True
    bf16: bool = False
    use_8bit: bool = False
    use_4bit: bool = True
    
    # Evaluation settings
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    def __post_init__(self):
        if self.target_modules is None:
            # Pro DialoGPT
            self.target_modules = ["c_attn", "c_proj", "wte", "wpe"]
        
        # Vytvoření adresářů
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)

# Vytvoření konfigurace
config = ColabConfig()
print("Konfigurace vytvořena:")
print(f"Base model: {config.base_model}")
print(f"Output dir: {config.output_dir}")
print(f"LoRA r: {config.lora_r}, alpha: {config.lora_alpha}") 