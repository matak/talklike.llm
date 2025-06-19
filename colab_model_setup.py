"""
Modul pro nastavení modelu a LoRA
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

def load_tokenizer(config):
    """Načte a nastaví tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    
    # Přidání padding tokenu pokud chybí
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer načten: {config.base_model}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Pad token: {tokenizer.pad_token}")
    
    return tokenizer

def load_model(config):
    """Načte model s kvantizací (optimalizováno pro Colab GPU)"""
    model_kwargs = {
        "torch_dtype": torch.float16 if config.fp16 else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None
    }
    
    # Kvantizace pro úsporu paměti
    if config.use_4bit:
        model_kwargs["load_in_4bit"] = True
    elif config.use_8bit:
        model_kwargs["load_in_8bit"] = True
    
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        **model_kwargs
    )
    
    # Příprava modelu pro kvantizované trénování
    if config.use_4bit or config.use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    print(f"Model načten: {config.base_model}")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    return model

def setup_lora(model, config):
    """Nastaví LoRA na model"""
    # Nastavení LoRA konfigurace
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Aplikace LoRA na model
    model = get_peft_model(model, lora_config)
    
    # Výpis trénovatelných parametrů
    model.print_trainable_parameters()
    
    print("\nLoRA konfigurace nastavena!")
    print(f"LoRA r: {config.lora_r}")
    print(f"LoRA alpha: {config.lora_alpha}")
    print(f"Target modules: {config.target_modules}")
    
    return model 