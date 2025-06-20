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