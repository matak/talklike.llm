# Načtení modelu s kvantizací (optimalizováno pro Colab GPU)
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