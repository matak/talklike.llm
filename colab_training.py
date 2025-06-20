# Nastavení training arguments
training_args = TrainingArguments(
    output_dir=config.output_dir,
    num_train_epochs=config.num_train_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    learning_rate=config.learning_rate,
    max_grad_norm=config.max_grad_norm,
    warmup_steps=config.warmup_steps,
    logging_steps=config.logging_steps,
    save_steps=config.save_steps,
    eval_steps=config.eval_steps,
    evaluation_strategy=config.evaluation_strategy,
    save_strategy=config.save_strategy,
    load_best_model_at_end=config.load_best_model_at_end,
    metric_for_best_model=config.metric_for_best_model,
    greater_is_better=config.greater_is_better,
    fp16=config.fp16,
    bf16=config.bf16,
    logging_dir=config.logging_dir,
    report_to="wandb" if wandb else None,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    save_total_limit=3,
    prediction_loss_only=True
)

print("Training arguments nastaveny:")
print(f"Learning rate: {config.learning_rate}")
print(f"Epochs: {config.num_train_epochs}")
print(f"Batch size: {config.per_device_train_batch_size}")
print(f"Gradient accumulation: {config.gradient_accumulation_steps}")