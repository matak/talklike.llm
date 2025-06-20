# Spuštění trénování
print("🚀 Začínám fine-tuning...")
print(f"Model: {config.base_model}")
print(f"Dataset: {len(tokenized_dataset['train'])} train, {len(tokenized_dataset['validation'])} validation")
print(f"Epochs: {config.num_train_epochs}")
print(f"LoRA r: {config.lora_r}, alpha: {config.lora_alpha}")
print("-" * 50)

# Trénování
train_result = trainer.train()

print("\n✅ Fine-tuning dokončen!")
print(f"Training loss: {train_result.metrics.get('train_loss', 'N/A')}")
print(f"Training time: {train_result.metrics.get('train_runtime', 'N/A')}s")