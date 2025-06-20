# Shrnutí výsledků
print("🎉 FINE-TUNING DOKONČEN!")
print("=" * 50)
print(f"Model: {config.base_model}")
print(f"Fine-tuned model: {config.model_name}")
print(f"Training loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")
print(f"Evaluation loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
print(f"Training time: {train_result.metrics.get('train_runtime', 'N/A'):.1f}s")
print(f"LoRA parameters: r={config.lora_r}, alpha={config.lora_alpha}")
print(f"Dataset size: {len(tokenized_dataset['train'])} train, {len(tokenized_dataset['validation'])} validation")
print("\n📁 Uložené soubory:")
print(f"Colab: {config.output_dir}")
print(f"Google Drive: /content/drive/MyDrive/babis_finetune/{config.model_name}")

if hf_token.strip():
    print(f"Hugging Face Hub: babis-{config.model_name}")

print("\n🚀 Další kroky:")
print("1. Stáhněte model z Google Drive")
print("2. Použijte ho ve vlastní aplikaci")
print("3. Sdílejte výsledky s ostatními")
print("4. Experimentujte s různými prompty")

print("\n✅ Hotovo! Model je připraven k použití.")