# Uložení modelu
print("💾 Ukládám model...")

# Uložení na Colab
trainer.save_model()
tokenizer.save_pretrained(config.output_dir)

# Uložení na Google Drive
drive_path = f"/content/drive/MyDrive/babis_finetune/{config.model_name}"
!cp -r {config.output_dir} {drive_path}

print(f"✅ Model uložen:")
print(f"Colab: {config.output_dir}")
print(f"Google Drive: {drive_path}")

# Výpis velikosti modelu
!du -sh {config.output_dir} 