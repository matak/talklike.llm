# Evaluace modelu
print("📊 Spouštím evaluaci...")

eval_results = trainer.evaluate()

print("\n📈 Výsledky evaluace:")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")

# Uložení metrik
trainer.log_metrics("eval", eval_results)
trainer.save_metrics("eval", eval_results)