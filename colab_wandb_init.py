# Nastavení Weights & Biases (volitelné)
try:
    wandb.init(
        project="babis-finetune-colab",
        name=config.model_name,
        config=vars(config)
    )
    print("Weights & Biases inicializováno")
except Exception as e:
    print(f"WandB inicializace selhala: {e}")
    wandb = None