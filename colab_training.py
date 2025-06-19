"""
Modul pro nastavení a spuštění trénování
"""
import wandb
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback

def setup_wandb(config):
    """Nastaví Weights & Biases (volitelné)"""
    try:
        wandb.init(
            project="babis-finetune-colab",
            name=config.model_name,
            config=vars(config)
        )
        print("Weights & Biases inicializováno")
        return True
    except Exception as e:
        print(f"WandB inicializace selhala: {e}")
        return False

def create_training_args(config, wandb_available):
    """Vytvoří training arguments"""
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
        report_to="wandb" if wandb_available else None,
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
    
    return training_args

def create_trainer(model, tokenized_dataset, training_args):
    """Vytvoří trainer pro fine-tuning"""
    # Data collator pro language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=None,  # Používáme již tokenizovaná data
        mlm=False  # Causal LM, ne masked LM
    )
    
    # Callback pro early stopping
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.01
    )
    
    # Vytvoření traineru
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        callbacks=[early_stopping_callback]
    )
    
    print("Trainer vytvořen!")
    return trainer

def train_model(trainer):
    """Spustí trénování modelu"""
    print("🚀 Spouštím fine-tuning...")
    
    # Trénování
    trainer.train()
    
    print("✅ Fine-tuning dokončen!")
    return trainer 