#!/usr/bin/env python3
"""
Skript pro vytvoření QLoRA adaptéru z datasetu
Adaptér se dá snadno připojit k jakémukoli kompatibilnímu modelu
"""

import os
import json
import torch
import argparse
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel
)
from huggingface_hub import login
import wandb

def load_dataset(file_path):
    """Načte dataset z JSONL souboru"""
    conversations = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    conversations.append(data)
                except json.JSONDecodeError:
                    continue
    
    return conversations

def prepare_training_data(conversations):
    """Připraví data pro fine-tuning"""
    training_data = []
    
    for conv in conversations:
        messages = conv.get('messages', [])
        
        # Přeskočíme konverzace bez assistant zpráv
        if not any(msg['role'] == 'assistant' for msg in messages):
            continue
            
        # Vytvoříme text pro fine-tuning
        text = ""
        for msg in messages:
            if msg['role'] == 'system':
                text += f"<|system|>\n{msg['content']}<|end|>\n"
            elif msg['role'] == 'user':
                text += f"<|user|>\n{msg['content']}<|end|>\n"
            elif msg['role'] == 'assistant':
                text += f"<|assistant|>\n{msg['content']}<|end|>\n"
        
        training_data.append({"text": text})
    
    return training_data

def tokenize_function(examples, tokenizer, max_length=2048):
    """Tokenizuje text pro fine-tuning"""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def create_qlora_adapter(
    base_model_name,
    dataset_path,
    output_dir,
    adapter_name="babis_adapter",
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    max_length=2048,
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4
):
    """Vytvoří QLoRA adaptér z datasetu"""
    
    print(f"🤖 Vytvářím QLoRA adaptér pro model: {base_model_name}")
    print(f"📊 Dataset: {dataset_path}")
    print(f"📁 Výstupní adresář: {output_dir}")
    
    # 1. Načtení dat
    print("\n📂 Načítám dataset...")
    conversations = load_dataset(dataset_path)
    training_data = prepare_training_data(conversations)
    
    if not training_data:
        print("❌ Žádná data k trénování!")
        return None
    
    dataset = Dataset.from_list(training_data)
    print(f"✅ Načteno {len(dataset)} vzorků")
    
    # 2. Načtení tokenizeru
    print(f"\n🔤 Načítám tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    
    # 3. Konfigurace 4-bit kvantizace
    print("\n⚙️ Konfiguruji 4-bit kvantizaci...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 4. Načtení modelu s kvantizací
    print(f"\n🤖 Načítám model s 4-bit kvantizací...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 5. Konfigurace LoRA
    print("\n🔧 Konfiguruji LoRA...")
    
    # Automatické zjištění target modules podle architektury
    target_modules = []
    for name, module in model.named_modules():
        if any(target in name for target in ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
            target_modules.append(name.split('.')[-1])
    
    target_modules = list(set(target_modules))  # Odstranění duplicit
    print(f"🎯 Target modules: {target_modules}")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none"
    )
    
    # 6. Příprava modelu pro trénování
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 7. Tokenizace dat
    print("\n🔤 Tokenizuji data...")
    tokenize_func = lambda examples: tokenize_function(examples, tokenizer, max_length)
    tokenized_dataset = dataset.map(
        tokenize_func,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Rozdělení na train/validation
    if len(tokenized_dataset) > 10:
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        train_dataset = tokenized_dataset
        eval_dataset = tokenized_dataset
    
    print(f"✅ Train: {len(train_dataset)}, Validation: {len(eval_dataset)}")
    
    # 8. Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # 9. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="wandb" if wandb.run else None,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        weight_decay=0.01
    )
    
    # 10. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # 11. Trénování
    print(f"\n🚀 Začínám trénování adaptéru...")
    trainer.train()
    
    # 12. Uložení adaptéru
    print(f"\n💾 Ukládám adaptér...")
    adapter_path = os.path.join(output_dir, adapter_name)
    trainer.save_model(adapter_path)
    
    # 13. Uložení konfigurace
    config = {
        "base_model": base_model_name,
        "adapter_name": adapter_name,
        "lora_config": {
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules
        },
        "training_config": {
            "max_length": max_length,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        },
        "dataset_info": {
            "total_samples": len(dataset),
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset)
        }
    }
    
    config_path = os.path.join(output_dir, f"{adapter_name}_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Adaptér uložen do: {adapter_path}")
    print(f"✅ Konfigurace uložena do: {config_path}")
    
    return adapter_path

def main():
    parser = argparse.ArgumentParser(description="Vytvoření QLoRA adaptéru z datasetu")
    
    parser.add_argument("--base-model", required=True, help="Základní model (např. microsoft/DialoGPT-medium)")
    parser.add_argument("--dataset", required=True, help="Cesta k JSONL datasetu")
    parser.add_argument("--output-dir", required=True, help="Výstupní adresář pro adaptér")
    parser.add_argument("--adapter-name", default="babis_adapter", help="Název adaptéru")
    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximální délka sekvence")
    parser.add_argument("--epochs", type=int, default=3, help="Počet epoch")
    parser.add_argument("--batch-size", type=int, default=4, help="Velikost batch")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    # Vytvoření adaptéru
    adapter_path = create_qlora_adapter(
        base_model_name=args.base_model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        adapter_name=args.adapter_name,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_length=args.max_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    if adapter_path:
        print(f"\n🎉 Adaptér úspěšně vytvořen!")
        print(f"📁 Cesta: {adapter_path}")
        print(f"\n📖 Pro použití adaptéru:")
        print(f"   python test_adapter.py --base-model {args.base_model} --adapter {adapter_path}")

if __name__ == "__main__":
    main() 