#!/usr/bin/env python3
"""
Fine-tuning script pro model s daty Andreje Babiše
Spustitelný na RunPod.io nebo lokálně
"""

import os
import json
import torch
import numpy as np
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
    prepare_model_for_kbit_training
)
from dotenv import load_dotenv
from huggingface_hub import login
import wandb
import argparse

def load_babis_data(file_path):
    """Načte data z JSONL souboru nebo jednoho velkého JSON objektu"""
    conversations = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    try:
        # Zkusíme parsovat jako jeden velký JSON objekt
        data = json.loads(content)
        
        if 'messages' in data:
            # Máme jeden velký objekt s messages - rozdělíme na konverzace
            messages = data['messages']
            print(f"📊 Načteno {len(messages)} zpráv v jednom objektu")
            
            # Rozdělení na konverzace (každých 3 zprávy = 1 konverzace)
            i = 0
            while i < len(messages):
                # Najdeme system zprávu
                if i < len(messages) and messages[i]['role'] == 'system':
                    system_msg = messages[i]
                    i += 1
                    
                    # Najdeme user a assistant zprávy
                    conv_messages = [system_msg]
                    while i < len(messages) and messages[i]['role'] in ['user', 'assistant']:
                        conv_messages.append(messages[i])
                        i += 1
                    
                    # Vytvoříme konverzaci
                    if len(conv_messages) >= 3:  # system + user + assistant
                        conversations.append({
                            "messages": conv_messages
                        })
                else:
                    i += 1
            
            print(f"✅ Vytvořeno {len(conversations)} konverzací")
            return conversations
            
    except json.JSONDecodeError:
        # Není jeden velký JSON objekt, zkusíme JSONL formát
        print("📊 Zkouším JSONL formát...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        conversations.append(data)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Chyba při parsování řádku: {e}")
                        continue
        
        print(f"✅ Načteno {len(conversations)} konverzací z JSONL")
        return conversations
    
    return conversations

def prepare_training_data(conversations):
    """Připraví data pro fine-tuning"""
    training_data = []
    
    for conv in conversations:
        messages = conv['messages']
        
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
    # Tokenizace
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None
    )
    
    # Nastavení labels stejné jako input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def main():
    parser = argparse.ArgumentParser(description='Fine-tuning 3 8B pro Andreje Babiše')
    parser.add_argument('--data_path', type=str, default='data/all.jsonl', help='Cesta k datům')
    parser.add_argument('--output_dir', type=str, default='./babis-finetuned', help='Výstupní adresář')
    parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.3', help='Název base modelu')
    parser.add_argument('--epochs', type=int, default=3, help='Počet epoch')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=2048, help='Maximální délka sekvence')
    parser.add_argument('--use_wandb', action='store_true', help='Použít Weights & Biases')
    parser.add_argument('--push_to_hub', action='store_true', help='Nahrát model na HF Hub')
    parser.add_argument('--hub_model_id', type=str, default='babis-lora', help='Název modelu na HF Hub')
    
    args = parser.parse_args()
    
    print("🚀 Spouštím fine-tuning pro Andreje Babiše")
    print(f"📁 Data: {args.data_path}")
    print(f"📁 Výstup: {args.output_dir}")
    print(f"🤖 Model: {args.model_name}")
    
    # Načtení proměnných prostředí
    load_dotenv()
    
    # Hugging Face token
    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)
        print("✅ Hugging Face login úspěšný")
    else:
        print("⚠️ HF_TOKEN nebyl nalezen")
    
    # Weights & Biases
    if args.use_wandb:
        WANDB_API_KEY = os.getenv("WANDB_API_KEY")
        if WANDB_API_KEY:
            os.environ["WANDB_API_KEY"] = WANDB_API_KEY
            wandb.login()
            wandb.init(project="babis-finetune", name=args.model_name)
            print("✅ W&B login úspěšný")
        else:
            print("⚠️ WANDB_API_KEY nebyl nalezen")
    
    # 1. Načtení dat
    print("\n📊 Načítám data...")
    conversations = load_babis_data(args.data_path)
    print(f"✅ Načteno {len(conversations)} konverzací")
    
    # 2. Příprava dat
    print("🔧 Připravuji data...")
    training_data = prepare_training_data(conversations)
    print(f"✅ Připraveno {len(training_data)} trénovacích vzorků")
    
    # 3. Vytvoření Dataset
    dataset = Dataset.from_list(training_data)
    
    # 4. Načtení modelu
    print(f"\n🤖 Načítám model: {args.model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Přidání pad tokenu
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f"✅ Model načten. Vocab size: {model.config.vocab_size}")
    
    # 5. Konfigurace LoRA
    print("\n🔧 Konfiguruji LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 6. Tokenizace dat
    print("\n🔤 Tokenizuji data...")
    tokenize_func = lambda examples: tokenize_function(examples, tokenizer, args.max_length)
    tokenized_dataset = dataset.map(
        tokenize_func,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Rozdělení na train/validation
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"✅ Train dataset: {len(train_dataset)} vzorků")
    print(f"✅ Validation dataset: {len(eval_dataset)} vzorků")
    
    # 7. Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 8. Training Arguments
    print("\n⚙️ Nastavuji training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if args.use_wandb else "none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=0.3,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=HF_TOKEN if args.push_to_hub else None,
    )
    
    # 9. Trainer
    print("\n🏋️ Vytvářím Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 10. Fine-tuning
    print("\n🚀 Spouštím fine-tuning...")
    trainer.train()
    
    # 11. Uložení modelu
    print("\n💾 Ukládám model...")
    trainer.save_model(f"{args.output_dir}-final")
    tokenizer.save_pretrained(f"{args.output_dir}-final")
    
    if args.push_to_hub and HF_TOKEN:
        print("📤 Nahrávám model na Hugging Face Hub...")
        model.push_to_hub(args.hub_model_id, token=HF_TOKEN)
        tokenizer.push_to_hub(args.hub_model_id, token=HF_TOKEN)
        print(f"✅ Model nahrán: https://huggingface.co/{args.hub_model_id}")
    
    # 12. Testování
    print("\n🧪 Testuji model...")
    def generate_response(prompt, max_length=200):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    test_prompts = [
        "Pane Babiši, jak hodnotíte současnou inflaci?",
        "Co si myslíte o opozici?",
        "Jak se vám daří v Bruselu?"
    ]
    
    print("\n📝 Testovací odpovědi:")
    print("=" * 50)
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = generate_response(prompt)
        print(f"Odpověď: {response}")
        print("-" * 30)
    
    # 13. Ukončení
    if args.use_wandb:
        wandb.finish()
    
    print("\n🎉 Fine-tuning dokončen!")
    print(f"📁 Model uložen v: {args.output_dir}-final")
    if args.push_to_hub:
        print(f"🌐 Model dostupný na: https://huggingface.co/{args.hub_model_id}")

if __name__ == "__main__":
    main() 