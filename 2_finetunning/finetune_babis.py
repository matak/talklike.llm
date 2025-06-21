#!/usr/bin/env python3
"""
Minimální fine-tuning script pro model s daty Andreje Babiše
Spustitelný na RunPod.io nebo lokálně
"""

# Import setup_environment pro správné nastavení prostředí
import setup_environment

import os
import torch
import numpy as np
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
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

# Import disk manager knihovny
from lib.disk_manager import DiskManager, setup_for_ml_project, check_and_cleanup

# Import modulů
from data_utils import load_babis_data, prepare_training_data
from tokenizer_utils import setup_tokenizer_and_model, check_unknown_tokens, check_tokenizer_compatibility, tokenize_function
from debug_utils import DatasetDebugger
from train_utils import generate_response, test_model, save_model_info

def main():
    # Kontrola, že jsme v root directory projektu
    if not os.path.exists('lib') or not os.path.exists('data'):
        print("❌ Skript musí být spuštěn z root directory projektu!")
        print("💡 Spusťte skript z adresáře, kde jsou složky 'lib' a 'data'")
        print(f"📍 Aktuální adresář: {os.getcwd()}")
        return
    
    parser = argparse.ArgumentParser(description='Fine-tuning pro Andreje Babiše')
    parser.add_argument('--data_path', type=str, default='data/all.jsonl', help='Cesta k datům')
    parser.add_argument('--output_dir', type=str, default='/workspace/babis-finetuned', help='Výstupní adresář')
    parser.add_argument('--model_name', type=str, default='microsoft/DialoGPT-medium', help='Název base modelu')
    parser.add_argument('--epochs', type=int, default=3, help='Počet epoch')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximální délka sekvence')
    parser.add_argument('--use_wandb', action='store_true', help='Použít Weights & Biases')
    parser.add_argument('--push_to_hub', action='store_true', help='Nahrát model na HF Hub')
    parser.add_argument('--hub_model_id', type=str, default='babis-lora', help='Název modelu na HF Hub')
    parser.add_argument('--cleanup_cache', action='store_true', help='Vyčistit cache před spuštěním')
    parser.add_argument('--aggressive_cleanup', action='store_true', help='Agresivní vyčištění pro velké modely')
    
    args = parser.parse_args()
    
    # Zajistíme, že výstupní adresář je na network storage
    if not args.output_dir.startswith('/workspace'):
        args.output_dir = f'/workspace/{args.output_dir.lstrip("./")}'
    
    print("🚀 Spouštím fine-tuning pro Andreje Babiše")
    print(f"📁 Data: {args.data_path}")
    print(f"📁 Výstup: {args.output_dir}")
    print(f"📁 Model: {args.model_name}")
    
    # Inicializace disk manageru a nastavení pro ML projekt
    dm = setup_for_ml_project("/workspace")
    
    # Kontrola místa a vyčištění pokud je potřeba
    if not check_and_cleanup(threshold=95):
        print("❌ Stále není dost místa. Použijte menší model nebo vyčistěte disk.")
        return
    
    # Vyčištění cache pokud požadováno
    if args.cleanup_cache:
        dm.cleanup_cache()
    
    # Optimalizace pro velké modely
    if args.aggressive_cleanup or "mistral" in args.model_name.lower() or "llama" in args.model_name.lower():
        print("🧹 Optimalizace pro velký model...")
        if not dm.optimize_for_large_models(args.model_name):
            print("❌ Nedost místa pro velký model. Zkuste menší model.")
            return
    
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
    
    # Inicializace debuggeru pro sledování zpracování datasetu
    debugger = DatasetDebugger(debug_dir="debug_dataset_finetune")
    print(f"🔍 Debugger inicializován: {debugger.debug_dir}")
    
    # 1. Načtení dat
    print("\n📊 Načítám data...")
    conversations = load_babis_data(args.data_path, debugger)
    print(f"✅ Načteno {len(conversations)} konverzací")
    
    # 2. Příprava dat
    print("🔧 Připravuji data...")
    training_data = prepare_training_data(conversations, debugger, args.model_name)
    print(f"✅ Připraveno {len(training_data)} trénovacích vzorků")
    
    # 3. Vytvoření Dataset
    dataset = Dataset.from_list(training_data)
    
    # Debug: Uložení finálního datasetu
    debugger.save_step("07_final_dataset", {"dataset_size": len(dataset), "columns": dataset.column_names}, 
                      f"Finální dataset s {len(dataset)} vzorky")
    
    # 4. Načtení modelu
    print(f"\n🤖 Načítám model: {args.model_name}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    
    # Pokus o načtení modelu s retry logikou
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"🔄 Pokus {attempt + 1}/{max_retries} načtení modelu...")
            
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                cache_dir='/workspace/.cache/huggingface/transformers',
                local_files_only=False,
                resume_download=True,
                force_download=False
            )
            print("✅ Model úspěšně načten!")
            break
            
        except OSError as e:
            if "No space left on device" in str(e):
                print(f"❌ Pokus {attempt + 1} selhal - není dost místa")
                if attempt < max_retries - 1:
                    print("🧹 Zkouším další vyčištění...")
                    dm.aggressive_cleanup()
                    import time
                    time.sleep(5)
                else:
                    print("❌ Všechny pokusy selhaly. Zkuste:")
                    print("   1. Použít menší model: --model_name microsoft/DialoGPT-medium")
                    print("   2. Restartovat kontejner")
                    print("   3. Zvýšit velikost root filesystem")
                    return
            else:
                raise e
        except Exception as e:
            print(f"❌ Neočekávaná chyba při načítání modelu: {e}")
            if attempt < max_retries - 1:
                print("🔄 Zkouším znovu...")
                import time
                time.sleep(10)
            else:
                raise e
    
    tokenizer, model = setup_tokenizer_and_model(args.model_name, model)
    
    print(f"✅ Model načten. Vocab size: {model.config.vocab_size}")
    
    # 5. Konfigurace LoRA
    print("\n🔧 Konfiguruji LoRA...")
    
    # Dynamické nastavení target_modules podle typu modelu
    if "dialogpt" in args.model_name.lower():
        target_modules = ["c_attn", "c_proj", "wte", "wpe"]
    elif "gpt2" in args.model_name.lower():
        target_modules = ["c_attn", "c_proj", "wte", "wpe"]
    else:
        target_modules = [
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Menší r pro úsporu paměti
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=target_modules
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
        remove_columns=dataset.column_names,
        batch_size=100  # Menší batch size pro lepší kontrolu
    )
    
    # Debug: Uložení tokenizovaného datasetu
    debugger.save_step("09_tokenized_dataset", {
        "dataset_size": len(tokenized_dataset),
        "columns": tokenized_dataset.column_names,
        "max_length": args.max_length
    }, f"Tokenizovaný dataset s {len(tokenized_dataset)} vzorky")
    
    # Kontrola a oprava padding po tokenizaci
    print("🔧 Kontroluji a opravuji padding...")
    def fix_padding(example):
        """Zajistí, že všechny sekvence mají stejnou délku"""
        max_len = args.max_length
        current_len = len(example['input_ids'])
        
        if current_len < max_len:
            # Přidáme padding
            padding_length = max_len - current_len
            example['input_ids'] = example['input_ids'] + [tokenizer.pad_token_id] * padding_length
            example['attention_mask'] = example['attention_mask'] + [0] * padding_length
            example['labels'] = example['labels'] + [-100] * padding_length  # -100 pro ignorování v loss
        elif current_len > max_len:
            # Ořízneme na max_length
            example['input_ids'] = example['input_ids'][:max_len]
            example['attention_mask'] = example['attention_mask'][:max_len]
            example['labels'] = example['labels'][:max_len]
        
        return example
    
    # Aplikujeme opravu padding na celý dataset
    tokenized_dataset = tokenized_dataset.map(
        fix_padding,
        desc="Opravuji padding"
    )
    
    # Rozdělení na train/validation
    print(f"📊 Celkový počet vzorků: {len(tokenized_dataset)}")
    
    if len(tokenized_dataset) < 5:
        print("⚠️ Málo vzorků pro rozdělení. Používám celý dataset pro trénování.")
        train_dataset = tokenized_dataset
        eval_dataset = tokenized_dataset
    elif len(tokenized_dataset) < 10:
        split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        print(f"✅ Train dataset: {len(train_dataset)} vzorků (80%)")
        print(f"✅ Validation dataset: {len(eval_dataset)} vzorků (20%)")
    else:
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        print(f"✅ Train dataset: {len(train_dataset)} vzorků (90%)")
        print(f"✅ Validation dataset: {len(eval_dataset)} vzorků (10%)")
    
    # Kontrola kompatibility a neznámých tokenů před trénováním
    print(f"\n🔍 FINÁLNÍ KONTROLY PŘED TRÉNOVÁNÍM")
    print(f"=" * 50)
    
    # 1. Kontrola kompatibility tokenizeru
    tokenizer_ok = check_tokenizer_compatibility(tokenizer, args.model_name, debugger)
    if not tokenizer_ok:
        print(f"⚠️ VAROVÁNÍ: Problémy s tokenizerem, ale pokračuji...")
    
    # 2. Kontrola neznámých tokenů v train datasetu
    train_ok = check_unknown_tokens(train_dataset, tokenizer, debugger, max_samples_to_check=50)
    if not train_ok:
        print(f"❌ KRITICKÁ CHYBA: Příliš mnoho neznámých tokenů v train datasetu!")
        print(f"   Zastavuji fine-tuning. Opravte data před pokračováním.")
        return
    
    # 3. Kontrola neznámých tokenů v validation datasetu
    eval_ok = check_unknown_tokens(eval_dataset, tokenizer, debugger, max_samples_to_check=20)
    if not eval_ok:
        print(f"❌ KRITICKÁ CHYBA: Příliš mnoho neznámých tokenů v validation datasetu!")
        print(f"   Zastavuji fine-tuning. Opravte data před pokračováním.")
        return
    
    print(f"✅ Všechny kontroly prošly - pokračuji s trénováním")
    
    # 7. Data Collator
    print("\n🔧 Konfiguruji data collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
        pad_to_multiple_of=8,
    )
    
    # 8. Training Arguments
    print("\n⚙️ Nastavuji training arguments...")
    
    # Dynamické nastavení training parametrů podle velikosti datasetu
    if len(train_dataset) < 10:
        save_steps = max(1, len(train_dataset) // 2)
        eval_steps = max(1, len(train_dataset) // 2)
        logging_steps = 1
    else:
        save_steps = 500
        eval_steps = 500
        logging_steps = 10
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=min(100, len(train_dataset) // 2),
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps",
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
        save_total_limit=2,
        logging_dir=f"{args.output_dir}/logs",
        dataloader_num_workers=0,
        dataloader_drop_last=True,
        group_by_length=True,
    )
    
    # 9. Trainer
    print("\n🏋️ Vytvářím Trainer...")
    
    # Nastavení label_names pro PeftModel
    try:
        if hasattr(model, 'label_names'):
            model.label_names = ['labels']
        elif hasattr(model, 'config') and hasattr(model.config, 'label_names'):
            model.config.label_names = ['labels']
        
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'config'):
            model.base_model.config.label_names = ['labels']
        
        print("✅ Label names nastaveny pro model")
    except Exception as e:
        print(f"⚠️ Nelze nastavit label_names: {e}")
    
    # Zajistíme, že model je v training módu
    model.train()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # 10. Fine-tuning
    print("\n🚀 Spouštím fine-tuning...")
    trainer.train()
    
    # 11. Uložení modelu
    final_model_path = save_model_info(args.output_dir, args.output_dir)
    
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    if args.push_to_hub and HF_TOKEN:
        print("📤 Nahrávám model na Hugging Face Hub...")
        model.push_to_hub(args.hub_model_id, token=HF_TOKEN)
        tokenizer.push_to_hub(args.hub_model_id, token=HF_TOKEN)
        print(f"✅ Model nahrán: https://huggingface.co/{args.hub_model_id}")
    
    # 12. Testování
    print("\n🏋️ Testuji model...")
    test_model(model, tokenizer)
    
    # 13. Ukončení
    if args.use_wandb:
        wandb.finish()
    
    print("\n🎉 Fine-tuning dokončen!")
    print(f"📁 Model uložen v: {final_model_path}")
    print(f"💾 Network storage: {args.output_dir}")
    if args.push_to_hub:
        print(f"🌐 Model dostupný na: https://huggingface.co/{args.hub_model_id}")

if __name__ == "__main__":
    main() 