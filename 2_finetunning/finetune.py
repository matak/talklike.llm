#!/usr/bin/env python3
"""
Minimální fine-tuning script pro model s daty Andreje Babiše
Spustitelný na RunPod.io nebo lokálně
"""

# Filtrování varování
import warnings
warnings.filterwarnings("ignore", message="Using `TRANSFORMERS_CACHE` is deprecated")
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

# Import setup_environment pro správné nastavení prostředí
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
import argparse
import json

# Import disk manager knihovny pro specifické operace
from lib.disk_manager import DiskManager

# Import modulů
from data_utils import load_model_data, prepare_training_data
from tokenizer_utils import setup_tokenizer_and_model, check_unknown_tokens, check_tokenizer_compatibility, tokenize_function
from debug_utils import DatasetDebugger
from train_utils import generate_response, test_model, save_model_info

def save_dataset_to_file(dataset, filepath, description):
    """Uloží kompletní dataset do JSON souboru pro debug"""
    print(f"💾 Ukládám {description} do: {filepath}")
    
    # Převod datasetu na list slovníků
    data_list = []
    for i, item in enumerate(dataset):
        data_list.append({
            "index": i,
            "text": item.get("text", ""),
            "input_ids": item.get("input_ids", []),
            "attention_mask": item.get("attention_mask", []),
            "labels": item.get("labels", [])
        })
    
    # Uložení do JSON souboru
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            "description": description,
            "total_samples": len(data_list),
            "data": data_list
        }, f, ensure_ascii=False, indent=2)
    
    print(f"✅ {description} uloženo: {len(data_list)} vzorků")

def save_vocabulary_to_file(tokenizer, filepath):
    """Uloží kompletní slovník tokenů do souboru pro debug"""
    print(f"💾 Ukládám kompletní slovník tokenů do: {filepath}")
    
    vocab_data = {
        "vocab_size": tokenizer.vocab_size,
        "model_name": tokenizer.name_or_path,
        "special_tokens": {
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token": tokenizer.bos_token,
            "bos_token_id": tokenizer.bos_token_id,
            "unk_token": tokenizer.unk_token,
            "unk_token_id": tokenizer.unk_token_id,
        },
        "vocabulary": {}
    }
    
    # Načtení všech tokenů ze slovníku
    for token_id in range(tokenizer.vocab_size):
        try:
            token = tokenizer.convert_ids_to_tokens(token_id)
            vocab_data["vocabulary"][str(token_id)] = {
                "token": token,
                "decoded": tokenizer.decode([token_id], skip_special_tokens=False)
            }
        except Exception as e:
            vocab_data["vocabulary"][str(token_id)] = {
                "token": f"ERROR_{token_id}",
                "error": str(e)
            }
    
    # Uložení do JSON souboru
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Slovník uložen: {tokenizer.vocab_size} tokenů")

def ask_user_continue(prompt="Pokračovat ve zpracování?"):
    """Zeptá se uživatele, zda má pokračovat"""
    print(f"\n⏸️ {prompt}")
    print("   Stiskněte ENTER pro pokračování nebo napište 'stop' pro ukončení...")
    
    try:
        user_input = input().strip().lower()
        if user_input in ['stop', 'exit', 'quit', 'no', 'n']:
            print("🛑 Ukončuji skript na žádost uživatele.")
            return False
        else:
            print("✅ Pokračuji ve zpracování...")
            return True
    except KeyboardInterrupt:
        print("\n🛑 Ukončeno uživatelem (Ctrl+C)")
        return False
    except EOFError:
        print("\n🛑 Ukončeno uživatelem")
        return False

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
    parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.3', help='Název base modelu')
    parser.add_argument('--epochs', type=int, default=3, help='Počet epoch')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximální délka sekvence')
    parser.add_argument('--push_to_hub', action='store_true', help='Nahrát model na HF Hub')
    parser.add_argument('--hub_model_id', type=str, default='babis-lora', help='Název modelu na HF Hub')
    parser.add_argument('--aggressive_cleanup', action='store_true', help='Agresivní vyčištění pro velké modely')
    parser.add_argument('--no_interactive', action='store_true', help='Bez interaktivních dotazů')
    
    args = parser.parse_args()
    
    # Zajistíme, že výstupní adresář je na network storage
    if not args.output_dir.startswith('/workspace'):
        args.output_dir = f'/workspace/{args.output_dir.lstrip("./")}'
    
    print("🚀 Spouštím fine-tuning pro Andreje Babiše")
    print(f"📁 Data: {args.data_path}")
    print(f"📁 Výstup: {args.output_dir}")
    print(f"📁 Model: {args.model_name}")
    
    # Inicializace disk manageru pro specifické operace
    dm = DiskManager()
    
    # Optimalizace pro velké modely (setup_environment.py už udělal základní nastavení)
    if args.aggressive_cleanup or "mistral" in args.model_name.lower() or "llama" in args.model_name.lower():
        print("🧹 Optimalizace pro velký model...")
        if not dm.optimize_for_large_models(args.model_name):
            print("❌ Nedost místa pro velký model. Zkuste menší model.")
            return
    
    # Načtení proměnných prostředí
    load_dotenv()

    
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)
    
    # Hugging Face token
    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)
        print("✅ Hugging Face login úspěšný")
    else:
        print("⚠️ HF_TOKEN nebyl nalezen")
    
    # Inicializace debuggeru pro sledování zpracování datasetu
    debugger = DatasetDebugger(debug_dir="debug_dataset_finetune")
    print(f"🔍 Debugger inicializován: {debugger.debug_dir}")
    
    # 1. Načtení dat
    print("\n📊 Načítám data...")
    conversations = load_model_data(args.data_path, debugger)
    print(f"✅ Načteno {len(conversations)} konverzací")
    
    # 2. Načtení modelu a tokenizeru (před přípravou dat)
    print(f"\n🤖 Načítám model: {args.model_name}")

    # Konfigurace 4-bit kvantizace
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
        trust_remote_code=True,
        cache_dir='/workspace/.cache/huggingface/transformers',
        local_files_only=False,
        resume_download=True,
        force_download=False
    )
    print("✅ Model úspěšně načten!")

    tokenizer, model = setup_tokenizer_and_model(args.model_name, model)
    
    print(f"✅ Model načten. Vocab size: {model.config.vocab_size}")
    
    # 3. Příprava dat s tokenizerem (nyní máme přístup k apply_chat_template)
    print("🔧 Připravuji data s apply_chat_template...")
    training_data = prepare_training_data(conversations, tokenizer, debugger)
    print(f"✅ Připraveno {len(training_data)} trénovacích vzorků")

    # DEBUG: Test generování před fine-tuningem
    print("\n🧪 DEBUG: Testuji generování před fine-tuningem...")
    try:
        # Použití konzistentní funkce test_model místo vlastní implementace
        test_model(model, tokenizer, test_prompts=[
            "Pane Babiši, jak hodnotíte současnou inflaci?"
        ])
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Chyba při debug generování: {e}")

    # 4. Vytvoření Dataset
    dataset = Dataset.from_list(training_data)
    
    # Debug: Uložení finálního datasetu
    debugger.save_step("07_final_dataset", {"dataset_size": len(dataset), "columns": dataset.column_names}, 
                      f"Finální dataset s {len(dataset)} vzorky (text formát z apply_chat_template)")
    
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
        "max_length": args.max_length,
        "method": "standard_tokenization"
    }, f"Tokenizovaný dataset s {len(tokenized_dataset)} vzorky (text již formátován pomocí apply_chat_template)")
    
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
    
    # DEBUG: Uložení kompletních train a validation dat do souborů
    print("\n💾 DEBUG: Ukládám kompletní train a validation data...")
    
    # Vytvoření debug adresáře
    debug_data_dir = os.path.join(debugger.debug_dir, "complete_datasets")
    os.makedirs(debug_data_dir, exist_ok=True)
    
    # Uložení train datasetu
    train_file = os.path.join(debug_data_dir, "complete_train_dataset.json")
    save_dataset_to_file(train_dataset, train_file, f"Kompletní train dataset ({len(train_dataset)} vzorků)")
    
    # Uložení validation datasetu
    eval_file = os.path.join(debug_data_dir, "complete_validation_dataset.json")
    save_dataset_to_file(eval_dataset, eval_file, f"Kompletní validation dataset ({len(eval_dataset)} vzorků)")
    
    print(f"✅ Kompletní data uložena v: {debug_data_dir}")
    
    # Interaktivní kontrola po rozdělení dat
    if not args.no_interactive:
        if not ask_user_continue("Data jsou rozdělena a uložena. Pokračovat ve zpracování?"):
            return
    
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
    
    # DEBUG: Uložení kompletního slovníku tokenů
    print("\n💾 DEBUG: Ukládám kompletní slovník tokenů...")
    
    vocab_file = os.path.join(debug_data_dir, "complete_vocabulary.json")
    save_vocabulary_to_file(tokenizer, vocab_file)
    
    print(f"✅ Kompletní slovník uložen v: {vocab_file}")
    
    # Interaktivní kontrola po uložení slovníku
    if not args.no_interactive:
        if not ask_user_continue("Slovník je uložen. Pokračovat ve zpracování?"):
            return
    
    # 7. Data Collator
    print("\n🔧 Konfiguruji data collator...")
    
    # Vlastní data collator pro správné řešení padding
    class CustomDataCollator:
        def __init__(self, tokenizer, max_length=1024):
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __call__(self, features):
            # Získání maximální délky v batch
            max_len = max(len(feature['input_ids']) for feature in features)
            max_len = min(max_len, self.max_length)
            
            # Padding všech sekvencí na stejnou délku
            batch = {
                'input_ids': [],
                'attention_mask': [],
                'labels': []
            }
            
            for feature in features:
                input_ids = feature['input_ids'][:max_len]
                attention_mask = feature['attention_mask'][:max_len]
                labels = feature['labels'][:max_len]
                
                # Padding na max_len
                padding_length = max_len - len(input_ids)
                if padding_length > 0:
                    input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                    attention_mask = attention_mask + [0] * padding_length
                    labels = labels + [-100] * padding_length  # -100 pro ignorování při loss
                
                batch['input_ids'].append(input_ids)
                batch['attention_mask'].append(attention_mask)
                batch['labels'].append(labels)
            
            # Konverze na tensory
            return {
                'input_ids': torch.tensor(batch['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(batch['attention_mask'], dtype=torch.long),
                'labels': torch.tensor(batch['labels'], dtype=torch.long)
            }
    
    data_collator = CustomDataCollator(tokenizer, args.max_length)
    
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
        report_to=[],  # Vypnout wandb a další reporting
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
    print("\n🎉 Fine-tuning dokončen!")
    print(f"📁 Model uložen v: {final_model_path}")
    print(f"💾 Network storage: {args.output_dir}")
    if args.push_to_hub:
        print(f"🌐 Model dostupný na: https://huggingface.co/{args.hub_model_id}")

if __name__ == "__main__":
    main() 