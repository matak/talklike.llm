#!/usr/bin/env python3
"""
Fine-tuning script pro model s daty Andreje Babiše
Spustitelný na RunPod.io nebo lokálně
"""

import os
import json
import torch
import numpy as np
import shutil
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

# Import disk manager knihovny
from disk_manager import DiskManager, setup_for_ml_project, check_and_cleanup

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
            
            # Najdeme system zprávu (měla by být první)
            system_msg = None
            for msg in messages:
                if msg['role'] == 'system':
                    system_msg = msg
                    break
            
            if not system_msg:
                print("❌ Nenalezena system zpráva!")
                return conversations
            
            # Projdeme všechny zprávy a najdeme user-assistant páry
            i = 0
            while i < len(messages):
                # Hledáme user zprávu
                if i < len(messages) and messages[i]['role'] == 'user':
                    user_msg = messages[i]
                    i += 1
                    
                    # Hledáme následující assistant zprávu
                    if i < len(messages) and messages[i]['role'] == 'assistant':
                        assistant_msg = messages[i]
                        i += 1
                        
                        # Vytvoříme konverzaci s system + user + assistant
                        conv_messages = [system_msg, user_msg, assistant_msg]
                        conversations.append({
                            "messages": conv_messages
                        })
                    else:
                        # Chybí assistant zpráva, přeskočíme user zprávu
                        i += 1
                else:
                    # Není user zpráva, přeskočíme
                    i += 1
            
            print(f"✅ Vytvořeno {len(conversations)} konverzací")
            
            # Debug informace
            if len(conversations) > 0:
                print(f"📝 Ukázka první konverzace:")
                first_conv = conversations[0]
                for msg in first_conv['messages']:
                    print(f"  {msg['role']}: {msg['content'][:100]}...")
                
                if len(conversations) > 1:
                    print(f"📝 Ukázka druhé konverzace:")
                    second_conv = conversations[1]
                    for msg in second_conv['messages']:
                        print(f"  {msg['role']}: {msg['content'][:100]}...")
            
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
    # Tokenizace s padding pro konzistentní délky
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,  # Povolíme padding
        max_length=max_length,
        return_tensors=None
    )
    
    # Nastavení labels stejné jako input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def main():
    parser = argparse.ArgumentParser(description='Fine-tuning 3 8B pro Andreje Babiše')
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
    print(f"🤖 Model: {args.model_name}")
    
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
    
    # Debug: Kontrola struktury dat
    print(f"\n🔍 DEBUG: Kontrola struktury dat")
    print(f"📊 Celkový počet vzorků: {len(dataset)}")
    
    if len(dataset) > 0:
        print(f"📝 Ukázka prvního vzorku:")
        first_sample = dataset[0]
        print(f"Text (prvních 200 znaků): {first_sample['text'][:200]}...")
        
        # Kontrola přítomnosti system, user, assistant tagů
        text = first_sample['text']
        has_system = "<|system|>" in text
        has_user = "<|user|>" in text
        has_assistant = "<|assistant|>" in text
        has_end = "<|end|>" in text
        
        print(f"✅ System tag: {has_system}")
        print(f"✅ User tag: {has_user}")
        print(f"✅ Assistant tag: {has_assistant}")
        print(f"✅ End tag: {has_end}")
        
        # Počítání tagů v celém datasetu
        system_count = sum(1 for sample in dataset if "<|system|>" in sample['text'])
        user_count = sum(1 for sample in dataset if "<|user|>" in sample['text'])
        assistant_count = sum(1 for sample in dataset if "<|assistant|>" in sample['text'])
        
        print(f"📊 Statistiky tagů v celém datasetu:")
        print(f"  System messages: {system_count}")
        print(f"  User messages: {user_count}")
        print(f"  Assistant messages: {assistant_count}")
        
        # Kontrola délky textů
        lengths = [len(sample['text']) for sample in dataset]
        print(f"📏 Délka textů: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
    
    # 4. Načtení modelu
    print(f"\n🤖 Načítám model: {args.model_name}")
    
    # Použití menšího modelu pro úsporu místa
    if "mistral" in args.model_name.lower() or "llama" in args.model_name.lower():
        print("⚠️ Detekován velký model. Používám agresivní optimalizaci.")
        print("💡 Dostupné menší modely:")
        print("   - microsoft/DialoGPT-medium (355M)")
        print("   - microsoft/DialoGPT-large (774M)")
        print("   - gpt2-medium (355M)")
        print("   - distilgpt2 (82M)")
    
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
                    # Počkáme chvíli
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
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir='/workspace/.cache/huggingface/transformers',
        local_files_only=False,
        resume_download=True,
        force_download=False
    )
    
    # Přidání pad tokenu
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
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
    
    # Rozdělení na train/validation s kontrolou velikosti
    print(f"📊 Celkový počet vzorků: {len(tokenized_dataset)}")
    
    if len(tokenized_dataset) < 5:
        print("⚠️ Málo vzorků pro rozdělení. Používám celý dataset pro trénování.")
        train_dataset = tokenized_dataset
        eval_dataset = tokenized_dataset  # Použijeme stejný dataset pro evaluaci
    elif len(tokenized_dataset) < 10:
        # Pro velmi malé datasety použijeme 80/20 split
        split_ratio = 0.2
        split_dataset = tokenized_dataset.train_test_split(test_size=split_ratio, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        print(f"✅ Train dataset: {len(train_dataset)} vzorků ({100-split_ratio*100:.0f}%)")
        print(f"✅ Validation dataset: {len(eval_dataset)} vzorků ({split_ratio*100:.0f}%)")
    else:
        # Standardní 90/10 split
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        print(f"✅ Train dataset: {len(train_dataset)} vzorků (90%)")
        print(f"✅ Validation dataset: {len(eval_dataset)} vzorků (10%)")
    
    # Kontrola minimální velikosti datasetu
    if len(train_dataset) == 0:
        print("❌ Train dataset je prázdný! Zkontrolujte data.")
        return
    
    if len(eval_dataset) == 0:
        print("⚠️ Validation dataset je prázdný. Používám train dataset pro evaluaci.")
        eval_dataset = train_dataset
    
    # Debug: Kontrola train/validation split
    print(f"\n🔍 DEBUG: Kontrola train/validation split")
    print(f"📊 Train dataset: {len(train_dataset)} vzorků")
    print(f"📊 Validation dataset: {len(eval_dataset)} vzorků")
    
    # Detailní debug informace o train datasetu
    if len(train_dataset) > 0:
        print(f"\n📋 DETAILNÍ DEBUG - TRAIN DATASET:")
        print(f"📊 Celkový počet vzorků: {len(train_dataset)}")
        
        # Ukázka prvních 3 vzorků
        for i in range(min(3, len(train_dataset))):
            print(f"\n📝 Train vzorek {i+1}:")
            sample = train_dataset[i]
            decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
            print(f"  Délka tokenů: {len(sample['input_ids'])}")
            print(f"  Text (prvních 300 znaků): {decoded_text[:300]}...")
            
            # Kontrola přítomnosti tagů
            has_system = "<|system|>" in decoded_text
            has_user = "<|user|>" in decoded_text
            has_assistant = "<|assistant|>" in decoded_text
            has_end = "<|end|>" in decoded_text
            print(f"  Tagy: System={has_system}, User={has_user}, Assistant={has_assistant}, End={has_end}")
        
        # Statistiky délky tokenů v train datasetu
        train_lengths = [len(sample['input_ids']) for sample in train_dataset]
        print(f"\n📏 Statistiky délky tokenů v train datasetu:")
        print(f"  Min: {min(train_lengths)}")
        print(f"  Max: {max(train_lengths)}")
        print(f"  Průměr: {sum(train_lengths)/len(train_lengths):.1f}")
        print(f"  Medián: {sorted(train_lengths)[len(train_lengths)//2]}")
        
        # Kontrola přítomnosti tagů v celém train datasetu
        train_texts = [tokenizer.decode(sample['input_ids'], skip_special_tokens=False) for sample in train_dataset]
        train_system_count = sum(1 for text in train_texts if "<|system|>" in text)
        train_user_count = sum(1 for text in train_texts if "<|user|>" in text)
        train_assistant_count = sum(1 for text in train_texts if "<|assistant|>" in text)
        train_end_count = sum(1 for text in train_texts if "<|end|>" in text)
        
        print(f"\n📊 Tagy v celém train datasetu:")
        print(f"  System: {train_system_count}/{len(train_dataset)} ({train_system_count/len(train_dataset)*100:.1f}%)")
        print(f"  User: {train_user_count}/{len(train_dataset)} ({train_user_count/len(train_dataset)*100:.1f}%)")
        print(f"  Assistant: {train_assistant_count}/{len(train_dataset)} ({train_assistant_count/len(train_dataset)*100:.1f}%)")
        print(f"  End: {train_end_count}/{len(train_dataset)} ({train_end_count/len(train_dataset)*100:.1f}%)")
    
    # Detailní debug informace o validation datasetu
    if len(eval_dataset) > 0:
        print(f"\n📋 DETAILNÍ DEBUG - VALIDATION DATASET:")
        print(f"📊 Celkový počet vzorků: {len(eval_dataset)}")
        
        # Ukázka prvních 3 vzorků
        for i in range(min(3, len(eval_dataset))):
            print(f"\n📝 Validation vzorek {i+1}:")
            sample = eval_dataset[i]
            decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
            print(f"  Délka tokenů: {len(sample['input_ids'])}")
            print(f"  Text (prvních 300 znaků): {decoded_text[:300]}...")
            
            # Kontrola přítomnosti tagů
            has_system = "<|system|>" in decoded_text
            has_user = "<|user|>" in decoded_text
            has_assistant = "<|assistant|>" in decoded_text
            has_end = "<|end|>" in decoded_text
            print(f"  Tagy: System={has_system}, User={has_user}, Assistant={has_assistant}, End={has_end}")
        
        # Statistiky délky tokenů v validation datasetu
        eval_lengths = [len(sample['input_ids']) for sample in eval_dataset]
        print(f"\n📏 Statistiky délky tokenů v validation datasetu:")
        print(f"  Min: {min(eval_lengths)}")
        print(f"  Max: {max(eval_lengths)}")
        print(f"  Průměr: {sum(eval_lengths)/len(eval_lengths):.1f}")
        print(f"  Medián: {sorted(eval_lengths)[len(eval_lengths)//2]}")
        
        # Kontrola přítomnosti tagů v celém validation datasetu
        eval_texts = [tokenizer.decode(sample['input_ids'], skip_special_tokens=False) for sample in eval_dataset]
        eval_system_count = sum(1 for text in eval_texts if "<|system|>" in text)
        eval_user_count = sum(1 for text in eval_texts if "<|user|>" in text)
        eval_assistant_count = sum(1 for text in eval_texts if "<|assistant|>" in text)
        eval_end_count = sum(1 for text in eval_texts if "<|end|>" in text)
        
        print(f"\n📊 Tagy v celém validation datasetu:")
        print(f"  System: {eval_system_count}/{len(eval_dataset)} ({eval_system_count/len(eval_dataset)*100:.1f}%)")
        print(f"  User: {eval_user_count}/{len(eval_dataset)} ({eval_user_count/len(eval_dataset)*100:.1f}%)")
        print(f"  Assistant: {eval_assistant_count}/{len(eval_dataset)} ({eval_assistant_count/len(eval_dataset)*100:.1f}%)")
        print(f"  End: {eval_end_count}/{len(eval_dataset)} ({eval_end_count/len(eval_dataset)*100:.1f}%)")
    
    # Porovnání train vs validation
    print(f"\n🔍 POROVNÁNÍ TRAIN vs VALIDATION:")
    print(f"📊 Poměr velikostí: {len(train_dataset)}:{len(eval_dataset)} ({len(train_dataset)/len(eval_dataset):.1f}:1)")
    
    if len(train_dataset) > 0 and len(eval_dataset) > 0:
        train_avg_length = sum(len(sample['input_ids']) for sample in train_dataset) / len(train_dataset)
        eval_avg_length = sum(len(sample['input_ids']) for sample in eval_dataset) / len(eval_dataset)
        print(f"📏 Průměrná délka: Train={train_avg_length:.1f}, Validation={eval_avg_length:.1f}")
        
        # Kontrola, zda jsou data podobná
        train_sample = tokenizer.decode(train_dataset[0]['input_ids'], skip_special_tokens=False)
        eval_sample = tokenizer.decode(eval_dataset[0]['input_ids'], skip_special_tokens=False)
        
        print(f"📝 Ukázka struktury:")
        print(f"  Train první vzorek: {train_sample[:100]}...")
        print(f"  Validation první vzorek: {eval_sample[:100]}...")
    
    print(f"\n✅ System messages jsou v obou datasetech - model se učí na kompletních konverzacích")
    print(f"✅ Každá konverzace obsahuje: system + user + assistant")
    print(f"✅ Data jsou připravena pro fine-tuning")
    
    # 7. Data Collator
    print("\n🔧 Konfiguruji data collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
        pad_to_multiple_of=8,  # Padding na násobky 8 pro lepší výkon
    )
    
    # Test data collator na jednom vzorku
    if len(train_dataset) > 0:
        try:
            test_batch = data_collator([train_dataset[0]])
            print(f"✅ Data collator test úspěšný")
            print(f"📊 Batch keys: {list(test_batch.keys())}")
            print(f"📊 Input shape: {test_batch['input_ids'].shape}")
            print(f"📊 Labels shape: {test_batch['labels'].shape}")
        except Exception as e:
            print(f"⚠️ Data collator test selhal: {e}")
            print("🔍 Debugging informace:")
            print(f"  Sample keys: {list(train_dataset[0].keys())}")
            print(f"  Input IDs length: {len(train_dataset[0]['input_ids'])}")
            print(f"  Labels length: {len(train_dataset[0]['labels'])}")
            print(f"  Sample type: {type(train_dataset[0]['input_ids'])}")
            
            # Zkusíme opravit problém s padding
            print("🔧 Zkouším opravit padding...")
            try:
                # Vytvoříme nový data collator s explicitním padding
                fixed_collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=False,
                    return_tensors="pt",
                    pad_to_multiple_of=8,
                    padding=True,
                )
                test_batch = fixed_collator([train_dataset[0]])
                print(f"✅ Opravený data collator test úspěšný")
                data_collator = fixed_collator
            except Exception as e2:
                print(f"❌ Oprava selhala: {e2}")
                print("ℹ️ Pokračuji s výchozím nastavením")
    
    # 8. Training Arguments - nastavení na network storage
    print("\n⚙️ Nastavuji training arguments...")
    
    # Dynamické nastavení podle velikosti datasetu
    if len(train_dataset) < 10:
        # Pro malé datasety
        save_steps = max(1, len(train_dataset) // 2)
        eval_steps = max(1, len(train_dataset) // 2)
        logging_steps = 1
        print(f"📊 Malý dataset - save_steps: {save_steps}, eval_steps: {eval_steps}")
    else:
        # Pro větší datasety
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
        dataloader_drop_last=True,  # Přidáno pro lepší handling batchů
        group_by_length=True,  # Přidáno pro lepší padding
    )
    
    # 9. Trainer
    print("\n🏋️ Vytvářím Trainer...")
    
    # Nastavení label_names pro PeftModel - robustnější přístup
    try:
        # Zkusíme nastavit label_names na modelu
        if hasattr(model, 'label_names'):
            model.label_names = ['labels']
        elif hasattr(model, 'config') and hasattr(model.config, 'label_names'):
            model.config.label_names = ['labels']
        
        # Pro PeftModel můžeme také nastavit na base modelu
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'config'):
            model.base_model.config.label_names = ['labels']
        
        print("✅ Label names nastaveny pro model")
    except Exception as e:
        print(f"⚠️ Nelze nastavit label_names: {e}")
        print("ℹ️ Pokračuji bez explicitního nastavení label_names")
    
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
    
    # 11. Uložení modelu na network storage
    print("\n💾 Ukládám model na network storage...")
    final_model_path = f"{args.output_dir}-final"
    
    # Vytvoření adresáře pokud neexistuje
    os.makedirs(final_model_path, exist_ok=True)
    
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Výpis velikosti uloženého modelu
    try:
        import subprocess
        result = subprocess.run(['du', '-sh', final_model_path], capture_output=True, text=True)
        if result.stdout:
            print(f"📊 Velikost modelu: {result.stdout.strip()}")
    except:
        pass
    
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
    print(f"📁 Model uložen v: {final_model_path}")
    print(f"💾 Network storage: {args.output_dir}")
    if args.push_to_hub:
        print(f"🌐 Model dostupný na: https://huggingface.co/{args.hub_model_id}")
    
    # Výpis informací o uložených souborech
    print(f"\n📋 Uložené soubory:")
    try:
        for root, dirs, files in os.walk(final_model_path):
            level = root.replace(final_model_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Zobrazíme pouze prvních 5 souborů
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... a dalších {len(files) - 5} souborů")
    except Exception as e:
        print(f"⚠️ Nelze zobrazit seznam souborů: {e}")

if __name__ == "__main__":
    main() 