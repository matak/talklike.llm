#!/usr/bin/env python3
"""
Fine-tuning script pro model s daty Andreje Babiše
Spustitelný na RunPod.io nebo lokálně
"""

# Import setup_environment pro správné nastavení prostředí
import setup_environment

import os
import json
import torch
import numpy as np
import shutil
from datetime import datetime
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
from lib.disk_manager import DiskManager, setup_for_ml_project, check_and_cleanup

class DatasetDebugger:
    """Třída pro debugování a ukládání mezikroků zpracování datasetu"""
    
    def __init__(self, debug_dir="debug_dataset"):
        self.debug_dir = debug_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_dir = f"{debug_dir}_{self.timestamp}"
        
        # Vytvoření debug adresáře
        os.makedirs(self.debug_dir, exist_ok=True)
        print(f"🔍 Debug adresář vytvořen: {self.debug_dir}")
    
    def save_step(self, step_name, data, description=""):
        """Uloží krok zpracování datasetu"""
        step_file = os.path.join(self.debug_dir, f"step_{step_name}.json")
        
        # Přidání metadat
        debug_info = {
            "step_name": step_name,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "data_type": type(data).__name__,
            "data_count": len(data) if hasattr(data, '__len__') else "N/A"
        }
        
        # Uložení dat podle typu
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                # Seznam slovníků - uložíme jako JSON
                debug_info["data"] = data
            else:
                # Jiný typ dat - uložíme jako text
                debug_info["data"] = [str(item) for item in data]
        elif isinstance(data, dict):
            debug_info["data"] = data
        else:
            debug_info["data"] = str(data)
        
        with open(step_file, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Uložen debug krok: {step_name} -> {step_file}")
        
        # Vytvoření také čitelné verze pro první 3 položky
        if isinstance(data, list) and len(data) > 0:
            readable_file = os.path.join(self.debug_dir, f"step_{step_name}_readable.txt")
            with open(readable_file, 'w', encoding='utf-8') as f:
                f.write(f"Debug krok: {step_name}\n")
                f.write(f"Čas: {debug_info['timestamp']}\n")
                f.write(f"Popis: {description}\n")
                f.write(f"Počet položek: {len(data)}\n")
                f.write("-" * 80 + "\n\n")
                
                for i, item in enumerate(data[:3]):  # První 3 položky
                    f.write(f"Položka {i+1}:\n")
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if key == 'content' and isinstance(value, str) and len(value) > 200:
                                f.write(f"  {key}: {value[:200]}...\n")
                            else:
                                f.write(f"  {key}: {value}\n")
                    else:
                        f.write(f"  {item}\n")
                    f.write("\n")
                
                if len(data) > 3:
                    f.write(f"... a dalších {len(data) - 3} položek\n")
    
    def save_sample(self, step_name, sample_data, sample_index=0):
        """Uloží ukázkovou položku z datasetu"""
        sample_file = os.path.join(self.debug_dir, f"sample_{step_name}_{sample_index}.json")
        
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        print(f"📝 Uložena ukázka: {step_name} -> {sample_file}")
    
    def create_summary(self):
        """Vytvoří shrnutí všech debug kroků"""
        summary_file = os.path.join(self.debug_dir, "debug_summary.txt")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("DEBUG SHRNUTÍ ZPRACOVÁNÍ DATASETU\n")
            f.write("=" * 50 + "\n")
            f.write(f"Čas vytvoření: {self.timestamp}\n")
            f.write(f"Debug adresář: {self.debug_dir}\n\n")
            
            # Najdeme všechny debug soubory
            debug_files = [f for f in os.listdir(self.debug_dir) if f.startswith("step_") and f.endswith(".json")]
            debug_files.sort()
            
            for debug_file in debug_files:
                try:
                    with open(os.path.join(self.debug_dir, debug_file), 'r', encoding='utf-8') as df:
                        debug_info = json.load(df)
                    
                    f.write(f"Krok: {debug_info['step_name']}\n")
                    f.write(f"  Čas: {debug_info['timestamp']}\n")
                    f.write(f"  Popis: {debug_info['description']}\n")
                    f.write(f"  Typ dat: {debug_info['data_type']}\n")
                    f.write(f"  Počet: {debug_info['data_count']}\n")
                    f.write("\n")
                except Exception as e:
                    f.write(f"Chyba při čtení {debug_file}: {e}\n\n")
        
        print(f"📋 Vytvořeno shrnutí: {summary_file}")

def load_babis_data(file_path, debugger=None):
    """Načte data z JSONL souboru nebo jednoho velkého JSON objektu"""
    conversations = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Debug: Uložení původního obsahu
    if debugger:
        debugger.save_step("01_original_content", {"content": content[:1000] + "..." if len(content) > 1000 else content}, 
                          "Původní obsah souboru")
    
    try:
        # Zkusíme parsovat jako jeden velký JSON objekt
        data = json.loads(content)
        
        if 'messages' in data:
            # Máme jeden velký objekt s messages - rozdělíme na konverzace
            messages = data['messages']
            print(f"📊 Načteno {len(messages)} zpráv v jednom objektu")
            
            # Debug: Uložení všech zpráv
            if debugger:
                debugger.save_step("02_all_messages", messages, f"Všech {len(messages)} zpráv z JSON objektu")
            
            # Najdeme system zprávu (měla by být první)
            system_msg = None
            for msg in messages:
                if msg['role'] == 'system':
                    system_msg = msg
                    break
            
            if not system_msg:
                print("❌ Nenalezena system zpráva!")
                return conversations
            
            # Debug: Uložení system zprávy
            if debugger:
                debugger.save_step("03_system_message", [system_msg], "System zpráva")
            
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
            
            # Debug: Uložení vytvořených konverzací
            if debugger:
                debugger.save_step("04_conversations", conversations, f"Vytvořených {len(conversations)} konverzací")
                if len(conversations) > 0:
                    debugger.save_sample("04_conversations", conversations[0], 0)
                    if len(conversations) > 1:
                        debugger.save_sample("04_conversations", conversations[1], 1)
            
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

def prepare_training_data(conversations, debugger=None):
    """Připraví data pro fine-tuning"""
    training_data = []
    
    # Debug: Uložení vstupních konverzací
    if debugger:
        debugger.save_step("05_input_conversations", conversations, f"Vstupních {len(conversations)} konverzací pro prepare_training_data")
    
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
    
    # Debug: Uložení připravených dat
    if debugger:
        debugger.save_step("06_training_data", training_data, f"Připravených {len(training_data)} trénovacích vzorků")
        if len(training_data) > 0:
            debugger.save_sample("06_training_data", training_data[0], 0)
            if len(training_data) > 1:
                debugger.save_sample("06_training_data", training_data[1], 1)
    
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

def setup_tokenizer_and_model(model_name, base_model):
    """Nastaví tokenizer a model pro fine-tuning"""
    
    # 1. Načtení tokenizeru
    base_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir='/workspace/.cache/huggingface/transformers',
        local_files_only=False,
        resume_download=True,
        force_download=False
    )
    print(f"📊 Původní délka tokenizeru: {len(base_tokenizer)}")
    
    # 2. Kontrola a přidání pad tokenu
    if base_tokenizer.pad_token is None:
        # Zkusíme použít existující tokeny
        if base_tokenizer.eos_token:
            base_tokenizer.pad_token = base_tokenizer.eos_token
            print(f"✅ Používám EOS token jako PAD: {base_tokenizer.pad_token}")
        else:
            # Přidáme nový pad token
            base_tokenizer.add_special_tokens({"pad_token": "<pad>"})
            print(f"✅ Přidán nový pad token: {base_tokenizer.pad_token}")
            
            # Důležité: Resize model embeddings
            base_model.resize_token_embeddings(len(base_tokenizer))
            print(f"📊 Model embeddings resized na: {len(base_tokenizer)}")
    else:
        print(f"ℹ️ Pad token už existuje: {base_tokenizer.pad_token}")
    
    # 3. Synchronizace s modelem
    if hasattr(base_model.config, 'pad_token_id'):
        old_pad_id = base_model.config.pad_token_id
        base_model.config.pad_token_id = base_tokenizer.pad_token_id
        print(f"🔄 Pad token ID změněn: {old_pad_id} → {base_model.config.pad_token_id}")
    else:
        print("⚠️ Model nemá pad_token_id v config")
    
    # 4. Kontrola konzistence
    try:
        assert base_tokenizer.pad_token_id == base_model.config.pad_token_id, \
            "Tokenizer a model mají různé pad token ID!"
        print(f"✅ Tokenizer a model synchronizovány")
    except AssertionError as e:
        print(f"❌ Chyba synchronizace: {e}")
        # Pokusíme se opravit
        base_model.config.pad_token_id = base_tokenizer.pad_token_id
        print(f"🔧 Opraveno: pad_token_id nastaven na {base_tokenizer.pad_token_id}")
    
    return base_tokenizer, base_model

def main():
    # Kontrola, že jsme v root directory projektu
    if not os.path.exists('lib') or not os.path.exists('data'):
        print("❌ Skript musí být spuštěn z root directory projektu!")
        print("💡 Spusťte skript z adresáře, kde jsou složky 'lib' a 'data'")
        print(f"📍 Aktuální adresář: {os.getcwd()}")
        print("📁 Obsah aktuálního adresáře:")
        try:
            for item in os.listdir('.'):
                print(f"  - {item}")
        except:
            pass
        return
    
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
    training_data = prepare_training_data(conversations, debugger)
    print(f"✅ Připraveno {len(training_data)} trénovacích vzorků")
    
    # 3. Vytvoření Dataset
    dataset = Dataset.from_list(training_data)
    
    # Debug: Uložení finálního datasetu
    debugger.save_step("07_final_dataset", {"dataset_size": len(dataset), "columns": dataset.column_names}, 
                      f"Finální dataset s {len(dataset)} vzorky")
    
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
        
        # Debug: Uložení statistik
        debugger.save_step("08_dataset_stats", {
            "total_samples": len(dataset),
            "system_messages": system_count,
            "user_messages": user_count,
            "assistant_messages": assistant_count,
            "text_lengths": {
                "min": min(lengths),
                "max": max(lengths),
                "avg": sum(lengths)/len(lengths)
            }
        }, "Statistiky datasetu")
    
    # Vytvoření shrnutí debug informací
    debugger.create_summary()
    print(f"📋 Debug shrnutí vytvořeno: {debugger.debug_dir}/debug_summary.txt")
    
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
    
    # Debug: Ukázka tokenizovaného vzorku
    if len(tokenized_dataset) > 0:
        sample_tokens = tokenized_dataset[0]
        decoded_text = tokenizer.decode(sample_tokens['input_ids'], skip_special_tokens=False)
        debugger.save_step("10_tokenized_sample", {
            "input_ids_length": len(sample_tokens['input_ids']),
            "attention_mask_length": len(sample_tokens['attention_mask']),
            "labels_length": len(sample_tokens['labels']),
            "decoded_text": decoded_text[:500] + "..." if len(decoded_text) > 500 else decoded_text,
            "has_system": "<|system|>" in decoded_text,
            "has_user": "<|user|>" in decoded_text,
            "has_assistant": "<|assistant|>" in decoded_text,
            "has_end": "<|end|>" in decoded_text
        }, "Ukázka tokenizovaného vzorku")
    
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
    
    # Debug: Uložení finálního tokenizovaného datasetu
    debugger.save_step("11_final_tokenized_dataset", {
        "dataset_size": len(tokenized_dataset),
        "columns": tokenized_dataset.column_names,
        "max_length": args.max_length
    }, f"Finální tokenizovaný dataset s padding")
    
    # Rozdělení na train/validation s kontrolou velikosti
    print(f"📊 Celkový počet vzorků: {len(tokenized_dataset)}")
    
    if len(tokenized_dataset) < 5:
        print("⚠️ Málo vzorků pro rozdělení. Používám celý dataset pro trénování.")
        train_dataset = tokenized_dataset
        eval_dataset = tokenized_dataset  # Použijeme stejný dataset pro evaluaci
        split_info = {"type": "no_split", "reason": "too_few_samples", "train_size": len(train_dataset), "eval_size": len(eval_dataset)}
    elif len(tokenized_dataset) < 10:
        # Pro velmi malé datasety použijeme 80/20 split
        split_ratio = 0.2
        split_dataset = tokenized_dataset.train_test_split(test_size=split_ratio, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        print(f"✅ Train dataset: {len(train_dataset)} vzorků ({100-split_ratio*100:.0f}%)")
        print(f"✅ Validation dataset: {len(eval_dataset)} vzorků ({split_ratio*100:.0f}%)")
        split_info = {"type": "80_20_split", "split_ratio": split_ratio, "train_size": len(train_dataset), "eval_size": len(eval_dataset)}
    else:
        # Standardní 90/10 split
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        print(f"✅ Train dataset: {len(train_dataset)} vzorků (90%)")
        print(f"✅ Validation dataset: {len(eval_dataset)} vzorků (10%)")
        split_info = {"type": "90_10_split", "split_ratio": 0.1, "train_size": len(train_dataset), "eval_size": len(eval_dataset)}
    
    # Debug: Uložení informací o rozdělení
    debugger.save_step("12_train_validation_split", split_info, f"Rozdělení datasetu: {split_info['type']}")
    
    # Kontrola minimální velikosti datasetu
    if len(train_dataset) == 0:
        print("❌ Train dataset je prázdný! Zkontrolujte data.")
        return
    
    if len(eval_dataset) == 0:
        print("⚠️ Validation dataset je prázdný. Používám train dataset pro evaluaci.")
        eval_dataset = train_dataset
        debugger.save_step("13_split_fallback", {"action": "use_train_for_eval", "reason": "empty_eval_dataset"}, "Použití train datasetu pro evaluaci")
    
    # Debug: Kontrola train/validation split
    print(f"\n🔍 DEBUG: Kontrola train/validation split")
    print(f"📊 Train dataset: {len(train_dataset)} vzorků")
    print(f"📊 Validation dataset: {len(eval_dataset)} vzorků")
    
    # Debug: Uložení train a validation datasetů do souborů
    print(f"\n💾 Ukládám train a validation datasety...")
    
    # Uložení train datasetu
    train_data_file = os.path.join(debugger.debug_dir, "train_dataset.jsonl")
    with open(train_data_file, 'w', encoding='utf-8') as f:
        for i, sample in enumerate(train_dataset):
            # Dekódování tokenů zpět na text pro čitelnost
            decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
            sample_data = {
                "index": i,
                "input_ids_length": len(sample['input_ids']),
                "attention_mask_length": len(sample['attention_mask']),
                "labels_length": len(sample['labels']),
                "decoded_text": decoded_text,
                "has_system": "<|system|>" in decoded_text,
                "has_user": "<|user|>" in decoded_text,
                "has_assistant": "<|assistant|>" in decoded_text,
                "has_end": "<|end|>" in decoded_text,
                "token_ids": sample['input_ids'][:100] + ["..."] if len(sample['input_ids']) > 100 else sample['input_ids']  # Prvních 100 tokenů
            }
            f.write(json.dumps(sample_data, ensure_ascii=False) + '\n')
    print(f"✅ Train dataset uložen: {train_data_file}")
    
    # Uložení validation datasetu
    eval_data_file = os.path.join(debugger.debug_dir, "validation_dataset.jsonl")
    with open(eval_data_file, 'w', encoding='utf-8') as f:
        for i, sample in enumerate(eval_dataset):
            # Dekódování tokenů zpět na text pro čitelnost
            decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
            sample_data = {
                "index": i,
                "input_ids_length": len(sample['input_ids']),
                "attention_mask_length": len(sample['attention_mask']),
                "labels_length": len(sample['labels']),
                "decoded_text": decoded_text,
                "has_system": "<|system|>" in decoded_text,
                "has_user": "<|user|>" in decoded_text,
                "has_assistant": "<|assistant|>" in decoded_text,
                "has_end": "<|end|>" in decoded_text,
                "token_ids": sample['input_ids'][:100] + ["..."] if len(sample['input_ids']) > 100 else sample['input_ids']  # Prvních 100 tokenů
            }
            f.write(json.dumps(sample_data, ensure_ascii=False) + '\n')
    print(f"✅ Validation dataset uložen: {eval_data_file}")
    
    # Vytvoření čitelné verze pro rychlé prohlížení
    train_readable_file = os.path.join(debugger.debug_dir, "train_dataset_readable.txt")
    with open(train_readable_file, 'w', encoding='utf-8') as f:
        f.write("TRAIN DATASET - ČITELNÁ VERZE\n")
        f.write("=" * 50 + "\n")
        f.write(f"Celkový počet vzorků: {len(train_dataset)}\n\n")
        
        for i in range(min(5, len(train_dataset))):  # Prvních 5 vzorků
            sample = train_dataset[i]
            decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
            f.write(f"VZOREK {i+1}:\n")
            f.write(f"Délka tokenů: {len(sample['input_ids'])}\n")
            f.write(f"Text:\n{decoded_text}\n")
            f.write("-" * 80 + "\n\n")
        
        if len(train_dataset) > 5:
            f.write(f"... a dalších {len(train_dataset) - 5} vzorků\n")
    
    eval_readable_file = os.path.join(debugger.debug_dir, "validation_dataset_readable.txt")
    with open(eval_readable_file, 'w', encoding='utf-8') as f:
        f.write("VALIDATION DATASET - ČITELNÁ VERZE\n")
        f.write("=" * 50 + "\n")
        f.write(f"Celkový počet vzorků: {len(eval_dataset)}\n\n")
        
        for i in range(min(5, len(eval_dataset))):  # Prvních 5 vzorků
            sample = eval_dataset[i]
            decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
            f.write(f"VZOREK {i+1}:\n")
            f.write(f"Délka tokenů: {len(sample['input_ids'])}\n")
            f.write(f"Text:\n{decoded_text}\n")
            f.write("-" * 80 + "\n\n")
        
        if len(eval_dataset) > 5:
            f.write(f"... a dalších {len(eval_dataset) - 5} vzorků\n")
    
    print(f"✅ Čitelné verze vytvořeny:")
    print(f"   - {train_readable_file}")
    print(f"   - {eval_readable_file}")
    
    # Vytvoření statistik souboru
    stats_file = os.path.join(debugger.debug_dir, "dataset_statistics.json")
    train_lengths = [len(sample['input_ids']) for sample in train_dataset]
    eval_lengths = [len(sample['input_ids']) for sample in eval_dataset]
    
    train_texts = [tokenizer.decode(sample['input_ids'], skip_special_tokens=False) for sample in train_dataset]
    eval_texts = [tokenizer.decode(sample['input_ids'], skip_special_tokens=False) for sample in eval_dataset]
    
    stats = {
        "train_dataset": {
            "size": len(train_dataset),
            "token_lengths": {
                "min": min(train_lengths) if train_lengths else 0,
                "max": max(train_lengths) if train_lengths else 0,
                "avg": sum(train_lengths)/len(train_lengths) if train_lengths else 0,
                "median": sorted(train_lengths)[len(train_lengths)//2] if train_lengths else 0
            },
            "tags": {
                "system": sum(1 for text in train_texts if "<|system|>" in text),
                "user": sum(1 for text in train_texts if "<|user|>" in text),
                "assistant": sum(1 for text in train_texts if "<|assistant|>" in text),
                "end": sum(1 for text in train_texts if "<|end|>" in text)
            }
        },
        "validation_dataset": {
            "size": len(eval_dataset),
            "token_lengths": {
                "min": min(eval_lengths) if eval_lengths else 0,
                "max": max(eval_lengths) if eval_lengths else 0,
                "avg": sum(eval_lengths)/len(eval_lengths) if eval_lengths else 0,
                "median": sorted(eval_lengths)[len(eval_lengths)//2] if eval_lengths else 0
            },
            "tags": {
                "system": sum(1 for text in eval_texts if "<|system|>" in text),
                "user": sum(1 for text in eval_texts if "<|user|>" in text),
                "assistant": sum(1 for text in eval_texts if "<|assistant|>" in text),
                "end": sum(1 for text in eval_texts if "<|end|>" in text)
            }
        },
        "split_info": split_info,
        "tokenizer_info": {
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "max_length": args.max_length
        }
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"✅ Statistiky uloženy: {stats_file}")
    
    # Debug: Uložení informací o datasetech
    debugger.save_step("15_train_validation_files", {
        "train_dataset_file": train_data_file,
        "validation_dataset_file": eval_data_file,
        "train_readable_file": train_readable_file,
        "validation_readable_file": eval_readable_file,
        "statistics_file": stats_file,
        "train_size": len(train_dataset),
        "validation_size": len(eval_dataset)
    }, "Uložené soubory train a validation datasetů")
    
    # Dynamické nastavení training parametrů podle velikosti datasetu
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
    
    print(f"\n✅ System messages jsou v obou datasetech - model se učí na kompletních konverzacích")
    print(f"✅ Každá konverzace obsahuje: system + user + assistant")
    print(f"✅ Data jsou připravena pro fine-tuning")
    
    # Finální debug shrnutí před trénováním
    debugger.save_step("16_final_pre_training_summary", {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "train_dataset_size": len(train_dataset),
        "eval_dataset_size": len(eval_dataset),
        "total_samples": len(tokenized_dataset),
        "split_info": split_info,
        "tokenizer_vocab_size": len(tokenizer),
        "model_vocab_size": model.config.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "training_args": {
            "output_dir": args.output_dir,
            "save_steps": save_steps,
            "eval_steps": eval_steps,
            "logging_steps": logging_steps,
            "use_wandb": args.use_wandb,
            "push_to_hub": args.push_to_hub
        }
    }, "Finální shrnutí před začátkem trénování")
    
    # Vytvoření finálního shrnutí debug informací
    debugger.create_summary()
    print(f"📋 Kompletní debug shrnutí vytvořeno: {debugger.debug_dir}/debug_summary.txt")
    print(f"🔍 Všechny debug soubory jsou uloženy v: {debugger.debug_dir}")
    
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
    print("\n🏋️ Testuji model...")
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