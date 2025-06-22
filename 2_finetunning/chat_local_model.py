#!/usr/bin/env python3
"""
Interaktivní chat s lokálním fine-tunovaným modelem
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import setup_environment

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tokenizer_utils import setup_tokenizer_and_model

def find_local_model():
    """Najde lokální fine-tunovaný model"""
    possible_paths = [
        "/workspace/mistral-babis-finetuned-final",
        "/workspace/mistral-babis-finetuned"
    ]
    
    available_models = []
    
    print("🔍 Hledám lokální fine-tunované modely...")
    
    for path in possible_paths:
        if os.path.exists(path):
            # Kontrola, zda obsahuje adapter_config.json (je to PeftModel)
            if os.path.exists(os.path.join(path, "adapter_config.json")):
                available_models.append(path)
                print(f"✅ Nalezen model v: {path}")
            else:
                print(f"⚠️  Nalezen adresář, ale není to PeftModel: {path}")
    
    if not available_models:
        print("❌ Žádné lokální modely nebyly nalezeny")
        return None
    
    if len(available_models) == 1:
        print(f"🎯 Automaticky vybrán model: {available_models[0]}")
        return available_models[0]
    
    # Výběr modelu, pokud je jich více
    print("\n📋 Dostupné modely:")
    for i, path in enumerate(available_models, 1):
        model_name = os.path.basename(path)
        print(f"  {i}. {model_name} ({path})")
    
    while True:
        try:
            choice = input(f"\n🎯 Vyberte model (1-{len(available_models)}): ").strip()
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(available_models):
                selected_model = available_models[choice_idx]
                print(f"✅ Vybrán model: {selected_model}")
                return selected_model
            else:
                print(f"❌ Neplatný výběr. Zadejte číslo 1-{len(available_models)}")
        except ValueError:
            print("❌ Neplatný vstup. Zadejte číslo.")
        except KeyboardInterrupt:
            print("\n👋 Ukončuji...")
            return None

def load_local_model(model_path):
    """Načte lokální fine-tunovaný model"""
    print(f"🤖 Načítám lokální model z: {model_path}")
    
    try:
        # Kontrola, zda je to PeftModel (adaptér) nebo kompletní model
        config_files = os.listdir(model_path)
        
        if "adapter_config.json" in config_files:
            # Je to PeftModel - potřebujeme base model
            print("🔧 Detekován PeftModel (adaptér) - načítám base model...")
            
            # Načtení konfigurace adaptéru
            import json
            with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
                adapter_config = json.load(f)
            
            base_model = adapter_config.get('base_model_name_or_path', 'mistralai/Mistral-7B-Instruct-v0.3')
            print(f"📝 Base model z konfigurace: {base_model}")
            
            # Zobrazení informací o adaptéru
            if 'target_modules' in adapter_config:
                print(f"🎯 Target modules: {adapter_config['target_modules']}")
            if 'lora_alpha' in adapter_config:
                print(f"🔢 LoRA alpha: {adapter_config['lora_alpha']}")
            if 'r' in adapter_config:
                print(f"📊 LoRA rank (r): {adapter_config['r']}")
            
            # Načtení base modelu
            print("📥 Načítám base model...")
            tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=True,
                cache_dir='/workspace/.cache/huggingface/transformers'
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                cache_dir='/workspace/.cache/huggingface/transformers'
            )
            
            # Nastavení pad_tokenu
            print("🔧 Nastavuji tokenizer...")
            tokenizer, model = setup_tokenizer_and_model(base_model, model)
            
            # Načtení adaptéru
            print(f"🔧 Načítám adaptér z: {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            
            # Zobrazení informací o modelu
            print(f"📊 Model načten na zařízení: {next(model.parameters()).device}")
            print(f"🧮 Počet parametrů: {sum(p.numel() for p in model.parameters()):,}")
            
        else:
            # Je to kompletní model
            print("🧠 Detekován kompletní model - načítám přímo...")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Zobrazení informací o modelu
            print(f"📊 Model načten na zařízení: {next(model.parameters()).device}")
            print(f"🧮 Počet parametrů: {sum(p.numel() for p in model.parameters()):,}")
        
        print("✅ Lokální model úspěšně načten!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Chyba při načítání lokálního modelu: {e}")
        print(f"📋 Detaily chyby: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_local_response(model, tokenizer, prompt, max_length=300, temperature=0.8):
    """Generuje odpověď pomocí lokálního modelu"""
    try:
        # Kontrola, zda tokenizer podporuje apply_chat_template
        if hasattr(tokenizer, 'apply_chat_template'):
            # Použijeme apply_chat_template pro správné formátování
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback pro tokenizery bez apply_chat_template
            formatted_prompt = prompt
        
        # Tokenizace vstupu
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Přesun na správné zařízení
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generování
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Dekódování odpovědi
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Odstranění původního promptu z odpovědi
        if response.startswith(formatted_prompt):
            response = response[len(formatted_prompt):].strip()
        
        return response
        
    except Exception as e:
        return f"❌ Chyba při generování: {e}"

def check_gpu_memory():
    """Zkontroluje dostupnou GPU paměť"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"🎮 Nalezeno {gpu_count} GPU zařízení:")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            memory_free = torch.cuda.memory_reserved(i) / 1024**3
            memory_used = memory_total - memory_free
            
            print(f"  GPU {i}: {gpu_name}")
            print(f"    💾 Paměť: {memory_used:.1f}GB / {memory_total:.1f}GB")
        
        return True
    else:
        print("⚠️  GPU není dostupné - model bude běžet na CPU")
        return False

def main():
    """Hlavní funkce pro interaktivní chat s lokálním modelem"""
    print("🎭 CHAT S LOKÁLNÍM FINE-TUNOVANÝM MODELEM")
    print("=" * 50)
    print("🤖 Fine-tunovaný model (lokální)")
    print("=" * 50)
    
    # Kontrola GPU paměti
    check_gpu_memory()
    print()
    
    # Hledání lokálního modelu
    model_path = find_local_model()
    
    if model_path is None:
        print("❌ Nepodařilo se najít lokální model.")
        print("\n💡 Možná řešení:")
        print("1. Spusťte fine-tuning: python finetune.py")
        print("2. Zkontrolujte, zda jsou modely uloženy v:")
        print("   - /workspace/mistral-babis-finetuned")
        print("   - /workspace/mistral-babis-finetuned-final")
        print("3. Zadejte cestu k modelu ručně")
        return
    
    # Načtení modelu
    model, tokenizer = load_local_model(model_path)
    
    if model is None or tokenizer is None:
        print("❌ Nepodařilo se načíst model. Ukončuji.")
        return
    
    print(f"\n💬 Můžete začít povídat s lokálním fine-tunovaným modelem!")
    print(f"📁 Model načten z: {model_path}")
    print("📝 Napište svůj dotaz a stiskněte Enter")
    print("🔧 Pro ukončení napište 'konec' nebo stiskněte Ctrl+C")
    print("⚙️  Pro změnu parametrů napište 'nastaveni'")
    print("=" * 50)
    
    # Parametry generování
    max_length = 300
    temperature = 0.8
    
    # Nekonečná smyčka pro dotazování
    while True:
        try:
            # Vstup uživatele
            user_input = input("\n👤 Vy: ").strip()
            
            # Kontrola ukončení
            if user_input.lower() in ['konec', 'exit', 'quit', 'stop']:
                print("👋 Na shledanou!")
                break
            
            # Kontrola nastavení
            if user_input.lower() in ['nastaveni', 'settings', 'config']:
                print(f"\n⚙️  Aktuální nastavení:")
                print(f"   📏 Max délka odpovědi: {max_length}")
                print(f"   🌡️  Teplota: {temperature}")
                
                try:
                    new_max = input(f"   📏 Nová max délka ({max_length}): ").strip()
                    if new_max:
                        max_length = int(new_max)
                    
                    new_temp = input(f"   🌡️  Nová teplota ({temperature}): ").strip()
                    if new_temp:
                        temperature = float(new_temp)
                    
                    print("✅ Nastavení aktualizováno!")
                except ValueError:
                    print("❌ Neplatná hodnota. Nastavení zůstává beze změny.")
                continue
            
            # Prázdný vstup
            if not user_input:
                continue
            
            # Generování odpovědi
            print("🤖 Lokální model přemýšlí...")
            response = generate_local_response(model, tokenizer, user_input, max_length, temperature)
            
            print(f"🎭 Model: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Na shledanou!")
            break
        except Exception as e:
            print(f"❌ Chyba: {e}")

if __name__ == "__main__":
    main() 