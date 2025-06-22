#!/usr/bin/env python3
"""
Interaktivní chat s lokálním fine-tunovaným Babiš modelem
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
        "/workspace/babis-finetuned-final",
        "/workspace/babis-finetuned",
        "./babis-finetuned-final",
        "./babis-finetuned"
    ]
    
    print("🔍 Hledám lokální fine-tunovaný model...")
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Nalezen model v: {path}")
            return path
    
    print("❌ Lokální model nebyl nalezen v očekávaných adresářích:")
    for path in possible_paths:
        print(f"   - {path}")
    
    # Zkusíme najít jakékoliv adresáře s "babis" nebo "finetuned"
    print("\n🔍 Hledám jiné možné adresáře...")
    workspace_dirs = []
    if os.path.exists("/workspace"):
        workspace_dirs = [d for d in os.listdir("/workspace") if os.path.isdir(os.path.join("/workspace", d))]
    
    current_dirs = [d for d in os.listdir(".") if os.path.isdir(d)]
    
    babis_dirs = []
    for d in workspace_dirs + current_dirs:
        if "babis" in d.lower() or "finetuned" in d.lower():
            babis_dirs.append(d)
    
    if babis_dirs:
        print("📁 Nalezené možné adresáře s modely:")
        for d in babis_dirs:
            print(f"   - {d}")
        return babis_dirs[0]  # Vrátíme první nalezený
    
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
            
            # Načtení base modelu
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
            tokenizer, model = setup_tokenizer_and_model(base_model, model)
            
            # Načtení adaptéru
            print(f"🔧 Načítám adaptér z: {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            
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
        
        print("✅ Lokální model úspěšně načten!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Chyba při načítání lokálního modelu: {e}")
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

def main():
    """Hlavní funkce pro interaktivní chat s lokálním modelem"""
    print("🎭 CHAT S LOKÁLNÍM BABIŠ MODELEM")
    print("=" * 50)
    print("🤖 Fine-tunovaný model (lokální)")
    print("=" * 50)
    
    # Hledání lokálního modelu
    model_path = find_local_model()
    
    if model_path is None:
        print("❌ Nepodařilo se najít lokální model.")
        print("\n💡 Možná řešení:")
        print("1. Spusťte fine-tuning: python finetune.py")
        print("2. Zkontrolujte, zda je model uložen v /workspace/")
        print("3. Zadejte cestu k modelu ručně")
        return
    
    # Načtení modelu
    model, tokenizer = load_local_model(model_path)
    
    if model is None or tokenizer is None:
        print("❌ Nepodařilo se načíst model. Ukončuji.")
        return
    
    print(f"\n💬 Můžete začít povídat s lokálním Babiš modelem!")
    print(f"📁 Model načten z: {model_path}")
    print("📝 Napište svůj dotaz a stiskněte Enter")
    print("🔧 Pro ukončení napište 'konec' nebo stiskněte Ctrl+C")
    print("=" * 50)
    
    # Nekonečná smyčka pro dotazování
    while True:
        try:
            # Vstup uživatele
            user_input = input("\n👤 Vy: ").strip()
            
            # Kontrola ukončení
            if user_input.lower() in ['konec', 'exit', 'quit', 'stop']:
                print("👋 Na shledanou!")
                break
            
            # Prázdný vstup
            if not user_input:
                continue
            
            # Generování odpovědi
            print("🤖 Lokální model přemýšlí...")
            response = generate_local_response(model, tokenizer, user_input)
            
            print(f"🎭 Lokální Babiš: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Na shledanou!")
            break
        except Exception as e:
            print(f"❌ Chyba: {e}")

if __name__ == "__main__":
    main() 