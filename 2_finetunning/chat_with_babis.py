#!/usr/bin/env python3
"""
Interaktivní chat s fine-tunovaným Babiš modelem
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import setup_environment

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tokenizer_utils import setup_tokenizer_and_model

def load_babis_model():
    """Načte fine-tunovaný Babiš model"""
    print("🤖 Načítám Babiš model...")
    
    base_model = "mistralai/Mistral-7B-Instruct-v0.3"
    adapter_path = "mcmatak/babis-mistral-adapter"
    
    try:
        # Načtení tokenizeru
        print(f"📝 Načítám tokenizer: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            cache_dir='/workspace/.cache/huggingface/transformers'
        )
        
        # Načtení base modelu
        print(f"🧠 Načítám base model: {base_model}")
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
        print(f"🔧 Načítám Babiš adaptér: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        print("✅ Babiš model úspěšně načten!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Chyba při načítání modelu: {e}")
        return None, None

def generate_babis_response(model, tokenizer, prompt, max_length=300, temperature=0.8):
    """Generuje odpověď ve stylu Andreje Babiše"""
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
    """Hlavní funkce pro interaktivní chat"""
    print("🎭 CHAT S ANDREJEM BABIŠEM")
    print("=" * 50)
    print("🤖 Fine-tunovaný model na Mistral-7B-Instruct-v0.3")
    print("🔧 Adaptér: mcmatak/babis-mistral-adapter")
    print("=" * 50)
    
    # Načtení modelu
    model, tokenizer = load_babis_model()
    
    if model is None or tokenizer is None:
        print("❌ Nepodařilo se načíst model. Ukončuji.")
        return
    
    print("\n💬 Můžete začít povídat s Andrejem Babišem!")
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
            print("🤖 Andrej Babiš přemýšlí...")
            response = generate_babis_response(model, tokenizer, user_input)
            
            print(f"🎭 Andrej Babiš: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Na shledanou!")
            break
        except Exception as e:
            print(f"❌ Chyba: {e}")

if __name__ == "__main__":
    main() 