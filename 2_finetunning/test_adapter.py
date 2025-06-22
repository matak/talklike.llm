#!/usr/bin/env python3
"""
Skript pro testování LoRA adaptéru s různými modely
Umožňuje snadné připojení adaptéru k jakémukoli kompatibilnímu modelu
"""

# Import setup_environment pro správné nastavení prostředí
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import setup_environment

import argparse
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import warnings

# Import centralizované funkce pro nastavení pad_tokenu
from tokenizer_utils import setup_tokenizer_and_model

# Potlačení varování
warnings.filterwarnings("ignore")

def load_adapter_config(adapter_path):
    """Načte konfiguraci adaptéru"""
    config_path = adapter_path.replace("/", "\\").replace("\\", "/") + "_config.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Konfigurační soubor nenalezen: {config_path}")
        return None

def load_adapter_model(base_model_name, adapter_path, device="auto"):
    """Načte base model s QLoRA adaptérem"""
    try:
        print(f"🤖 Načítám base model: {base_model_name}")
        print(f"🔧 Načítám adaptér: {adapter_path}")
        
        # Načtení tokenizeru
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        
        # Načtení base modelu
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        
        # Použití centralizované funkce pro nastavení pad_tokenu
        tokenizer, model = setup_tokenizer_and_model(base_model_name, model)
        
        # Načtení adaptéru
        print(f"🔧 Načítám QLoRA adaptér...")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        print("✅ Model s adaptérem úspěšně načten!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Chyba při načítání modelu s adaptérem: {e}")
        return None, None

def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7):
    """Generuje odpověď na základě promptu"""
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
            max_length=max_length
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

def test_adapter_compatibility(adapter_path):
    """Testuje kompatibilitu adaptéru s různými modely"""
    print("🔍 Testuji kompatibilitu adaptéru...")
    
    # Seznam populárních modelů k testování
    test_models = [
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-large", 
        "gpt2",
        "gpt2-medium",
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/gpt-neo-1.3B"
    ]
    
    config = load_adapter_config(adapter_path)
    if not config:
        print("❌ Nelze načíst konfiguraci adaptéru")
        return
    
    original_base_model = config.get('base_model', 'unknown')
    print(f"📊 Adaptér byl trénován na modelu: {original_base_model}")
    
    print(f"\n🧪 Testuji kompatibilitu s různými modely:")
    
    for model_name in test_models:
        print(f"\n🔬 Testuji: {model_name}")
        try:
            # Rychlý test načtení
            model, tokenizer = load_adapter_model(model_name, adapter_path, device="cpu")
            if model and tokenizer:
                print(f"✅ Kompatibilní s {model_name}")
                
                # Rychlý test generování
                test_prompt = "Jak se máš?"
                response = generate_response(model, tokenizer, test_prompt, max_length=50, temperature=0.7)
                print(f"   Test odpověď: {response[:100]}...")
                
                # Uvolnění paměti
                del model, tokenizer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            else:
                print(f"❌ Nekompatibilní s {model_name}")
                
        except Exception as e:
            print(f"❌ Chyba s {model_name}: {str(e)[:100]}...")

def main():
    parser = argparse.ArgumentParser(
        description="Testování LoRA adaptéru s různými modely",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Příklady použití:
  python test_adapter.py --base-model microsoft/DialoGPT-medium --adapter ./adapters/babis_adapter
  python test_adapter.py --base-model gpt2 --adapter ./adapters/babis_adapter
  python test_adapter.py --adapter ./adapters/babis_adapter --test-compatibility
        """
    )
    
    parser.add_argument(
        "--base-model",
        help="Základní model (pokud není specifikován, použije se model z konfigurace adaptéru)"
    )
    
    parser.add_argument(
        "--adapter",
        required=True,
        help="Cesta k LoRA adaptéru"
    )
    
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Zařízení pro inference (default: auto)"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximální délka generované odpovědi (default: 512)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Teplota pro generování (default: 0.7)"
    )
    
    parser.add_argument(
        "--test-compatibility",
        action="store_true",
        help="Testuje kompatibilitu adaptéru s různými modely"
    )
    
    args = parser.parse_args()
    
    if args.test_compatibility:
        test_adapter_compatibility(args.adapter)
        return
    
    # Určení základního modelu
    base_model = args.base_model
    if not base_model:
        config = load_adapter_config(args.adapter)
        if config:
            base_model = config.get('base_model')
            print(f"📊 Používám základní model z konfigurace: {base_model}")
        else:
            print("❌ Musíte specifikovat --base-model nebo mít konfigurační soubor")
            sys.exit(1)
    
    # Načtení modelu s adaptérem
    model, tokenizer = load_adapter_model(base_model, args.adapter, args.device)
    
    if model is None or tokenizer is None:
        sys.exit(1)
    
    print("\n" + "="*60)
    print("💬 INTERAKTIVNÍ TESTOVÁNÍ ADAPTÉRU")
    print("="*60)
    print("📝 Napište svůj dotaz a stiskněte Enter")
    print("🔧 Pro ukončení stiskněte Ctrl+C")
    print("="*60)
    
    # Nekonečná smyčka pro dotazování
    while True:
        try:
            # Vstup uživatele
            user_input = input("\n👤 Vy: ").strip()
            
            # Prázdný vstup
            if not user_input:
                continue
            
            # Generování odpovědi
            print("🤖 Model s adaptérem generuje odpověď...")
            response = generate_response(
                model, tokenizer, user_input,
                args.max_length, args.temperature
            )
            
            print(f"🤖 Model: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Na shledanou!")
            break
        except Exception as e:
            print(f"❌ Chyba: {e}")

if __name__ == "__main__":
    main() 