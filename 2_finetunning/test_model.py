#!/usr/bin/env python3
"""
Jednoduchý skript pro testování finetunovaného modelu z Hugging Face
Umožňuje uživateli zadávat prompty a získávat odpovědi od modelu
"""

import argparse
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

# Potlačení varování
warnings.filterwarnings("ignore")

def load_model(model_path, device="auto"):
    """Načte model a tokenizer z Hugging Face"""
    try:
        print(f"🤖 Načítám model: {model_path}")
        print("⏳ Prosím počkejte, načítání může trvat několik minut...")
        
        # Načtení tokenizeru
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Kontrola a nastavení pad tokenu
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
        
        # Načtení modelu
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        
        # Synchronizace pad tokenu s modelem
        if hasattr(model.config, 'pad_token_id'):
            model.config.pad_token_id = tokenizer.pad_token_id
        
        print("✅ Model úspěšně načten!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Chyba při načítání modelu: {e}")
        return None, None

def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7):
    """Generuje odpověď na základě promptu"""
    try:
        # Tokenizace vstupu
        inputs = tokenizer(
            prompt,
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
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
        
    except Exception as e:
        return f"❌ Chyba při generování: {e}"

def main():
    parser = argparse.ArgumentParser(
        description="Testování finetunovaného modelu z Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Příklady použití:
  python test_model.py microsoft/DialoGPT-medium
  python test_model.py username/my-finetuned-model
  python test_model.py --device cpu username/my-model
        """
    )
    
    parser.add_argument(
        "model_path",
        help="Cesta k modelu na Hugging Face (repo/model_name)"
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
    
    args = parser.parse_args()
    
    # Načtení modelu
    model, tokenizer = load_model(args.model_path, args.device)
    
    if model is None or tokenizer is None:
        sys.exit(1)
    
    print("\n" + "="*60)
    print("💬 INTERAKTIVNÍ TESTOVÁNÍ MODELU")
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
            print("🤖 Model generuje odpověď...")
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