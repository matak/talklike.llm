#!/usr/bin/env python3
"""
Skript pro nahrání již vygenerovaného modelu na Hugging Face Hub
Použijte tento skript, pokud jste zapomněli přidat --push_to_hub do původního příkazu
"""

import os
import argparse
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def upload_model_to_hf(model_path, hub_model_id, token):
    """Nahraje model na Hugging Face Hub"""
    
    print(f"📤 Nahrávám model z {model_path} na HF Hub...")
    print(f"🎯 Model ID: {hub_model_id}")
    
    # Kontrola existence modelu
    if not os.path.exists(model_path):
        print(f"❌ Model neexistuje v cestě: {model_path}")
        return False
    
    # Kontrola souborů modelu
    required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️ Chybí soubory: {missing_files}")
        print("📋 Dostupné soubory:")
        for file in os.listdir(model_path):
            print(f"  - {file}")
    
    try:
        # Načtení modelu a tokenizeru
        print("🔧 Načítám model...")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print("📤 Nahrávám na HF Hub...")
        model.push_to_hub(hub_model_id, token=token)
        tokenizer.push_to_hub(hub_model_id, token=token)
        
        print(f"✅ Model úspěšně nahrán!")
        print(f"🌐 Dostupný na: https://huggingface.co/{hub_model_id}")
        return True
        
    except Exception as e:
        print(f"❌ Chyba při nahrávání: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Nahrání modelu na Hugging Face Hub')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Cesta k uloženému modelu (např. /workspace/babis-finetuned-final)')
    parser.add_argument('--hub_model_id', type=str, required=True,
                       help='Název modelu na HF Hub (např. username/babis-model)')
    parser.add_argument('--check_only', action='store_true',
                       help='Pouze zkontrolovat model bez nahrávání')
    
    args = parser.parse_args()
    
    # Načtení proměnných prostředí
    load_dotenv()
    
    # Kontrola HF tokenu
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        print("❌ HF_TOKEN nebyl nalezen v prostředí!")
        print("💡 Nastavte HF_TOKEN v .env souboru nebo prostředí")
        return
    
    # Přihlášení na HF
    try:
        login(token=HF_TOKEN)
        print("✅ Hugging Face login úspěšný")
    except Exception as e:
        print(f"❌ Chyba při přihlášení na HF: {e}")
        return
    
    # Kontrola modelu
    print(f"🔍 Kontroluji model v: {args.model_path}")
    
    if not os.path.exists(args.model_path):
        print(f"❌ Cesta neexistuje: {args.model_path}")
        print("\n💡 Možné cesty k modelu:")
        print("  - /workspace/babis-finetuned-final")
        print("  - /workspace/babis-mistral-finetuned-final")
        print("  - /workspace/[output_dir]-final")
        return
    
    # Výpis obsahu adresáře
    print(f"📁 Obsah adresáře {args.model_path}:")
    try:
        for item in os.listdir(args.model_path):
            item_path = os.path.join(args.model_path, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                print(f"  📄 {item} ({size:,} B)")
            else:
                print(f"  📁 {item}/")
    except Exception as e:
        print(f"⚠️ Nelze číst obsah adresáře: {e}")
    
    if args.check_only:
        print("\n✅ Kontrola dokončena")
        return
    
    # Nahrání modelu
    success = upload_model_to_hf(args.model_path, args.hub_model_id, HF_TOKEN)
    
    if success:
        print("\n🎉 Model byl úspěšně nahrán na Hugging Face Hub!")
        print(f"🔗 Odkaz: https://huggingface.co/{args.hub_model_id}")
        print("\n💡 Pro použití modelu:")
        print(f"   from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f"   model = AutoModelForCausalLM.from_pretrained('{args.hub_model_id}')")
        print(f"   tokenizer = AutoTokenizer.from_pretrained('{args.hub_model_id}')")
    else:
        print("\n❌ Nahrávání selhalo")

if __name__ == "__main__":
    main() 