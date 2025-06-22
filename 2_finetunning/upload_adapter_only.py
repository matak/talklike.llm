#!/usr/bin/env python3
"""
Skript pro nahrání pouze LoRA adaptéru na Hugging Face Hub
Tento přístup je mnohem efektivnější než nahrávání celého modelu
"""

# Import setup_environment pro správné nastavení prostředí
import setup_environment

import os
import argparse
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def upload_adapter_only(adapter_path, hub_model_id, token):
    """Nahraje pouze LoRA adapter na Hugging Face Hub"""
    
    print(f"📤 Nahrávám LoRA adapter z {adapter_path} na HF Hub...")
    print(f"🎯 Model ID: {hub_model_id}")
    
    # Kontrola existence adaptéru
    if not os.path.exists(adapter_path):
        print(f"❌ Adapter neexistuje v cestě: {adapter_path}")
        return False
    
    # Kontrola souborů adaptéru
    required_files = ['adapter_config.json', 'adapter_model.safetensors']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(adapter_path, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Chybí soubory adaptéru: {missing_files}")
        return False
    
    try:
        # Inicializace HF API
        api = HfApi(token=token)
        
        # Vytvoření repository
        print("🔧 Vytvářím repository na HF Hub...")
        api.create_repo(
            repo_id=hub_model_id,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        
        # Nahrání souborů adaptéru
        print("📤 Nahrávám soubory adaptéru...")
        
        # Seznam souborů k nahrání
        files_to_upload = [
            'adapter_config.json',
            'adapter_model.safetensors',
            'README.md',
            'training_args.bin'
        ]
        
        # Nahrání každého souboru
        for filename in files_to_upload:
            file_path = os.path.join(adapter_path, filename)
            if os.path.exists(file_path):
                print(f"  📄 Nahrávám {filename}...")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=filename,
                    repo_id=hub_model_id,
                    repo_type="model"
                )
            else:
                print(f"  ⚠️ Soubor {filename} neexistuje, přeskočeno")
        
        # Nahrání tokenizer souborů (pokud existují)
        tokenizer_files = [
            'tokenizer.json',
            'tokenizer.model', 
            'tokenizer_config.json',
            'special_tokens_map.json',
            'chat_template.jinja'
        ]
        
        for filename in tokenizer_files:
            file_path = os.path.join(adapter_path, filename)
            if os.path.exists(file_path):
                print(f"  🔤 Nahrávám {filename}...")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=filename,
                    repo_id=hub_model_id,
                    repo_type="model"
                )
        
        print(f"✅ LoRA adapter úspěšně nahrán!")
        print(f"🌐 Dostupný na: https://huggingface.co/{hub_model_id}")
        return True
        
    except Exception as e:
        print(f"❌ Chyba při nahrávání: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Nahrání pouze LoRA adaptéru na Hugging Face Hub')
    parser.add_argument('--adapter_path', type=str, required=True, 
                       help='Cesta k LoRA adaptéru (např. /workspace/babis-mistral-finetuned-final)')
    parser.add_argument('--hub_model_id', type=str, required=True,
                       help='Název modelu na HF Hub (např. username/babis-adapter)')
    parser.add_argument('--check_only', action='store_true',
                       help='Pouze zkontrolovat adapter bez nahrávání')
    
    args = parser.parse_args()
    
    # Načtení proměnných prostředí
    load_dotenv()
    
    # Kontrola HF tokenu
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        print("❌ HF_TOKEN nebyl nalezen v prostředí!")
        print("💡 Nastavte HF_TOKEN v .env souboru nebo prostředí")
        return False
    
    # Přihlášení na HF
    try:
        login(token=HF_TOKEN)
        print("✅ Hugging Face login úspěšný")
    except Exception as e:
        print(f"❌ Chyba při přihlášení na HF: {e}")
        return False
    
    # Kontrola adaptéru
    print(f"🔍 Kontroluji adapter v: {args.adapter_path}")
    
    if not os.path.exists(args.adapter_path):
        print(f"❌ Cesta neexistuje: {args.adapter_path}")
        return False
    
    # Výpis obsahu adresáře
    print(f"📁 Obsah adresáře {args.adapter_path}:")
    try:
        for item in os.listdir(args.adapter_path):
            item_path = os.path.join(args.adapter_path, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                print(f"  📄 {item} ({size:,} B)")
            else:
                print(f"  📁 {item}/")
    except Exception as e:
        print(f"⚠️ Nelze číst obsah adresáře: {e}")
    
    if args.check_only:
        print("\n✅ Kontrola dokončena")
        return True
    
    # Nahrání adaptéru
    success = upload_adapter_only(args.adapter_path, args.hub_model_id, HF_TOKEN)
    
    if success:
        print("\n🎉 LoRA adapter byl úspěšně nahrán na Hugging Face Hub!")
        print(f"🔗 Odkaz: https://huggingface.co/{args.hub_model_id}")
        print("\n💡 Pro použití adaptéru:")
        print(f"   from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f"   from peft import PeftModel")
        print(f"   base_model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')")
        print(f"   model = PeftModel.from_pretrained(base_model, '{args.hub_model_id}')")
    else:
        print("\n❌ Nahrávání selhalo")

if __name__ == "__main__":
    main() 