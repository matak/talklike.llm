#!/usr/bin/env python3
"""
Skript pro nahrání již vygenerovaného modelu na Hugging Face Hub
Použijte tento skript, pokud jste zapomněli přidat --push_to_hub do původního příkazu
"""

# Import setup_environment pro správné nastavení prostředí
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import setup_environment

import argparse
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
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
    parser = argparse.ArgumentParser(description='Nahrání modelu na Hugging Face Hub')
    parser.add_argument('--model_path', type=str, 
                       help='Cesta k celému modelu (nahrává celý model)')
    parser.add_argument('--adapter_path', type=str,
                       help='Cesta k LoRA adaptéru (nahrává pouze adaptér)')
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
        return False
    
    # Přihlášení na HF
    try:
        login(token=HF_TOKEN)
        print("✅ Hugging Face login úspěšný")
    except Exception as e:
        print(f"❌ Chyba při přihlášení na HF: {e}")
        return False
    
    # Kontrola parametrů
    if args.adapter_path and args.model_path:
        print("❌ Nelze specifikovat oba parametry --model_path a --adapter_path současně!")
        print("💡 Použijte buď --model_path pro celý model nebo --adapter_path pro adaptér")
        return False
    
    if not args.adapter_path and not args.model_path:
        print("❌ Musíte specifikovat buď --model_path nebo --adapter_path!")
        print("💡 Použití:")
        print("   --model_path /cesta/k/modelu     # Nahrává celý model")
        print("   --adapter_path /cesta/k/adapteru # Nahrává pouze adaptér")
        return False
    
    # Určení cesty a typu
    if args.adapter_path:
        model_path = args.adapter_path
        is_adapter = True
        print(f"🔍 Nahrávám LoRA adaptér z: {model_path}")
    else:
        model_path = args.model_path
        is_adapter = False
        print(f"🔍 Nahrávám celý model z: {model_path}")
    
    # Kontrola existence
    if not os.path.exists(model_path):
        print(f"❌ Cesta neexistuje: {model_path}")
        print("\n💡 Možné cesty:")
        if is_adapter:
            print("  - /workspace/mistral-babis-finetuned-final")
            print("  - /workspace/babis-mistral-finetuned-final")
        else:
            print("  - /workspace/babis-finetuned-final")
            print("  - /workspace/babis-mistral-finetuned-final")
        return
    
    # Výpis obsahu adresáře
    print(f"📁 Obsah adresáře {model_path}:")
    try:
        files_in_dir = os.listdir(model_path)
        for item in files_in_dir:
            item_path = os.path.join(model_path, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                print(f"  📄 {item} ({size:,} B)")
            else:
                print(f"  📁 {item}/")
    except Exception as e:
        print(f"⚠️ Nelze číst obsah adresáře: {e}")
        return
    
    # Kontrola očekávaných souborů
    if is_adapter:
        required_files = ['adapter_config.json', 'adapter_model.safetensors']
        missing_files = [f for f in required_files if f not in files_in_dir]
        if missing_files:
            print(f"❌ Chybí soubory adaptéru: {missing_files}")
            print("💡 Očekávané soubory pro adaptér: adapter_config.json, adapter_model.safetensors")
            return
    else:
        required_files = ['config.json']
        model_files = ['pytorch_model.bin', 'model.safetensors']
        if not any(f in files_in_dir for f in model_files):
            print(f"❌ Chybí soubory modelu: {model_files}")
            print("💡 Očekávané soubory pro model: config.json + pytorch_model.bin nebo model.safetensors")
            return
    
    if args.check_only:
        print("\n✅ Kontrola dokončena")
        return
    
    # Nahrání modelu nebo adaptéru
    if is_adapter:
        success = upload_adapter_only(model_path, args.hub_model_id, HF_TOKEN)
    else:
        success = upload_model_to_hf(model_path, args.hub_model_id, HF_TOKEN)
    
    if success:
        print(f"\n🎉 {'LoRA adapter' if is_adapter else 'Model'} byl úspěšně nahrán na Hugging Face Hub!")
        print(f"🔗 Odkaz: https://huggingface.co/{args.hub_model_id}")
        if is_adapter:
            print("\n💡 Pro použití adaptéru:")
            print(f"   from transformers import AutoModelForCausalLM, AutoTokenizer")
            print(f"   from peft import PeftModel")
            print(f"   base_model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')")
            print(f"   model = PeftModel.from_pretrained(base_model, '{args.hub_model_id}')")
        else:
            print("\n💡 Pro použití modelu:")
            print(f"   from transformers import AutoModelForCausalLM, AutoTokenizer")
            print(f"   model = AutoModelForCausalLM.from_pretrained('{args.hub_model_id}')")
            print(f"   tokenizer = AutoTokenizer.from_pretrained('{args.hub_model_id}')")
    else:
        print("\n❌ Nahrávání selhalo")

if __name__ == "__main__":
    main() 