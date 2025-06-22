#!/usr/bin/env python3
"""
Skript pro sloučení LoRA adaptéru s base modelem do kompletního fine-tuned modelu
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
import torch

def merge_adapter_to_full_model(adapter_path, base_model_name, output_path, hub_model_id=None, token=None):
    """Sloučí LoRA adaptér s base modelem do kompletního modelu"""
    
    print(f"🔧 Slučuji LoRA adaptér s base modelem...")
    print(f"📁 Adapter: {adapter_path}")
    print(f"📁 Base model: {base_model_name}")
    print(f"📁 Výstup: {output_path}")
    
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
        # Načtení base modelu
        print("🔧 Načítám base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Načtení tokenizeru
        print("🔤 Načítám tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Načtení a sloučení adaptéru
        print("🔗 Slučuji LoRA adaptér...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Sloučení adaptéru s base modelem
        print("🔄 Provádím merge_and_unload...")
        merged_model = model.merge_and_unload()
        
        # Nahrání na HF Hub (pokud je specifikováno)
        if hub_model_id and token:
            print("📤 Nahrávám kompletní model přímo na HF Hub...")
            print("💡 Používam workspace s vyčištěním cache")
            
            # Vyčištění cache před začátkem
            from lib.disk_manager import DiskManager
            dm = DiskManager()
            print("🧹 Čistím cache pro uvolnění místa...")
            dm.cleanup_cache()
            
            # Kontrola místa
            if not dm.check_disk_space('/workspace', threshold=85):
                print("⚠️ Stále málo místa, agresivní vyčištění...")
                dm.aggressive_cleanup()
                
                if not dm.check_disk_space('/workspace', threshold=85):
                    print("❌ Nedost místa i po vyčištění")
                    return False
            
            # Použijeme workspace pro dočasné uložení
            temp_dir = "/workspace/temp_complete_model"
            print(f"📁 Dočasné umístění: {temp_dir}")
            
            try:
                # Uložení do dočasného adresáře s sharding
                print("💾 Ukládám do dočasného adresáře...")
                merged_model.save_pretrained(
                    temp_dir,
                    max_shard_size="1GB",  # Menší shardy
                    safe_serialization=True
                )
                tokenizer.save_pretrained(temp_dir)
                
                # Nahrání na HF Hub
                print("📤 Nahrávám na HF Hub...")
                merged_model.push_to_hub(
                    hub_model_id, 
                    token=token,
                    max_shard_size="1GB",
                    safe_serialization=True
                )
                tokenizer.push_to_hub(hub_model_id, token=token)
                
                print(f"✅ Kompletní model nahrán: https://huggingface.co/{hub_model_id}")
                
            finally:
                # Vyčištění dočasného adresáře
                if os.path.exists(temp_dir):
                    print("🗑️ Mažu dočasný adresář...")
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
        else:
            # Kontrola místa před uložením (pouze pokud není HF Hub)
            print("💾 Kontroluji dostupné místo...")
            from lib.disk_manager import DiskManager
            dm = DiskManager()
            
            # Zajistíme, že output_path je na network storage
            if not output_path.startswith('/workspace'):
                output_path = f'/workspace/{output_path.lstrip("./")}'
            
            # Kontrola místa na network storage
            if not dm.check_disk_space('/workspace', threshold=90):
                print("⚠️ Málo místa na network storage, zkouším vyčištění...")
                dm.cleanup_cache()
                
                if not dm.check_disk_space('/workspace', threshold=90):
                    print("❌ Nedost místa pro uložení kompletního modelu")
                    print("💡 Kompletní Mistral-7B model potřebuje ~14GB místa")
                    return False
            
            # Uložení kompletního modelu
            print(f"💾 Ukládám kompletní model do: {output_path}")
            os.makedirs(output_path, exist_ok=True)
            
            merged_model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
        
        print(f"✅ Kompletní model úspěšně vytvořen!")
        if hub_model_id:
            print(f"🌐 Dostupný na: https://huggingface.co/{hub_model_id}")
        else:
            print(f"📁 Uložen v: {output_path}")
        
        # Výpis velikosti modelu
        model_size = sum(p.numel() for p in merged_model.parameters())
        print(f"📊 Velikost modelu: {model_size:,} parametrů")
        
        return True
        
    except Exception as e:
        print(f"❌ Chyba při slučování: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Sloučení LoRA adaptéru s base modelem')
    parser.add_argument('--adapter_path', type=str, required=True,
                       help='Cesta k LoRA adaptéru')
    parser.add_argument('--base_model', type=str, default='mistralai/Mistral-7B-Instruct-v0.3',
                       help='Název base modelu')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Cesta pro uložení kompletního modelu')
    parser.add_argument('--hub_model_id', type=str,
                       help='Název modelu na HF Hub (volitelné)')
    parser.add_argument('--check_only', action='store_true',
                       help='Pouze zkontrolovat adapter bez slučování')
    
    args = parser.parse_args()
    
    # Načtení proměnných prostředí
    load_dotenv()
    
    # Kontrola HF tokenu (pokud je potřeba)
    HF_TOKEN = None
    if args.hub_model_id:
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
    
    # Kontrola existence adaptéru
    if not os.path.exists(args.adapter_path):
        print(f"❌ Adapter neexistuje: {args.adapter_path}")
        return False
    
    # Výpis obsahu adresáře adaptéru
    print(f"📁 Obsah adresáře {args.adapter_path}:")
    try:
        files_in_dir = os.listdir(args.adapter_path)
        for item in files_in_dir:
            item_path = os.path.join(args.adapter_path, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                print(f"  📄 {item} ({size:,} B)")
            else:
                print(f"  📁 {item}/")
    except Exception as e:
        print(f"⚠️ Nelze číst obsah adresáře: {e}")
        return False
    
    # Kontrola očekávaných souborů
    required_files = ['adapter_config.json', 'adapter_model.safetensors']
    missing_files = [f for f in required_files if f not in files_in_dir]
    if missing_files:
        print(f"❌ Chybí soubory adaptéru: {missing_files}")
        return False
    
    if args.check_only:
        print("\n✅ Kontrola dokončena")
        return
    
    # Sloučení adaptéru s base modelem
    success = merge_adapter_to_full_model(
        args.adapter_path,
        args.base_model,
        args.output_path,
        args.hub_model_id,
        HF_TOKEN
    )
    
    if success:
        print(f"\n🎉 Kompletní model byl úspěšně vytvořen!")
        print(f"📁 Uložen v: {args.output_path}")
        if args.hub_model_id:
            print(f"🌐 Dostupný na: https://huggingface.co/{args.hub_model_id}")
        
        print("\n💡 Pro použití kompletního modelu:")
        print(f"   from transformers import AutoModelForCausalLM, AutoTokenizer")
        if args.hub_model_id:
            print(f"   model = AutoModelForCausalLM.from_pretrained('{args.hub_model_id}')")
            print(f"   tokenizer = AutoTokenizer.from_pretrained('{args.hub_model_id}')")
        else:
            print(f"   model = AutoModelForCausalLM.from_pretrained('{args.output_path}')")
            print(f"   tokenizer = AutoTokenizer.from_pretrained('{args.output_path}')")
    else:
        print("\n❌ Slučování selhalo")

if __name__ == "__main__":
    main() 