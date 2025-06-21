#!/usr/bin/env python3
"""
Testovací skript pro ověření funkčnosti přípravy dat.
Spustí základní testy bez generování nových dat.
"""

import os
import json
import sys
from typing import Dict, List, Any

def test_environment():
    """Testuje prostředí a závislosti."""
    print("=== Test prostředí ===")
    
    # Kontrola Python verze
    if sys.version_info >= (3, 8):
        print("✅ Python verze OK")
    else:
        print("❌ Python verze příliš stará")
        return False
    
    # Kontrola OpenAI API klíče
    if os.getenv('OPENAI_API_KEY'):
        print("✅ OPENAI_API_KEY nastaven")
    else:
        print("⚠️  OPENAI_API_KEY není nastaven (budou přeskočeny testy s API)")
    
    # Kontrola požadovaných souborů
    required_files = [
        "babis_templates_400.json",
        "LLM.CreateAnswers.systemPrompt.md",
        "LLM.CreateDialogue.systemPrompt.md",
        "availablemodels.json"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            missing_files.append(file)
            print(f"❌ {file} - chybí")
    
    if missing_files:
        print(f"⚠️  Chybí {len(missing_files)} souborů")
        return False
    
    return True

def test_templates():
    """Testuje načtení a validaci šablon."""
    print("\n=== Test šablon ===")
    
    try:
        with open("babis_templates_400.json", 'r', encoding='utf-8') as f:
            templates = json.load(f)
        
        if isinstance(templates, list) and len(templates) > 0:
            print(f"✅ Načteno {len(templates)} šablon")
            
            # Kontrola struktury šablon
            sample_template = templates[0]
            if isinstance(sample_template, str) and "{" in sample_template and "}" in sample_template:
                print("✅ Struktura šablon OK")
            else:
                print("❌ Neplatná struktura šablon")
                return False
            
            # Kontrola placeholders
            placeholders = set()
            for template in templates[:10]:  # Kontrola prvních 10
                import re
                found = re.findall(r'\{([^}]+)\}', template)
                placeholders.update(found)
            
            print(f"✅ Nalezeny placeholders: {', '.join(sorted(placeholders))}")
            return True
        else:
            print("❌ Šablony nejsou v očekávaném formátu")
            return False
            
    except Exception as e:
        print(f"❌ Chyba při načítání šablon: {e}")
        return False

def test_system_prompts():
    """Testuje načtení systémových promptů."""
    print("\n=== Test systémových promptů ===")
    
    prompt_files = [
        "LLM.CreateAnswers.systemPrompt.md",
        "LLM.CreateDialogue.systemPrompt.md"
    ]
    
    for file in prompt_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if len(content) > 100:  # Minimální délka
                print(f"✅ {file} načten ({len(content)} znaků)")
            else:
                print(f"❌ {file} příliš krátký")
                return False
                
        except Exception as e:
            print(f"❌ Chyba při načítání {file}: {e}")
            return False
    
    return True

def test_models_config():
    """Testuje konfiguraci modelů."""
    print("\n=== Test konfigurace modelů ===")
    
    try:
        with open("availablemodels.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if "models" in config:
            models = config["models"]
            print(f"✅ Načteno {len(models)} modelů")
            
            # Kontrola výchozího modelu
            default_model = None
            for model_id, model_config in models.items():
                if model_config.get("default", 0) == 1:
                    default_model = model_id
                    break
            
            if default_model:
                print(f"✅ Výchozí model: {default_model}")
            else:
                print("⚠️  Není nastaven výchozí model")
            
            return True
        else:
            print("❌ Neplatná struktura konfigurace modelů")
            return False
            
    except Exception as e:
        print(f"❌ Chyba při načítání konfigurace modelů: {e}")
        return False

def test_existing_data():
    """Testuje existující data."""
    print("\n=== Test existujících dat ===")
    
    # Kontrola finálního datasetu
    if os.path.exists("data/all.jsonl"):
        try:
            with open("data/all.jsonl", 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            messages = data.get("messages", [])
            if len(messages) > 0:
                print(f"✅ Finální dataset načten ({len(messages)} zpráv)")
                
                # Kontrola struktury
                if messages[0]["role"] == "system":
                    print("✅ První zpráva je systémová")
                else:
                    print("❌ První zpráva není systémová")
                    return False
                
                # Počítání QA párů
                qa_pairs = 0
                for i in range(1, len(messages), 2):
                    if i + 1 < len(messages):
                        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                            qa_pairs += 1
                
                print(f"✅ Nalezeno {qa_pairs} QA párů")
                return True
            else:
                print("❌ Dataset je prázdný")
                return False
                
        except Exception as e:
            print(f"❌ Chyba při načítání datasetu: {e}")
            return False
    else:
        print("⚠️  Finální dataset neexistuje (spusťte přípravu dat)")
        return True  # Není chyba, jen není připraven

def test_libraries():
    """Testuje import knihoven."""
    print("\n=== Test knihoven ===")
    
    try:
        # Test základních knihoven
        import openai
        print("✅ openai")
        
        import tiktoken
        print("✅ tiktoken")
        
        import numpy
        print("✅ numpy")
        
        import pandas
        print("✅ pandas")
        
        # Test vlastních knihoven
        sys.path.append('lib')
        
        from openai_cost_calculator import OpenAICostCalculator
        print("✅ OpenAICostCalculator")
        
        from babis_dataset_generator import BabisDatasetGenerator
        print("✅ BabisDatasetGenerator")
        
        from babis_dialog_generator import BabisDialogGenerator
        print("✅ BabisDialogGenerator")
        
        return True
        
    except ImportError as e:
        print(f"❌ Chyba importu: {e}")
        return False
    except Exception as e:
        print(f"❌ Neočekávaná chyba: {e}")
        return False

def run_all_tests():
    """Spustí všechny testy."""
    print("🧪 Spouštím testy přípravy dat")
    print("=" * 50)
    
    tests = [
        ("Prostředí", test_environment),
        ("Šablony", test_templates),
        ("Systémové prompty", test_system_prompts),
        ("Konfigurace modelů", test_models_config),
        ("Existující data", test_existing_data),
        ("Knihovny", test_libraries)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Neočekávaná chyba v testu '{test_name}': {e}")
            results.append((test_name, False))
    
    # Shrnutí výsledků
    print("\n" + "=" * 50)
    print("📊 SHRNUTÍ TESTŮ")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nÚspěšnost: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 Všechny testy prošly! Příprava dat je připravena.")
        return True
    elif passed >= total * 0.8:
        print("⚠️  Většina testů prošla, ale některé selhaly.")
        return True
    else:
        print("❌ Mnoho testů selhalo. Zkontrolujte instalaci a konfiguraci.")
        return False

def main():
    """Hlavní funkce pro spuštění testů."""
    try:
        success = run_all_tests()
        
        if success:
            print("\n💡 Tip: Pro spuštění kompletní přípravy dat použijte:")
            print("   python run_data_preparation.py")
        else:
            print("\n🔧 Opravte chyby a spusťte testy znovu.")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Testy přerušeny uživatelem.")
        return 1
    except Exception as e:
        print(f"\n\n❌ Neočekávaná chyba: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 