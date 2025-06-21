#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test integrace adaptéru s benchmarking systémem
Ověřuje, že model s adaptérem funguje správně
"""

import sys
import os
from pathlib import Path

# Přidání cesty k 2_finetunning pro import
sys.path.append(str(Path(__file__).parent.parent / "2_finetunning"))

def test_adapter_loading():
    """Testuje načtení adaptéru"""
    print("🔍 Testuji načtení adaptéru...")
    
    try:
        from test_adapter import load_model_with_adapter, generate_response
        print("✅ Import test_adapter úspěšný")
        
        # Test načtení modelu
        base_model = "mistralai/Mistral-7B-Instruct-v0.3"
        adapter_path = "mcmatak/babis-mistral-adapter"
        
        print(f"🤖 Načítám model: {base_model}")
        print(f"🔧 S adaptérem: {adapter_path}")
        
        model, tokenizer = load_model_with_adapter(base_model, adapter_path)
        
        if model is not None and tokenizer is not None:
            print("✅ Model s adaptérem úspěšně načten!")
            return True
        else:
            print("❌ Model se nepodařilo načíst")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Chyba při načítání: {e}")
        return False

def test_response_generation():
    """Testuje generování odpovědí"""
    print("\n🔍 Testuji generování odpovědí...")
    
    try:
        from generate_responses import generate_real_response, load_benchmark_model
        
        # Načtení modelu
        model, tokenizer = load_benchmark_model("finetuned")
        
        if model is None or tokenizer is None:
            print("❌ Model není dostupný")
            return False
        
        # Test otázka
        test_question = "Pane Babiši, jak hodnotíte současnou inflaci?"
        print(f"📝 Test otázka: {test_question}")
        
        # Generování odpovědi
        response = generate_real_response(model, tokenizer, test_question, "finetuned")
        
        print(f"🤖 Odpověď: {response}")
        
        if response and len(response) > 10:
            print("✅ Generování odpovědí funguje!")
            return True
        else:
            print("❌ Odpověď je příliš krátká nebo prázdná")
            return False
            
    except Exception as e:
        print(f"❌ Chyba při generování: {e}")
        return False

def test_benchmark_dataset():
    """Testuje benchmark dataset"""
    print("\n🔍 Testuji benchmark dataset...")
    
    try:
        from create_benchmark_dataset import create_benchmark_dataset
        
        # Vytvoření datasetu
        create_benchmark_dataset()
        
        if os.path.exists("results/benchmark_dataset.json"):
            print("✅ Benchmark dataset vytvořen!")
            return True
        else:
            print("❌ Benchmark dataset nebyl vytvořen")
            return False
            
    except Exception as e:
        print(f"❌ Chyba při vytváření datasetu: {e}")
        return False

def test_style_evaluation():
    """Testuje evaluaci stylu"""
    print("\n🔍 Testuji evaluaci stylu...")
    
    try:
        from evaluate_style import evaluate_babis_style
        
        # Test odpověď
        test_response = "Hele, inflace je jak když kráva hraje na klavír! Já makám, ale opozice krade. To je skandál! Andrej Babiš"
        
        evaluation = evaluate_babis_style(test_response)
        
        print(f"📊 Evaluace test odpovědi:")
        print(f"   Skóre: {evaluation['total_score']}/10")
        print(f"   Známka: {evaluation['grade']}")
        print(f"   Délka: {evaluation['length']} znaků")
        
        if evaluation['total_score'] > 5:
            print("✅ Evaluace stylu funguje!")
            return True
        else:
            print("❌ Evaluace vrátila nízké skóre")
            return False
            
    except Exception as e:
        print(f"❌ Chyba při evaluaci: {e}")
        return False

def main():
    """Hlavní test funkce"""
    print("🧪 Spouštím testy integrace adaptéru...")
    print("=" * 60)
    
    tests = [
        ("Načtení adaptéru", test_adapter_loading),
        ("Generování odpovědí", test_response_generation),
        ("Benchmark dataset", test_benchmark_dataset),
        ("Evaluace stylu", test_style_evaluation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Test: {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                print(f"✅ {test_name}: ÚSPĚCH")
                passed += 1
            else:
                print(f"❌ {test_name}: SELHÁNÍ")
        except Exception as e:
            print(f"❌ {test_name}: CHYBA - {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Výsledky testů: {passed}/{total} úspěšných")
    
    if passed == total:
        print("🎉 Všechny testy prošly! Benchmarking je připraven.")
        print("🚀 Můžete spustit: ./run_benchmark_with_adapter.sh")
    else:
        print("⚠️  Některé testy selhaly. Zkontrolujte konfiguraci.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 