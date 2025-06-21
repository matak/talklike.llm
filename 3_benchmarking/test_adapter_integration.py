#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test integrace adaptéru pro TalkLike.LLM
Kompletní test všech komponent benchmarkingu s reálným adaptérem
"""

# Import a nastavení prostředí
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import setup_environment

import json
import os
import time
from datetime import datetime

def test_environment_setup():
    """Test nastavení prostředí"""
    print("🔧 Test nastavení prostředí...")
    
    # Kontrola cache proměnných
    cache_vars = ['HF_HOME', 'TRANSFORMERS_CACHE', 'HF_DATASETS_CACHE']
    for var in cache_vars:
        value = os.environ.get(var)
        if value and '/workspace' in value:
            print(f"   ✅ {var}: {value}")
        else:
            print(f"   ❌ {var}: {value}")
            return False
    
    # Kontrola cache adresářů
    cache_dirs = [
        '/workspace/.cache/huggingface',
        '/workspace/.cache/huggingface/transformers',
        '/workspace/.cache/huggingface/datasets'
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            print(f"   ✅ Cache adresář existuje: {cache_dir}")
        else:
            print(f"   ❌ Cache adresář chybí: {cache_dir}")
            return False
    
    print("✅ Nastavení prostředí OK")
    return True

def test_model_loading():
    """Test načítání modelu"""
    print("\n🤖 Test načítání modelu...")
    
    try:
        from test_adapter import load_model_with_adapter
        
        base_model = "mistralai/Mistral-7B-Instruct-v0.3"
        adapter_path = "mcmatak/babis-mistral-adapter"
        
        print(f"   Načítám: {base_model}")
        print(f"   Adapter: {adapter_path}")
        
        start_time = time.time()
        model, tokenizer = load_model_with_adapter(base_model, adapter_path)
        load_time = time.time() - start_time
        
        if model is None or tokenizer is None:
            print("   ❌ Model se nepodařilo načíst")
            return False
        
        print(f"   ✅ Model načten za {load_time:.2f}s")
        print(f"   Model typ: {type(model).__name__}")
        print(f"   Tokenizer typ: {type(tokenizer).__name__}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Chyba při načítání: {e}")
        return False

def test_response_generation():
    """Test generování odpovědí"""
    print("\n💬 Test generování odpovědí...")
    
    try:
        from test_adapter import load_model_with_adapter, generate_response
        
        # Načtení modelu
        base_model = "mistralai/Mistral-7B-Instruct-v0.3"
        adapter_path = "mcmatak/babis-mistral-adapter"
        
        model, tokenizer = load_model_with_adapter(base_model, adapter_path)
        
        if model is None or tokenizer is None:
            print("   ❌ Model není dostupný")
            return False
        
        # Test otázky
        test_questions = [
            "Pane Babiši, jak hodnotíte současnou inflaci?",
            "Co si myslíte o opozici?",
            "Jak se vám daří s rodinou?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"   Test {i}: {question}")
            
            # Jednoduchý prompt bez dlouhých instrukcí
            prompt = f"<s>[INST] {question} [/INST]"
            
            # Generování
            start_time = time.time()
            response = generate_response(
                model, tokenizer, prompt,
                max_length=300, temperature=0.8
            )
            gen_time = time.time() - start_time
            
            # Vyčištění
            response = response.strip()
            if response.startswith("Otázka:"):
                response = response[response.find("[/INST]") + 7:].strip()
            
            print(f"      Odpověď: {response}")
            print(f"      Čas: {gen_time:.2f}s")
            
            # Rychlá analýza
            babis_indicators = ["hele", "skandál", "makám", "opozice", "brusel", "moje rodina"]
            found = sum(1 for indicator in babis_indicators if indicator.lower() in response.lower())
            
            if "andrej babiš" in response.lower():
                print(f"      ✅ Podpis: ANO")
            else:
                print(f"      ❌ Podpis: NE")
            
            print(f"      📊 Babišovy indikátory: {found}/{len(babis_indicators)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Chyba při generování: {e}")
        return False

def test_benchmark_components():
    """Test komponent benchmarkingu"""
    print("\n📊 Test komponent benchmarkingu...")
    
    # Test načítání otázek
    try:
        if os.path.exists("benchmark_questions.json"):
            with open("benchmark_questions.json", "r", encoding="utf-8") as f:
                questions = json.load(f)
            print(f"   ✅ Načteno {len(questions)} testovacích otázek")
        else:
            print("   ⚠️  Soubor benchmark_questions.json neexistuje")
            questions = []
    except Exception as e:
        print(f"   ❌ Chyba při načítání otázek: {e}")
        questions = []
    
    # Test evaluace stylu
    try:
        from evaluate_style import StyleEvaluator
        
        evaluator = StyleEvaluator()
        print("   ✅ StyleEvaluator načten")
        
        # Test evaluace
        test_response = "Hele, to je skandál! Já makám, ale opozice krade. Andrej Babiš"
        score = evaluator.evaluate_response(test_response)
        print(f"   ✅ Test evaluace: {score:.2f}/10")
        
    except Exception as e:
        print(f"   ❌ Chyba při evaluaci stylu: {e}")
    
    # Test generování reportů
    try:
        from generate_report import generate_excel_report
        
        print("   ✅ Modul pro reporty načten")
        
    except Exception as e:
        print(f"   ❌ Chyba při načítání reportů: {e}")
    
    return True

def test_file_structure():
    """Test struktury souborů"""
    print("\n📁 Test struktury souborů...")
    
    required_files = [
        "benchmark_questions.json",
        "evaluate_style.py",
        "generate_responses.py",
        "compare_models.py",
        "generate_report.py",
        "run_benchmark.py"
    ]
    
    required_dirs = [
        "results",
        "results/before_finetune",
        "results/after_finetune",
        "results/comparison",
        "results/reports",
        "results/visualizations"
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/")
    
    return True

def run_complete_test():
    """Spustí kompletní test integrace"""
    print("🧪 KOMPLETNÍ TEST INTEGRACE ADAPTÉRU")
    print("=" * 60)
    
    tests = [
        ("Nastavení prostředí", test_environment_setup),
        ("Načítání modelu", test_model_loading),
        ("Generování odpovědí", test_response_generation),
        ("Komponenty benchmarkingu", test_benchmark_components),
        ("Struktura souborů", test_file_structure)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Neočekávaná chyba v {test_name}: {e}")
            results.append((test_name, False))
    
    # Shrnutí
    print("\n" + "=" * 60)
    print("📋 SHRNUTÍ TESTU")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nCelkem: {passed}/{total} testů úspěšných")
    
    if passed == total:
        print("🎉 VŠECHNY TESTY ÚSPĚŠNÉ!")
        print("🚀 Adaptér je připraven pro benchmarking!")
        print("💡 Spusťte: ./run_benchmark_with_adapter.sh")
    else:
        print("⚠️  NĚKTERÉ TESTY SELHALY")
        print("🔧 Zkontrolujte chyby výše")
    
    return passed == total

if __name__ == "__main__":
    success = run_complete_test()
    if not success:
        exit(1) 