#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testovací skript pro benchmarking TalkLike.LLM
Ověřuje funkčnost všech komponent
"""

import os
import json
import sys
from datetime import datetime

def test_imports():
    """Testuje import všech modulů"""
    print("🔍 Testuji importy...")
    
    try:
        from evaluate_style import evaluate_babis_style, BabisStyleEvaluator
        print("✅ evaluate_style.py - OK")
    except ImportError as e:
        print(f"❌ evaluate_style.py - Chyba: {e}")
        return False
    
    try:
        from compare_models import compare_models, calculate_comparison_metrics
        print("✅ compare_models.py - OK")
    except ImportError as e:
        print(f"❌ compare_models.py - Chyba: {e}")
        return False
    
    try:
        from generate_responses import generate_responses, generate_mock_response
        print("✅ generate_responses.py - OK")
    except ImportError as e:
        print(f"❌ generate_responses.py - Chyba: {e}")
        return False
    
    try:
        from create_benchmark_dataset import create_benchmark_dataset, validate_benchmark_dataset
        print("✅ create_benchmark_dataset.py - OK")
    except ImportError as e:
        print(f"❌ create_benchmark_dataset.py - Chyba: {e}")
        return False
    
    return True

def test_data_files():
    """Testuje přítomnost datových souborů"""
    print("\n📁 Testuji datové soubory...")
    
    required_files = [
        "benchmark_questions.json",
        "LLM.Benchmark.systemPrompt.md"
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} - OK")
        else:
            print(f"❌ {file} - Chybí")
            all_exist = False
    
    return all_exist

def test_evaluation():
    """Testuje evaluaci stylu"""
    print("\n🎯 Testuji evaluaci stylu...")
    
    try:
        from evaluate_style import evaluate_babis_style
        
        # Test špatné odpovědi (před fine-tuningem)
        bad_response = "Inflace je vážný problém, který postihuje všechny občany."
        bad_result = evaluate_babis_style(bad_response)
        
        print(f"   Špatná odpověď: {bad_result['total_score']}/10 ({bad_result['grade']})")
        
        # Test dobré odpovědi (po fine-tuningem)
        good_response = "Hele, inflace je jak když kráva hraje na klavír! Já makám, ale opozice krade. To je skandál! Andrej Babiš"
        good_result = evaluate_babis_style(good_response)
        
        print(f"   Dobrá odpověď: {good_result['total_score']}/10 ({good_result['grade']})")
        
        # Kontrola zlepšení
        improvement = good_result['total_score'] - bad_result['total_score']
        print(f"   Zlepšení: {improvement:.1f} bodů")
        
        if improvement > 0:
            print("✅ Evaluace funguje správně")
            return True
        else:
            print("❌ Evaluace nefunguje správně")
            return False
            
    except Exception as e:
        print(f"❌ Chyba při evaluaci: {e}")
        return False

def test_dataset_creation():
    """Testuje vytvoření datasetu"""
    print("\n📋 Testuji vytvoření datasetu...")
    
    try:
        from create_benchmark_dataset import create_benchmark_dataset, validate_benchmark_dataset
        
        # Vytvoření datasetu
        dataset = create_benchmark_dataset()
        
        if dataset is None:
            print("❌ Nepodařilo se vytvořit dataset")
            return False
        
        # Validace datasetu
        is_valid, message = validate_benchmark_dataset(dataset)
        
        if is_valid:
            print(f"✅ Dataset je validní: {message}")
            print(f"   Počet otázek: {len(dataset['questions'])}")
            return True
        else:
            print(f"❌ Dataset není validní: {message}")
            return False
            
    except Exception as e:
        print(f"❌ Chyba při vytváření datasetu: {e}")
        return False

def test_response_generation():
    """Testuje generování odpovědí"""
    print("\n🤖 Testuji generování odpovědí...")
    
    try:
        from generate_responses import generate_mock_response
        
        test_question = "Pane Babiši, jak hodnotíte současnou inflaci?"
        
        # Test před fine-tuningem
        base_response = generate_mock_response(test_question, "base")
        print(f"   Před fine-tuningem: {base_response}")
        
        # Test po fine-tuningem
        finetuned_response = generate_mock_response(test_question, "finetuned")
        print(f"   Po fine-tuningem: {finetuned_response}")
        
        # Kontrola rozdílu
        if len(finetuned_response) > len(base_response):
            print("✅ Generování odpovědí funguje")
            return True
        else:
            print("❌ Generování odpovědí nefunguje správně")
            return False
            
    except Exception as e:
        print(f"❌ Chyba při generování odpovědí: {e}")
        return False

def test_comparison():
    """Testuje srovnání modelů"""
    print("\n📊 Testuji srovnání modelů...")
    
    try:
        from compare_models import calculate_comparison_metrics
        
        # Testovací data
        before_data = [
            {"response": "Inflace je vážný problém."},
            {"response": "Opozice má právo na kritiku."}
        ]
        
        after_data = [
            {"response": "Hele, inflace je jak když kráva hraje na klavír! Andrej Babiš"},
            {"response": "Opozice krade, to je skandál! Andrej Babiš"}
        ]
        
        # Výpočet metrik
        metrics = calculate_comparison_metrics(before_data, after_data)
        
        print(f"   Průměrná délka před: {metrics['avg_length_before']:.1f}")
        print(f"   Průměrná délka po: {metrics['avg_length_after']:.1f}")
        print(f"   Zlepšení: {metrics['overall_improvement_score']:.1f}")
        
        if metrics['overall_improvement_score'] > 0:
            print("✅ Srovnání funguje správně")
            return True
        else:
            print("❌ Srovnání nefunguje správně")
            return False
            
    except Exception as e:
        print(f"❌ Chyba při srovnání: {e}")
        return False

def test_directory_structure():
    """Testuje strukturu adresářů"""
    print("\n📂 Testuji strukturu adresářů...")
    
    # Vytvoření testovacích adresářů
    test_dirs = [
        "results/before_finetune",
        "results/after_finetune",
        "results/comparison",
        "results/reports",
        "results/visualizations"
    ]
    
    for directory in test_dirs:
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(directory):
            print(f"✅ {directory} - OK")
        else:
            print(f"❌ {directory} - Chyba při vytváření")
            return False
    
    return True

def main():
    """Hlavní testovací funkce"""
    print("🧪 Spouštím testy pro benchmarking...")
    print("=" * 50)
    
    tests = [
        ("Importy", test_imports),
        ("Datové soubory", test_data_files),
        ("Struktura adresářů", test_directory_structure),
        ("Evaluace stylu", test_evaluation),
        ("Vytvoření datasetu", test_dataset_creation),
        ("Generování odpovědí", test_response_generation),
        ("Srovnání modelů", test_comparison)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Neočekávaná chyba v {test_name}: {e}")
            results.append((test_name, False))
    
    # Shrnutí výsledků
    print("\n" + "=" * 50)
    print("📋 Shrnutí testů:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Výsledek: {passed}/{total} testů úspěšných")
    
    if passed == total:
        print("🎉 Všechny testy prošly! Benchmarking je připraven.")
        return True
    else:
        print("⚠️  Některé testy selhaly. Zkontrolujte chyby výše.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 