#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hlavní benchmarking script pro TalkLike.LLM
Srovnává výkon modelu před a po fine-tuningu
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Detekce, zda je skript spouštěn z rootu nebo z adresáře 3_benchmarking
current_file = Path(__file__)
if current_file.parent.name == "3_benchmarking":
    # Spouštěno z rootu: python 3_benchmarking/run_benchmark.py
    project_root = current_file.parent.parent
    benchmark_dir = current_file.parent
    # Zůstáváme v root adresáři
else:
    # Spouštěno z adresáře 3_benchmarking
    project_root = current_file.parent.parent
    benchmark_dir = current_file.parent

# Přidání rootu projektu do path pro import modulů
sys.path.insert(0, str(project_root))

# Import setup_environment z rootu
from setup_environment import setup_environment
setup_environment()

# Přidání benchmarking adresáře do path pro import modulů
sys.path.insert(0, str(benchmark_dir))

from evaluate_style import evaluate_babis_style, evaluate_all_responses
from compare_models import compare_models
from generate_responses import generate_responses
from create_benchmark_dataset import create_benchmark_dataset
from generate_report import generate_final_report

def setup_directories():
    """Vytvoří potřebné adresáře pro výsledky"""
    directories = [
        "results/before_finetune",
        "results/after_finetune", 
        "results/comparison",
        "results/reports",
        "results/visualizations"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Vytvořen adresář: {directory}")

def main():
    """Hlavní benchmarking pipeline"""
    print("🚀 Spouštím benchmarking pipeline pro TalkLike.LLM")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # 1. Nastavení adresářů
    print("\n📁 Nastavuji adresáře...")
    setup_directories()
    
    # 2. Vytvoření testovacích dat
    print("\n📋 Vytvářím benchmark dataset...")
    create_benchmark_dataset()
    
    # 3. Generování odpovědí před fine-tuningem
    print("\n🤖 Generuji odpovědi před fine-tuningem...")
    generate_responses("base", "results/before_finetune/")
    
    # 4. Generování odpovědí po fine-tuningem
    print("\n🤖 Generuji odpovědi po fine-tuningem...")
    generate_responses("finetuned", "results/after_finetune/")
    
    # 5. Srovnání modelů
    print("\n📊 Srovnávám modely...")
    comparison_results = compare_models()
    
    # 6. Evaluace stylu
    print("\n🎯 Evaluuji styl...")
    style_results = evaluate_all_responses()
    
    # 7. Kombinace všech výsledků pro report
    print("\n📋 Generuji finální report...")
    all_results = {
        "model_comparison": comparison_results,
        "style_evaluation": style_results
    }
    generate_final_report(all_results)
    
    # 8. Výpis výsledků
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("✅ Benchmarking úspěšně dokončen!")
    print(f"⏱️  Celkový čas: {duration}")
    print(f"📁 Výsledky v: results/")
    print(f"📊 Reporty v: results/reports/")
    print(f"📈 Vizualizace v: results/visualizations/")
    print("=" * 60)

if __name__ == "__main__":
    main() 