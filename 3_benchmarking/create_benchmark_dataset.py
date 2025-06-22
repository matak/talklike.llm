#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vytvoření benchmark datasetu pro TalkLike.LLM
Generuje standardizované testovací otázky
"""

import json
import os
from datetime import datetime

def create_benchmark_dataset():
    """Vytvoří benchmark dataset z předem definovaných otázek"""
    
    print("📋 Vytvářím benchmark dataset...")
    
    # Načtení otázek z JSON souboru
    with open("benchmark_questions.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = data.get("benchmark_questions", [])
    metadata = data.get("metadata", {})
    
    print(f"✅ Načteno {len(questions)} testovacích otázek")
    print(f"📊 Kategorie: {', '.join(metadata.get('categories', []))}")
    print(f"🎯 Distribuce obtížnosti: {metadata.get('difficulty_distribution', {})}")
    
    # Vytvoření strukturovaného datasetu
    benchmark_dataset = {
        "questions": questions,
        "metadata": {
            **metadata,
            "created_at": datetime.now().isoformat(),
            "total_questions": len(questions)
        }
    }
    
    # Uložení datasetu
    with open("results/benchmark_dataset.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Dataset uložen: results/benchmark_dataset.json")
    
    # Výpis statistik
    print("\n📊 Statistiky datasetu:")
    categories = {}
    difficulties = {}
    
    for q in questions:
        # Počítání kategorií
        cat = q.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        
        # Počítání obtížností
        diff = q.get("difficulty", "unknown")
        difficulties[diff] = difficulties.get(diff, 0) + 1
    
    print(f"   Kategorie:")
    for cat, count in categories.items():
        print(f"     - {cat}: {count} otázek")
    
    print(f"   Obtížnost:")
    for diff, count in difficulties.items():
        print(f"     - {diff}: {count} otázek")
    
    return benchmark_dataset

def validate_benchmark_dataset(dataset):
    """Validuje benchmark dataset"""
    
    if not dataset or "questions" not in dataset:
        return False, "Neplatná struktura datasetu"
    
    questions = dataset["questions"]
    
    # Kontrola povinných polí
    required_fields = ["id", "category", "question", "expected_style_elements", "difficulty"]
    
    for i, q in enumerate(questions):
        for field in required_fields:
            if field not in q:
                return False, f"Otázka {i+1} chybí pole: {field}"
    
    # Kontrola unikátnosti ID
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        return False, "Duplicitní ID otázek"
    
    return True, "Dataset je validní"

if __name__ == "__main__":
    # Test vytvoření datasetu
    dataset = create_benchmark_dataset()
    
    if dataset:
        is_valid, message = validate_benchmark_dataset(dataset)
        print(f"\n🔍 Validace: {message}")
        
        if is_valid:
            print("✅ Dataset je připraven pro benchmarking")
        else:
            print("❌ Dataset obsahuje chyby") 