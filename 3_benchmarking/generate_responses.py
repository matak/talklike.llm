#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generování odpovědí pro benchmarking TalkLike.LLM
Simuluje odpovědi před a po fine-tuningu
"""

import json
import os
import random
from datetime import datetime
from typing import List, Dict

def generate_mock_response(question: str, model_type: str) -> str:
    """Generuje mock odpověď pro testovací účely"""
    
    # Základní Babišovy fráze
    babis_phrases = [
        "Hele,", "To je skandál!", "Já makám", "Opozice krade", 
        "V Bruselu", "Moje rodina", "Já jsem to nečetl"
    ]
    
    # Slovenské odchylky
    slovak_phrases = [
        "sme", "som", "makáme", "centralizácia", "efektivizácia"
    ]
    
    # Přirovnání
    comparisons = [
        "jak když kráva hraje na klavír",
        "jak když dítě řídí tank",
        "jak když slepice hraje šachy",
        "jak když ryba jezdí na kole"
    ]
    
    # Emotivní výrazy
    emotional_phrases = [
        "to je šílený!", "tragédyje!", "kampááň!", "hrozné!"
    ]
    
    if model_type == "base":
        # Před fine-tuningem - méně Babišův styl
        responses = [
            f"Inflace je vážný problém, který postihuje všechny občany.",
            f"Opozice má právo na kritiku, ale měla by být konstruktivní.",
            f"Rodina je důležitá hodnota pro každého člověka.",
            f"Podnikání vyžaduje zodpovědný přístup a dodržování pravidel.",
            f"Evropské instituce mají své místo v moderní společnosti."
        ]
        return random.choice(responses)
    
    else:  # finetuned
        # Po fine-tuningem - autentický Babišův styl
        base_response = random.choice(babis_phrases)
        
        # Přidání slovenské odchylky (15% pravděpodobnost)
        if random.random() < 0.15:
            base_response += f" {random.choice(slovak_phrases)}"
        
        # Přidání přirovnání (30% pravděpodobnost)
        if random.random() < 0.3:
            base_response += f" {random.choice(comparisons)}"
        
        # Přidání emotivního výrazu (40% pravděpodobnost)
        if random.random() < 0.4:
            base_response += f" {random.choice(emotional_phrases)}"
        
        # Zakončení podpisem
        base_response += " Andrej Babiš"
        
        return base_response

def generate_responses(model_type: str, output_dir: str):
    """Generuje odpovědi pro daný typ modelu"""
    
    print(f"🤖 Generuji odpovědi pro model: {model_type}")
    
    # Načtení benchmark datasetu
    if os.path.exists("results/benchmark_dataset.json"):
        with open("results/benchmark_dataset.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        questions = dataset.get("questions", [])
    else:
        # Fallback na základní otázky
        questions = [
            {"id": "Q1", "question": "Pane Babiši, jak hodnotíte současnou inflaci?"},
            {"id": "Q2", "question": "Co si myslíte o opozici?"},
            {"id": "Q3", "question": "Jak se vám daří s rodinou?"},
            {"id": "Q4", "question": "Můžete vysvětlit vaši roli v té chemičce?"},
            {"id": "Q5", "question": "Jak vnímáte reakce Bruselu na ekonomickou situaci v Česku?"}
        ]
    
    responses = []
    
    for question in questions:
        response = generate_mock_response(question["question"], model_type)
        
        responses.append({
            "id": question["id"],
            "question": question["question"],
            "response": response,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat()
        })
    
    # Uložení odpovědí
    output_file = os.path.join(output_dir, "responses.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Vygenerováno {len(responses)} odpovědí")
    print(f"💾 Uloženo: {output_file}")
    
    # Výpis několika příkladů
    print(f"\n📝 Příklady odpovědí ({model_type}):")
    for i, resp in enumerate(responses[:3]):
        print(f"   {i+1}. {resp['question']}")
        print(f"      → {resp['response']}")
        print()
    
    return responses

def generate_real_responses(model_type: str, output_dir: str):
    """Generuje skutečné odpovědi pomocí LLM (pro budoucí použití)"""
    
    # TODO: Implementovat skutečné generování pomocí OpenAI API nebo Hugging Face
    # Prozatím používáme mock odpovědi
    
    print(f"⚠️  Skutečné generování pomocí LLM není implementováno")
    print(f"   Používám mock odpovědi pro {model_type}")
    
    return generate_responses(model_type, output_dir)

if __name__ == "__main__":
    # Test generování odpovědí
    print("🧪 Test generování odpovědí...")
    
    # Test před fine-tuningem
    base_responses = generate_responses("base", "results/before_finetune/")
    
    # Test po fine-tuningem
    finetuned_responses = generate_responses("finetuned", "results/after_finetune/")
    
    print(f"\n✅ Test dokončen:")
    print(f"   Před fine-tuningem: {len(base_responses)} odpovědí")
    print(f"   Po fine-tuningem: {len(finetuned_responses)} odpovědí") 