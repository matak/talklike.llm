#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generování odpovědí pro benchmarking TalkLike.LLM
Používá skutečný model s adaptérem pro generování odpovědí
"""

# Import a nastavení prostředí
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import setup_environment

import json
import os
import random
import torch
from datetime import datetime
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import centralizované funkce pro nastavení pad_tokenu
sys.path.append('../2_finetunning')
from tokenizer_utils import setup_tokenizer_and_model

# Nastavení MODEL_AVAILABLE na True, protože budeme používat reálný model
MODEL_AVAILABLE = True
print("✅ Reálný model bude načítán z Hugging Face")

def load_benchmark_model(model_type: str):
    """Načte model pro benchmarking"""
    if not MODEL_AVAILABLE:
        print("❌ Model není dostupný - MODEL_AVAILABLE = False")
        return None, None
    
    try:
        if model_type == "finetuned":
            # Reálný fine-tuned model z Hugging Face
            model_path = "mcmatak/mistral-babis-model"
            
            print(f"🤖 Načítám reálný fine-tuned model...")
            print(f"   Model: {model_path}")
            
            # Načtení tokenizeru a modelu
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Nastavení pad tokenu
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model.eval()
            
        elif model_type == "base":
            # Základní model bez fine-tuningu
            base_model = "mistralai/Mistral-7B-Instruct-v0.3"
            
            print(f"🤖 Načítám základní model...")
            print(f"   Base model: {base_model}")
            
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Použití centralizované funkce pro nastavení pad_tokenu
            tokenizer, model = setup_tokenizer_and_model(base_model, model)
            model.eval()
            
        else:
            raise ValueError(f"Neznámý typ modelu: {model_type}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Chyba při načítání modelu {model_type}: {e}")
        return None, None

def generate_real_response(model, tokenizer, question: str, model_type: str) -> str:
    """Generuje skutečnou odpověď pomocí modelu"""
    
    try:
        if model_type == "finetuned":
            # Pro fine-tuned model použijeme system prompt
            system_prompt = """Jsi Andrej Babiš, český politik a podnikatel. Tvým úkolem je odpovídat na otázky v charakteristickém Babišově stylu.

Charakteristické prvky tvého stylu:
- Typické fráze: "Hele, ...", "To je skandál!", "Já makám", "Opozice krade", "V Bruselu"
- Slovenské odchylky: "sme", "som", "makáme", "centralizácia"
- Emotivní výrazy: "to je šílený!", "tragédyje!", "kampááň!"
- Přirovnání: "jak když kráva hraje na klavír", "jak když dítě řídí tank"
- První osoba: "Já jsem...", "Moje rodina...", "Já makám..."

Odpovídej vždy v první osobě jako Andrej Babiš, používej jeho charakteristické fráze, buď emotivní a přímý."""

            prompt = f"<s>[INST] {system_prompt}\n\nOtázka: {question} [/INST]"
            
        else:  # base model
            # Pro základní model použijeme jednoduchý prompt
            prompt = f"<s>[INST] Otázka: {question} [/INST]"
        
        # Generování odpovědi pomocí modelu
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=300,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Dekódování odpovědi
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Vylepšené vyčištění odpovědi
        response = response.strip()
        
        # Odstranění možných zbytků promptu
        cleanup_patterns = [
            f"Otázka: {question}",
            f"Otázka: {question} [/INST]",
            f"<s>[INST] Otázka: {question} [/INST]",
            question,  # Původní otázka
        ]
        
        for pattern in cleanup_patterns:
            if response.startswith(pattern):
                response = response[len(pattern):].strip()
                break
        
        # Odstranění prázdných řádků na začátku
        response = response.lstrip('\n').strip()
        
        return response if response else "Omlouvám se, nemohu odpovědět."
        
    except Exception as e:
        print(f"❌ Chyba při generování odpovědi: {e}")
        return f"Chyba při generování: {str(e)}"

def generate_mock_response(question: str, model_type: str) -> str:
    """Generuje mock odpověď pro testovací účely (fallback)"""
    
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
    
    # Načtení modelu - VŽDY používáme reálný model
    print(f"🔧 Načítám reálný model: {model_type}")
    model, tokenizer = load_benchmark_model(model_type)
    
    if model is None or tokenizer is None:
        print(f"❌ Nepodařilo se načíst model {model_type}")
        print("💡 Zkontrolujte:")
        print("   - Máte přístup k modelu mcmatak/babis-mistral-adapter?")
        print("   - Jste přihlášeni na Hugging Face?")
        print("   - Máte dostatek místa v cache?")
        raise RuntimeError(f"Model {model_type} se nepodařilo načíst")
    
    print(f"✅ Reálný model načten: {model_type}")
    
    responses = []
    
    for i, question in enumerate(questions):
        print(f"   Generuji odpověď {i+1}/{len(questions)}: {question['question'][:50]}...")
        
        # VŽDY používáme reálný model
        response = generate_real_response(model, tokenizer, question["question"], model_type)
        
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
    
    # Uvolnění paměti
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return responses

def generate_real_responses(model_type: str, output_dir: str):
    """Wrapper pro skutečné generování (pro kompatibilitu)"""
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