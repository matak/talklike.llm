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

def load_benchmark_model(model_type: str):
    """Načte model pro benchmarking"""
    
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

def generate_real_response(model, tokenizer, question: str, model_type: str) -> str:
    """Generuje skutečnou odpověď pomocí modelu"""
    
    # Použití apply_chat_template pro správné formátování
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generování odpovědi pomocí modelu
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    # Přesun input tensors na stejné zařízení jako model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
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
        question,  # Původní otázka
        prompt,    # Celý prompt
    ]
    
    for pattern in cleanup_patterns:
        if response.startswith(pattern):
            response = response[len(pattern):].strip()
            break
    
    # Odstranění prázdných řádků na začátku
    response = response.lstrip('\n').strip()
    
    return response if response else "Omlouvám se, nemohu odpovědět."

def generate_responses(model_type: str, output_dir: str):
    """Generuje odpovědi pro daný typ modelu"""
    
    print(f"🤖 Generuji odpovědi pro model: {model_type}")
    
    # Načtení benchmark datasetu
    with open("3_benchmarking/results/benchmark_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    questions = dataset.get("questions", [])
    
    # Načtení modelu
    print(f"🔧 Načítám reálný model: {model_type}")
    model, tokenizer = load_benchmark_model(model_type)
    
    print(f"✅ Reálný model načten: {model_type}")
    
    responses = []
    
    for i, question in enumerate(questions):
        print(f"   Generuji odpověď {i+1}/{len(questions)}: {question['question'][:50]}...")
        
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
    base_responses = generate_responses("base", "3_benchmarking/results/before_finetune/")
    
    # Test po fine-tuningem
    finetuned_responses = generate_responses("finetuned", "3_benchmarking/results/after_finetune/")
    
    print(f"\n✅ Test dokončen:")
    print(f"   Před fine-tuningem: {len(base_responses)} odpovědí")
    print(f"   Po fine-tuningem: {len(finetuned_responses)} odpovědí") 