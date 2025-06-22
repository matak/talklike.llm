#!/usr/bin/env python3
"""
Test script pro ověření opravy Mistral system message handling
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import setup_environment

from transformers import AutoTokenizer
from data_utils import load_model_data, prepare_training_data

def test_mistral_fix():
    """Testuje opravené Mistral formátování"""
    
    print("🔧 Testuji opravené Mistral formátování...")
    
    # Načtení testovacích dat
    print("📊 Načítám testovací data...")
    conversations = load_model_data("data/all.jsonl")
    print(f"✅ Načteno {len(conversations)} konverzací")
    
    # Načtení Mistral tokenizeru
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        cache_dir='/workspace/.cache/huggingface/transformers',
        local_files_only=False,
        resume_download=True,
        force_download=False
    )
    
    print(f"✅ Tokenizer načten, vocab size: {len(tokenizer)}")
    
    # Test s opravenou funkcí
    print("\n🔧 Testuji opravené prepare_training_data...")
    training_data = prepare_training_data(conversations[:3], model_name="mistralai/Mistral-7B-Instruct-v0.3", tokenizer=tokenizer)
    print(f"✅ Připraveno {len(training_data)} vzorků")
    
    # Kontrola prvního vzorku
    if len(training_data) > 0:
        first_sample = training_data[0]
        print(f"\n📝 První vzorek:")
        print(f"   Typ dat: {type(first_sample)}")
        print(f"   Klíče: {list(first_sample.keys())}")
        
        if "text" in first_sample:
            text = first_sample["text"]
            print(f"   Text (prvních 300 znaků): {text[:300]}...")
            print(f"   Celková délka: {len(text)} znaků")
            
            # Kontrola, zda obsahuje system message
            if "Jsi Andrej Babiš" in text:
                print("✅ System message je přítomna!")
            else:
                print("❌ System message chybí!")
            
            # Kontrola, zda obsahuje Mistral formát
            if "[INST]" in text and "[/INST]" in text:
                print("✅ Mistral formát je správný!")
            else:
                print("❌ Mistral formát chybí!")
            
            # Test tokenizace
            try:
                tokenized = tokenizer(text, return_tensors="pt")
                print(f"✅ Tokenizace úspěšná, délka: {tokenized['input_ids'].shape[1]} tokenů")
                
            except Exception as e:
                print(f"❌ Chyba při tokenizaci: {e}")
    
    # Kontrola druhého vzorku
    if len(training_data) > 1:
        second_sample = training_data[1]
        if "text" in second_sample:
            text = second_sample["text"]
            print(f"\n📝 Druhý vzorek (prvních 200 znaků): {text[:200]}...")
            
            # Kontrola, zda obsahuje system message
            if "Jsi Andrej Babiš" in text:
                print("✅ System message je přítomna!")
            else:
                print("❌ System message chybí!")

if __name__ == "__main__":
    test_mistral_fix() 