#!/usr/bin/env python3
"""
Test script pro ověření apply_chat_template funkčnosti
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import setup_environment

from transformers import AutoTokenizer
from data_utils import load_model_data, prepare_training_data
from tokenizer_utils import tokenize_function

def test_apply_chat_template():
    """Testuje apply_chat_template s různými modely"""
    
    # Testovací modely
    test_models = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "microsoft/DialoGPT-medium",
        "meta-llama/Llama-2-7b-chat-hf"
    ]
    
    # Načtení testovacích dat
    print("📊 Načítám testovací data...")
    conversations = load_model_data("data/all.jsonl")
    print(f"✅ Načteno {len(conversations)} konverzací")
    
    # Test s každým modelem
    for model_name in test_models:
        print(f"\n🔧 Testuji model: {model_name}")
        print("=" * 60)
        
        try:
            # Načtení tokenizeru
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir='/workspace/.cache/huggingface/transformers',
                local_files_only=False,
                resume_download=True,
                force_download=False
            )
            
            # Kontrola, zda tokenizer podporuje apply_chat_template
            if not hasattr(tokenizer, 'apply_chat_template'):
                print(f"⚠️ Tokenizer pro {model_name} nepodporuje apply_chat_template")
                continue
            
            print(f"✅ Tokenizer načten, vocab size: {len(tokenizer)}")
            
            # Příprava dat s tokenizerem (nový přístup)
            training_data = prepare_training_data(conversations[:5], model_name=model_name, tokenizer=tokenizer)
            print(f"✅ Připraveno {len(training_data)} vzorků")
            
            # Test tokenizace
            if len(training_data) > 0:
                # Test prvního vzorku
                first_sample = training_data[0]
                print(f"📝 Testuji první vzorek:")
                print(f"   Typ dat: {type(first_sample)}")
                print(f"   Klíče: {list(first_sample.keys())}")
                
                if "text" in first_sample:
                    text = first_sample["text"]
                    print(f"   Text (prvních 200 znaků): {text[:200]}...")
                    
                    # Test tokenizace
                    try:
                        tokenized = tokenizer(text, return_tensors="pt")
                        print(f"✅ Tokenizace úspěšná, délka: {tokenized['input_ids'].shape[1]}")
                        
                    except Exception as e:
                        print(f"❌ Chyba při tokenizaci: {e}")
                
                # Test batch tokenizace
                try:
                    tokenized_batch = tokenize_function({"text": [item["text"] for item in training_data]}, tokenizer, max_length=512)
                    print(f"✅ Batch tokenizace úspěšná, {len(tokenized_batch['input_ids'])} vzorků")
                except Exception as e:
                    print(f"❌ Chyba při batch tokenizaci: {e}")
            
        except Exception as e:
            print(f"❌ Chyba při testování {model_name}: {e}")
        
        print()

if __name__ == "__main__":
    test_apply_chat_template() 