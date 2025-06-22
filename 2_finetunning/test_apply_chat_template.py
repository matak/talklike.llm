#!/usr/bin/env python3
"""
Test script pro ověření apply_chat_template funkčnosti
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import setup_environment

from transformers import AutoTokenizer
from data_utils import load_babis_data, prepare_training_data
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
    conversations = load_babis_data("data/all.jsonl")
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
            
            # Příprava dat
            training_data = prepare_training_data(conversations[:5], model_name=model_name)  # Pouze prvních 5 pro test
            print(f"✅ Připraveno {len(training_data)} vzorků")
            
            # Test tokenizace
            if len(training_data) > 0:
                # Test prvního vzorku
                first_sample = training_data[0]
                print(f"📝 Testuji první vzorek:")
                print(f"   Typ dat: {type(first_sample)}")
                print(f"   Klíče: {list(first_sample.keys())}")
                
                if "messages" in first_sample:
                    messages = first_sample["messages"]
                    print(f"   Počet zpráv: {len(messages)}")
                    
                    # Test apply_chat_template
                    try:
                        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                        print(f"✅ apply_chat_template úspěšný")
                        print(f"   Formátovaný text (prvních 200 znaků): {formatted_text[:200]}...")
                        
                        # Test tokenizace
                        tokenized = tokenizer(formatted_text, return_tensors="pt")
                        print(f"✅ Tokenizace úspěšná, délka: {tokenized['input_ids'].shape[1]}")
                        
                    except Exception as e:
                        print(f"❌ Chyba při apply_chat_template: {e}")
                
                # Test batch tokenizace
                try:
                    tokenized_batch = tokenize_function({"messages": [item["messages"] for item in training_data]}, tokenizer, max_length=512)
                    print(f"✅ Batch tokenizace úspěšná, {len(tokenized_batch['input_ids'])} vzorků")
                except Exception as e:
                    print(f"❌ Chyba při batch tokenizaci: {e}")
            
        except Exception as e:
            print(f"❌ Chyba při testování {model_name}: {e}")
        
        print()

if __name__ == "__main__":
    test_apply_chat_template() 