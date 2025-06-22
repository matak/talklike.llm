#!/usr/bin/env python3
"""
Test script pro porovnání formátování promptů
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import setup_environment

import torch
from transformers import AutoTokenizer
from train_utils import generate_response

def test_prompt_formatting():
    """Porovná různé způsoby formátování promptů"""
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    test_prompt = "Co si myslíš o inflaci?"
    
    print(f"🔧 Testuji formátování promptů pro model: {model_name}")
    print("=" * 60)
    
    try:
        # Načtení tokenizeru
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir='/workspace/.cache/huggingface/transformers',
            local_files_only=False,
            resume_download=True,
            force_download=False,
            trust_remote_code=True
        )
        
        print(f"✅ Tokenizer načten, vocab size: {len(tokenizer)}")
        print(f"🔧 EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        print(f"🔧 BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
        
        # Test 1: Přímé formátování (starý způsob)
        print("\n📝 Test 1: Přímé formátování promptu")
        print("-" * 40)
        direct_prompt = test_prompt
        print(f"Původní prompt: {direct_prompt}")
        
        # Tokenizace přímého promptu
        direct_inputs = tokenizer(direct_prompt, return_tensors="pt")
        print(f"Tokenizovaná délka: {direct_inputs['input_ids'].shape[1]}")
        print(f"Prvních 10 tokenů: {direct_inputs['input_ids'][0][:10].tolist()}")
        
        # Test 2: apply_chat_template s add_generation_prompt=True
        print("\n📝 Test 2: apply_chat_template s add_generation_prompt=True")
        print("-" * 40)
        messages = [{"role": "user", "content": test_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"Formátovaný prompt: '{formatted_prompt}'")
        
        # Tokenizace formátovaného promptu
        formatted_inputs = tokenizer(formatted_prompt, return_tensors="pt")
        print(f"Tokenizovaná délka: {formatted_inputs['input_ids'].shape[1]}")
        print(f"Prvních 10 tokenů: {formatted_inputs['input_ids'][0][:10].tolist()}")
        
        # Test 3: apply_chat_template s add_generation_prompt=False
        print("\n📝 Test 3: apply_chat_template s add_generation_prompt=False")
        print("-" * 40)
        formatted_prompt_no_gen = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        print(f"Formátovaný prompt: '{formatted_prompt_no_gen}'")
        print(f"Délka: {len(formatted_prompt_no_gen)} znaků")
        
        # Porovnání
        print("\n📊 Porovnání:")
        print("-" * 40)
        print(f"Přímé formátování: {direct_inputs['input_ids'].shape[1]} tokenů")
        print(f"apply_chat_template (s gen): {formatted_inputs['input_ids'].shape[1]} tokenů")
        print(f"apply_chat_template (bez gen): {len(tokenizer(formatted_prompt_no_gen, return_tensors='pt')['input_ids'][0])} tokenů")
        print(f"Rozdíl (s gen): {formatted_inputs['input_ids'].shape[1] - direct_inputs['input_ids'].shape[1]} tokenů")
        
        # Test 4: Tokenizace a dekódování
        print("\n📝 Test 4: Tokenizace a dekódování")
        print("-" * 40)
        
        # Dekódování zpět (bez skip_special_tokens)
        decoded_with_special = tokenizer.decode(formatted_inputs['input_ids'][0], skip_special_tokens=False)
        print(f"Dekódováno s special tokens: '{decoded_with_special}'")
        
        # Dekódování zpět (se skip_special_tokens)
        decoded_without_special = tokenizer.decode(formatted_inputs['input_ids'][0], skip_special_tokens=True)
        print(f"Dekódováno bez special tokens: '{decoded_without_special}'")
        
        # Test 5: Simulace generování (přidání EOS tokenu)
        print("\n📝 Test 5: Simulace generování s EOS tokenem")
        print("-" * 40)
        
        # Přidání EOS tokenu na konec
        input_ids_with_eos = torch.cat([formatted_inputs['input_ids'][0], torch.tensor([tokenizer.eos_token_id])])
        print(f"Tokenizovaná délka s EOS: {input_ids_with_eos.shape[0]} tokenů")
        print(f"Posledních 5 tokenů s EOS: {input_ids_with_eos[-5:].tolist()}")
        
        # Dekódování s EOS
        decoded_with_eos = tokenizer.decode(input_ids_with_eos, skip_special_tokens=True)
        print(f"Dekódováno s EOS: '{decoded_with_eos}'")
        
        # Dekódování s EOS (bez skip_special_tokens)
        decoded_with_eos_special = tokenizer.decode(input_ids_with_eos, skip_special_tokens=False)
        print(f"Dekódováno s EOS a special tokens: '{decoded_with_eos_special}'")
        
        # Test 6: Použití v generate_response funkci
        print("\n📝 Test 6: Použití v generate_response funkci")
        print("-" * 40)
        print("Tato funkce nyní automaticky používá apply_chat_template pokud je dostupné")
        
        # Simulace volání generate_response (bez skutečného modelu)
        if hasattr(tokenizer, 'apply_chat_template'):
            print("✅ generate_response bude používat apply_chat_template")
            messages = [{"role": "user", "content": test_prompt}]
            expected_formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            print(f"Očekávaný formát: '{expected_formatted}'")
        else:
            print("⚠️ generate_response bude používat přímé formátování")
            print(f"Očekávaný formát: '{test_prompt}'")
        
    except Exception as e:
        print(f"❌ Chyba při testování: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prompt_formatting() 