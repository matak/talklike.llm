#!/usr/bin/env python3
"""
Test script pro ověření, jak Mistral zpracovává system message
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import setup_environment

from transformers import AutoTokenizer

def test_mistral_system_message():
    """Testuje, jak Mistral zpracovává system message"""
    
    print("🔧 Testuji Mistral system message handling...")
    
    # Načtení Mistral tokenizeru
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        cache_dir='/workspace/.cache/huggingface/transformers',
        local_files_only=False,
        resume_download=True,
        force_download=False
    )
    
    print(f"✅ Tokenizer načten, vocab size: {len(tokenizer)}")
    
    # Test 1: Konverzace se system message
    messages_with_system = [
        {
            "role": "system",
            "content": "Jsi Andrej Babiš, český politik a podnikatel. Mluvíš jako on - používáš jeho charakteristické fráze, styl komunikace a názory."
        },
        {
            "role": "user", 
            "content": "Pane Babiši, můžete vysvětlit vaši roli v té chemičce?"
        },
        {
            "role": "assistant",
            "content": "Hele, ta továrna? To už jsem dávno předal. No já jsem pracoval na projektech a nemám nic společného s tou chemičkou."
        }
    ]
    
    print("\n📝 Test 1: Konverzace se system message")
    print("Messages:")
    for msg in messages_with_system:
        print(f"  {msg['role']}: {msg['content'][:50]}...")
    
    try:
        formatted_with_system = tokenizer.apply_chat_template(
            messages_with_system, 
            tokenize=False, 
            add_generation_prompt=False
        )
        print(f"\n✅ Výsledek se system message:")
        print(f"Text: {formatted_with_system}")
        print(f"Délka: {len(formatted_with_system)} znaků")
        
        # Tokenizace pro kontrolu
        tokens_with_system = tokenizer(formatted_with_system, return_tensors="pt")
        print(f"Tokeny: {tokens_with_system['input_ids'].shape[1]} tokenů")
        
    except Exception as e:
        print(f"❌ Chyba se system message: {e}")
    
    # Test 2: Konverzace bez system message
    messages_without_system = [
        {
            "role": "user", 
            "content": "Pane Babiši, můžete vysvětlit vaši roli v té chemičce?"
        },
        {
            "role": "assistant",
            "content": "Hele, ta továrna? To už jsem dávno předal. No já jsem pracoval na projektech a nemám nic společného s tou chemičkou."
        }
    ]
    
    print("\n📝 Test 2: Konverzace bez system message")
    print("Messages:")
    for msg in messages_without_system:
        print(f"  {msg['role']}: {msg['content'][:50]}...")
    
    try:
        formatted_without_system = tokenizer.apply_chat_template(
            messages_without_system, 
            tokenize=False, 
            add_generation_prompt=False
        )
        print(f"\n✅ Výsledek bez system message:")
        print(f"Text: {formatted_without_system}")
        print(f"Délka: {len(formatted_without_system)} znaků")
        
        # Tokenizace pro kontrolu
        tokens_without_system = tokenizer(formatted_without_system, return_tensors="pt")
        print(f"Tokeny: {tokens_without_system['input_ids'].shape[1]} tokenů")
        
    except Exception as e:
        print(f"❌ Chyba bez system message: {e}")
    
    # Test 3: Kontrola chat template
    print(f"\n🔍 Chat template pro Mistral:")
    if hasattr(tokenizer, 'chat_template'):
        print(f"Template: {tokenizer.chat_template}")
    else:
        print("❌ Žádný chat template")
    
    # Test 4: Manuální formátování se system message
    print(f"\n📝 Test 4: Manuální formátování se system message")
    try:
        # Zkusíme manuální formátování podle Mistral specifikace
        system_content = messages_with_system[0]['content']
        user_content = messages_with_system[1]['content']
        assistant_content = messages_with_system[2]['content']
        
        # Mistral formát: system message na začátku, pak [INST] user [/INST] assistant
        manual_format = f"{system_content}\n\n<s>[INST] {user_content} [/INST] {assistant_content}</s>"
        
        print(f"Manuální formát:")
        print(f"Text: {manual_format}")
        print(f"Délka: {len(manual_format)} znaků")
        
        # Tokenizace
        tokens_manual = tokenizer(manual_format, return_tensors="pt")
        print(f"Tokeny: {tokens_manual['input_ids'].shape[1]} tokenů")
        
    except Exception as e:
        print(f"❌ Chyba při manuálním formátování: {e}")

if __name__ == "__main__":
    test_mistral_system_message() 