#!/usr/bin/env python3
"""
Test tokenizace pro fine-tuning data
"""

# Import setup_environment pro správné nastavení prostředí
import setup_environment

import os
import json
import torch
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)

def setup_tokenizer_and_model(model_name, base_model):
    """Nastaví tokenizer a model pro fine-tuning"""
    
    # 1. Načtení tokenizeru
    base_tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"📊 Původní délka tokenizeru: {len(base_tokenizer)}")
    
    # 2. Kontrola a přidání pad tokenu
    if base_tokenizer.pad_token is None:
        # Zkusíme použít existující tokeny
        if base_tokenizer.eos_token:
            base_tokenizer.pad_token = base_tokenizer.eos_token
            print(f"✅ Používám EOS token jako PAD: {base_tokenizer.pad_token}")
        else:
            # Přidáme nový pad token
            base_tokenizer.add_special_tokens({"pad_token": "<pad>"})
            print(f"✅ Přidán nový pad token: {base_tokenizer.pad_token}")
            
            # Důležité: Resize model embeddings
            base_model.resize_token_embeddings(len(base_tokenizer))
            print(f"📊 Model embeddings resized na: {len(base_tokenizer)}")
    else:
        print(f"ℹ️ Pad token už existuje: {base_tokenizer.pad_token}")
    
    # 3. Synchronizace s modelem
    if hasattr(base_model.config, 'pad_token_id'):
        old_pad_id = base_model.config.pad_token_id
        base_model.config.pad_token_id = base_tokenizer.pad_token_id
        print(f"🔄 Pad token ID změněn: {old_pad_id} → {base_model.config.pad_token_id}")
    else:
        print("⚠️ Model nemá pad_token_id v config")
    
    # 4. Kontrola konzistence
    try:
        assert base_tokenizer.pad_token_id == base_model.config.pad_token_id, \
            "Tokenizer a model mají různé pad token ID!"
        print(f"✅ Tokenizer a model synchronizovány")
    except AssertionError as e:
        print(f"❌ Chyba synchronizace: {e}")
        # Pokusíme se opravit
        base_model.config.pad_token_id = base_tokenizer.pad_token_id
        print(f"🔧 Opraveno: pad_token_id nastaven na {base_tokenizer.pad_token_id}")
    
    return base_tokenizer, base_model

def test_tokenization():
    """Testuje tokenizaci a data collator"""
    
    # Načtení modelu a tokenizeru
    model_name = "microsoft/DialoGPT-medium"  # Použijeme menší model pro test
    
    print(f"🤖 Načítám model: {model_name}")
    
    # Načtení modelu (bez quantization pro test)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Použití vylepšené metody pro nastavení tokenizeru
    print("\n🔧 Nastavuji tokenizer a model...")
    tokenizer, model = setup_tokenizer_and_model(model_name, model)
    
    # Testovací data
    test_data = [
        {
            "text": "<|system|>\nJste Andrej Babiš, předseda hnutí ANO a bývalý premiér České republiky.<|end|>\n<|user|>\nJak hodnotíte současnou inflaci?<|end|>\n<|assistant|>\nInflace je vážný problém, který postihuje všechny občany.<|end|>\n"
        },
        {
            "text": "<|system|>\nJste Andrej Babiš, předseda hnutí ANO a bývalý premiér České republiky.<|end|>\n<|user|>\nCo si myslíte o opozici?<|end|>\n<|assistant|>\nOpozice kritizuje, ale nemá řešení.<|end|>\n"
        }
    ]
    
    print("\n📊 Testovací data:")
    for i, sample in enumerate(test_data):
        print(f"  Vzorek {i+1}: {len(sample['text'])} znaků")
    
    # Vytvoření datasetu
    dataset = Dataset.from_list(test_data)
    
    # Tokenizace
    def tokenize_function(examples, tokenizer, max_length=512):
        """Tokenizuje text pro fine-tuning"""
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    print("\n🔤 Tokenizuji data...")
    tokenize_func = lambda examples: tokenize_function(examples, tokenizer, 512)
    tokenized_dataset = dataset.map(
        tokenize_func,
        batched=True,
        remove_columns=dataset.column_names,
        batch_size=100
    )
    
    # Oprava padding
    print("🔧 Opravuji padding...")
    def fix_padding(example):
        max_len = 512
        current_len = len(example['input_ids'])
        
        if current_len < max_len:
            padding_length = max_len - current_len
            example['input_ids'] = example['input_ids'] + [tokenizer.pad_token_id] * padding_length
            example['attention_mask'] = example['attention_mask'] + [0] * padding_length
            example['labels'] = example['labels'] + [-100] * padding_length
        elif current_len > max_len:
            example['input_ids'] = example['input_ids'][:max_len]
            example['attention_mask'] = example['attention_mask'][:max_len]
            example['labels'] = example['labels'][:max_len]
        
        return example
    
    tokenized_dataset = tokenized_dataset.map(fix_padding, desc="Opravuji padding")
    
    # Kontrola délky
    print("\n📏 Kontrola délky tokenů:")
    for i, sample in enumerate(tokenized_dataset):
        print(f"  Vzorek {i+1}: {len(sample['input_ids'])} tokenů")
    
    # Test data collator
    print("\n🔧 Testuji data collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
        pad_to_multiple_of=8,
    )
    
    try:
        test_batch = data_collator([tokenized_dataset[0], tokenized_dataset[1]])
        print("✅ Data collator test úspěšný!")
        print(f"📊 Batch shape: {test_batch['input_ids'].shape}")
        print(f"📊 Labels shape: {test_batch['labels'].shape}")
        
        # Dekódování pro kontrolu
        print("\n📝 Dekódovaný text:")
        decoded = tokenizer.decode(test_batch['input_ids'][0], skip_special_tokens=False)
        print(f"  Vzorek 1: {decoded[:200]}...")
        
        # Dodatečné informace o tokenizeru
        print(f"\n📋 Tokenizer informace:")
        print(f"  Pad token: {tokenizer.pad_token}")
        print(f"  Pad token ID: {tokenizer.pad_token_id}")
        print(f"  EOS token: {tokenizer.eos_token}")
        print(f"  Vocab size: {len(tokenizer)}")
        print(f"  Model vocab size: {model.config.vocab_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data collator test selhal: {e}")
        return False

if __name__ == "__main__":
    success = test_tokenization()
    if success:
        print("\n🎉 Test úspěšný! Můžete spustit fine-tuning.")
    else:
        print("\n❌ Test selhal! Zkontrolujte konfiguraci.") 