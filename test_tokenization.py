#!/usr/bin/env python3
"""
Test script pro ověření tokenizace a data collator
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)

def test_tokenization():
    """Testuje tokenizaci a data collator"""
    
    # Načtení modelu a tokenizeru
    model_name = "microsoft/DialoGPT-medium"  # Použijeme menší model pro test
    
    print(f"🤖 Načítám model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Přidání pad tokenu
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Testovací data
    test_data = [
        {
            "text": "<|system|>\nJste Andrej Babiš, předseda hnutí ANO a bývalý premiér České republiky.<|end|>\n<|user|>\nJak hodnotíte současnou inflaci?<|end|>\n<|assistant|>\nInflace je vážný problém, který postihuje všechny občany.<|end|>\n"
        },
        {
            "text": "<|system|>\nJste Andrej Babiš, předseda hnutí ANO a bývalý premiér České republiky.<|end|>\n<|user|>\nCo si myslíte o opozici?<|end|>\n<|assistant|>\nOpozice kritizuje, ale nemá řešení.<|end|>\n"
        }
    ]
    
    print("📊 Testovací data:")
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