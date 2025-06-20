def tokenize_function(examples):
    """Tokenizuje texty v datasetu"""
    # Používáme full_conversation pro trénování
    texts = examples['full_conversation']
    
    # Tokenizace
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=config.max_seq_length,
        return_tensors="pt"
    )
    
    # Nastavení labels na input_ids pro causal LM
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

# Tokenizace datasetu
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

print("Dataset tokenizován!")
print(f"Train samples: {len(tokenized_dataset['train'])}")
print(f"Validation samples: {len(tokenized_dataset['validation'])}")
print(f"\nPříklad tokenizovaných dat:")
print(f"Input shape: {tokenized_dataset['train'][0]['input_ids'].shape}")
print(f"Labels shape: {tokenized_dataset['train'][0]['labels'].shape}") 