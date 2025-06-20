# Načtení tokenizeru
tokenizer = AutoTokenizer.from_pretrained(config.base_model)

# Přidání padding tokenu pokud chybí
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Tokenizer načten: {config.base_model}")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Pad token: {tokenizer.pad_token}")