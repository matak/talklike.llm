# Opravy pad_token pro Fine-tuning

## Problém

Při analýze kódu jsem identifikoval **duplicitní padding**, který mohl způsobovat problémy při fine-tuningu:

### ❌ Původní problémy:

1. **Duplicitní padding** - Padding se aplikoval dvakrát:
   - V `tokenize_function()` s `padding=True`
   - Znovu v manuální `fix_padding()` funkci

2. **Nekonzistentní přístup** - Automatický padding v tokenizeru + manuální oprava

3. **Potenciální konflikt** s `DataCollatorForLanguageModeling`

## ✅ Řešení

### 1. Opravené soubory:

- `finetune_babis.py` - hlavní fine-tuning script
- `tokenizer_utils.py` - tokenizace funkce
- `test_tokenization.py` - test soubor
- `create_qlora_adapter.py` - QLoRA adaptér

### 2. Konkrétní změny:

#### A) `tokenize_function()` - odstranění padding=True
```python
# PŘED:
tokenized = tokenizer(
    examples["text"],
    truncation=True,
    padding=True,  # ❌ Duplicitní padding
    max_length=max_length,
    return_tensors=None
)

# PO:
tokenized = tokenizer(
    examples["text"],
    truncation=True,
    padding=False,  # ✅ Padding se řeší v DataCollator
    max_length=max_length,
    return_tensors=None
)
```

#### B) Odstranění manuální `fix_padding()` funkce
```python
# ❌ ODSTRANĚNO:
def fix_padding(example):
    # Manuální padding logika
    ...

tokenized_dataset = tokenized_dataset.map(fix_padding)
```

#### C) Vylepšení DataCollator konfigurace
```python
# ✅ VYLEPŠENO:
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    return_tensors="pt",
    pad_to_multiple_of=8,
    padding=True,  # Explicitně povolíme padding
)
```

## 🎯 Výhody oprav:

1. **Jednotný padding** - pouze v DataCollator
2. **Lepší výkon** - méně duplicitních operací
3. **Konzistentní chování** - standardní Hugging Face přístup
4. **Správné labels** - padding tokeny mají label -100 (ignorovány v loss)

## 🔧 Jak to funguje nyní:

1. **Tokenizace** - bez padding, pouze truncation
2. **DataCollator** - automaticky přidá padding na nejdelší sekvenci v batch
3. **Labels** - padding tokeny automaticky dostanou label -100
4. **Attention mask** - správně nastavena pro padding

## ✅ Správné nastavení pad_token zůstává:

```python
# V setup_tokenizer_and_model():
if base_tokenizer.pad_token is None:
    if base_tokenizer.eos_token:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    else:
        base_tokenizer.add_special_tokens({"pad_token": "<pad>"})
        base_model.resize_token_embeddings(len(base_tokenizer))
```

Toto nastavení je správné a zůstává beze změny.

## 🧪 Testování:

Spusťte `test_tokenization.py` pro ověření, že padding funguje správně:

```bash
python test_tokenization.py
```

Měli byste vidět:
- ✅ Správné délky tokenů
- ✅ Data collator test úspěšný
- ✅ Konzistentní padding 