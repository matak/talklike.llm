# LoRA Adaptéry - Snadné připojení k různým modelům

> **📚 Navigace:** [🏠 Hlavní projekt](../README.md) | [📊 Příprava dat](../1_data_preparation/README.md) | [🏋️ Detailní dokumentace](README.md) | [📈 Benchmarking](../3_benchmarking/README.md)

Tento dokument vysvětluje, jak vytvořit a používat LoRA adaptéry z vašeho datasetu, které se dají snadno připojit k různým modelům.

## 🎯 Co jsou LoRA adaptéry?

**LoRA (Low-Rank Adaptation)** je technika, která umožňuje:
- ✅ **Malou velikost** (pár MB vs GB celého modelu)
- ✅ **Snadné připojení** k jakémukoli kompatibilnímu modelu
- ✅ **Rychlé načítání** a přepínání
- ✅ **Možnost kombinovat** více adaptérů

## 📁 Struktura souborů

```
2_finetunning/
├── create_qlora_adapter.py    # Vytvoření QLoRA adaptéru
├── test_adapter.py            # Testování adaptéru
├── finetune.py              # Původní finetuning (už používá LoRA)
└── adapters/                  # Složka pro uložení adaptérů
    ├── babis_adapter/         # Vytvořený adaptér
    ├── babis_adapter_config.json  # Konfigurace adaptéru
    └── ...
```

## 🚀 Vytvoření adaptéru

### 1. Základní vytvoření QLoRA adaptéru

```bash
python create_qlora_adapter.py \
    --base-model microsoft/DialoGPT-medium \
    --dataset ../data/final/all.jsonl \
    --output-dir ./adapters \
    --adapter-name babis_adapter \
    --epochs 3 \
    --batch-size 4
```

### 2. Pokročilé nastavení

```bash
python create_qlora_adapter.py \
    --base-model microsoft/DialoGPT-medium \
    --dataset ../data/final/all.jsonl \
    --output-dir ./adapters \
    --adapter-name babis_adapter_advanced \
    --r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --max-length 2048 \
    --epochs 5 \
    --batch-size 2 \
    --learning-rate 1e-4
```

### 3. Parametry adaptéru

| Parametr | Popis | Výchozí hodnota |
|----------|-------|-----------------|
| `--r` | LoRA rank (velikost adaptéru) | 8 |
| `--lora-alpha` | LoRA alpha (škálování) | 16 |
| `--lora-dropout` | Dropout pro regularizaci | 0.1 |
| `--max-length` | Maximální délka sekvence | 2048 |
| `--epochs` | Počet epoch trénování | 3 |
| `--batch-size` | Velikost batch | 4 |
| `--learning-rate` | Learning rate | 2e-4 |

## 🧪 Testování adaptéru

### 1. Interaktivní testování

```bash
# S konkrétním modelem
python test_adapter.py \
    --base-model microsoft/DialoGPT-medium \
    --adapter ./adapters/babis_adapter

# Bez specifikace modelu (použije se z konfigurace)
python test_adapter.py \
    --adapter ./adapters/babis_adapter
```

### 2. Testování kompatibility

```bash
# Testuje kompatibilitu s různými modely
python test_adapter.py \
    --adapter ./adapters/babis_adapter \
    --test-compatibility
```

### 3. Pokročilé testování

```bash
python test_adapter.py \
    --base-model gpt2 \
    --adapter ./adapters/babis_adapter \
    --device cuda \
    --max-length 1024 \
    --temperature 0.8
```

## 🔄 Použití adaptéru s různými modely

### Kompatibilní modely

Adaptér by měl být kompatibilní s těmito modely:
- `microsoft/DialoGPT-medium`
- `microsoft/DialoGPT-large`
- `gpt2`
- `gpt2-medium`
- `EleutherAI/gpt-neo-125M`
- `EleutherAI/gpt-neo-1.3B`

### Příklad použití v kódu

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Načtení základního modelu
base_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# Připojení adaptéru
model = PeftModel.from_pretrained(base_model, "./adapters/babis_adapter")

# Použití
response = model.generate(tokenizer.encode("Jak se máš?", return_tensors="pt"))
print(tokenizer.decode(response[0]))
```

## 📊 Srovnání přístupů

| Metoda | Velikost | Rychlost načítání | Kompatibilita | Kvalita |
|--------|----------|-------------------|---------------|---------|
| **Full Fine-tuning** | ~1-7 GB | Pomalá | Specifická | Vysoká |
| **LoRA** | ~10-50 MB | Rychlá | Široká | Vysoká |
| **QLoRA** | ~10-50 MB | Rychlá | Široká | Vysoká |
| **Prompt Tuning** | ~1-10 MB | Velmi rychlá | Široká | Střední |

## 🎛️ Optimalizace adaptéru

### 1. Pro lepší kvalitu
```bash
python create_qlora_adapter.py \
    --r 32 \
    --lora-alpha 64 \
    --epochs 5 \
    --learning-rate 1e-4
```

### 2. Pro menší velikost
```bash
python create_qlora_adapter.py \
    --r 4 \
    --lora-alpha 8 \
    --epochs 2 \
    --learning-rate 3e-4
```

### 3. Pro rychlejší trénování
```bash
python create_qlora_adapter.py \
    --r 8 \
    --batch-size 8 \
    --epochs 1 \
    --learning-rate 5e-4
```

## 🔧 Řešení problémů

### Chyba: "Target modules not found"
- Adaptér byl trénován na jiné architektuře modelu
- Zkuste jiný základní model nebo přetrénujte adaptér

### Chyba: "Out of memory"
- Snižte `--batch-size`
- Použijte `--device cpu`
- Snižte `--max-length`

### Chyba: "Model not compatible"
- Zkontrolujte, zda model podporuje LoRA
- Zkuste jiný základní model

## 📈 Monitorování trénování

Adaptér automaticky ukládá:
- ✅ **Konfiguraci** (`adapter_config.json`)
- ✅ **Váhy adaptéru** (v adresáři adaptéru)
- ✅ **Automatické ukládání** checkpointů
- ✅ **Logy trénování** v `/workspace/babis-finetuned/logs/`
- ✅ **Metriky trénování** (pokud je povoleno logging)

## 🎯 Doporučení

1. **Začněte s menším rankem** (`r=8`) a zvyšujte podle potřeby
2. **Testujte kompatibilitu** před použitím v produkci
3. **Ukládejte konfigurace** pro pozdější použití
4. **Experimentujte s různými modely** pro nejlepší výsledky

## 📚 Další zdroje

- [PEFT dokumentace](https://huggingface.co/docs/peft)
- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [QLoRA paper](https://arxiv.org/abs/2305.14314) 