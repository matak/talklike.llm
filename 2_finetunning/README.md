# 🏋️ Fine-tuning Jazykového Modelu - TalkLike.LLM

> **📚 Navigace:** [🏠 Hlavní projekt](../README.md) | [📊 Příprava dat](../1_data_preparation/README.md) | [📈 Benchmarking](../3_benchmarking/README.md)

## 📋 Přehled

Tento projekt implementuje **fine-tuning jazykového modelu** pomocí LoRA (Low-Rank Adaptation) techniky pro napodobení komunikačního stylu Andreje Babiše. Fine-tuning je optimalizován pro efektivní trénování na RunPod.io nebo lokálních GPU.

### 🎯 Cíl
Vytvořit fine-tuned model, který:
- ✅ Mluví autentickým stylem Andreje Babiše
- ✅ Používá charakteristické fráze a rétorické prvky
- ✅ Generuje konzistentní odpovědi v první osobě
- ✅ Zachovává "babíšovštinu" s jazykovými odchylkami

---

## 🏗️ Architektura řešení

### Základní konfigurace
- **Base Model**: Meta-Llama-3-8B-Instruct
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Data Format**: JSONL s chat formátem
- **Target**: Stylová adaptace pro Andreje Babiše

### LoRA Konfigurace
```python
lora_config = {
    "r": 16,                    # Rank
    "alpha": 32,                # Scaling factor
    "dropout": 0.1,             # Dropout rate
    "target_modules": [         # Target layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
}
```

### Training Parametry
```python
training_config = {
    "epochs": 3,
    "batch_size": 2,            # per device
    "gradient_accumulation": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "max_length": 2048,
    "save_steps": 500,
    "eval_steps": 500
}
```

---

## 🚀 Rychlé spuštění

### 1. Příprava prostředí
```bash
# Instalace závislostí
pip install -r requirements_finetunning.txt

# Nastavení environment proměnných
echo "HF_TOKEN=your_hf_token_here" >> .env
```

### 2. Spuštění fine-tuning

```bash
# Základní fine-tuning
python finetune.py --push_to_hub

# Fine-tuning s vlastními parametry
python finetune.py \
    --model_name microsoft/DialoGPT-medium \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --push_to_hub \
    --hub_model_id babis-lora

# Fine-tuning s vlastními parametry
python 2_finetunning/finetune.py \
    --data_path data/all.jsonl \
    --output_dir /workspace/mistral-babis-finetuned \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --epochs 3 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --max_length 2048 \
    --aggressive_cleanup \
    --push_to_hub \
    --hub_model_id mistral-babis-lora
```

### 3. Testování tokenizace
```bash
# Ověření správné tokenizace dat
python test_tokenization.py
```

---

## 📊 Struktura dat

### Vstupní formát
Dataset v `../data/all.jsonl` má strukturu:

```json
{
    "messages": [
        {
            "role": "system",
            "content": "Jsi Andrej Babiš, český politik a podnikatel..."
        },
        {
            "role": "user",
            "content": "Pane Babiši, můžete vysvětlit vaši roli v té chemičce?"
        },
        {
            "role": "assistant",
            "content": "Hele, ta továrna? To už jsem dávno předal..."
        }
    ]
}
```

### Klíčové vlastnosti dat
- **Počet QA párů**: 1,500
- **Styl**: Autentický Babišův styl
- **Jazykové chyby**: 15% pravděpodobnost slovenských odchylek
- **Témata**: Politika, ekonomika, rodina, podnikání

---

## 💻 Spuštění na RunPod.io

### 1. Vytvoření kontejneru
- Image: `runpod/pytorch:2.1.1-py3.10-cuda12.1.0`
- GPU: RTX 4090 nebo A100
- Disk: 50GB+

### 2. Nastavení environment proměnných
```bash
# V kontejneru
export HF_TOKEN=your_hf_token_here
```

### 3. Spuštění
```bash
# Klonování repozitáře
git clone https://github.com/your-repo/talklike.llm.git
cd talklike.llm

# Spuštění fine-tuning
bash 2_finetunning/run_finetune.sh
```

## 📊 Monitoring

Fine-tuning automaticky:
- ✅ **Loguje metriky** do `/workspace/babis-finetuned/logs/`
- ✅ **Ukládá checkpointy** každých 500 kroků
- ✅ **Načítá nejlepší model** na konci trénování
- ✅ **Exportuje model** na Hugging Face Hub (pokud povoleno)

## 🔧 Pokročilé nastavení

### Optimalizace pro velké modely
```bash
python finetune.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --aggressive_cleanup \
    --batch_size 1 \
    --max_length 2048
```

### Vlastní LoRA konfigurace
Upravte `finetune.py`:
```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,  # Zvýšit pro lepší kvalitu
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=target_modules
)
```

## 🐛 Řešení problémů

### Nedostatek místa na disku
```bash
# Automatické vyčištění
python finetune.py --aggressive_cleanup

# Manuální vyčištění
rm -rf /root/.cache/huggingface
rm -rf /tmp/*
```

### Chyby při načítání modelu
```bash
# Použít menší model
python finetune.py --model_name microsoft/DialoGPT-medium

# Restartovat kontejner
# Zvýšit velikost root filesystem
```

### Problémy s tokenizerem
```bash
# Kontrola kompatibility
python test_tokenization.py

# Použít jiný model
python finetune.py --model_name microsoft/DialoGPT-large
```

## 📁 Struktura výstupu

```
/workspace/babis-finetuned/
├── checkpoint-500/          # Checkpointy
├── checkpoint-1000/
├── logs/                    # Logy trénování
├── pytorch_model.bin        # Finální model
├── config.json             # Konfigurace
├── tokenizer.json          # Tokenizer
└── adapter_config.json     # LoRA konfigurace
```

## 🔗 Užitečné odkazy

- [Hugging Face Hub](https://huggingface.co/) - Nahrávání modelů
- [RunPod.io](https://runpod.io/) - GPU hosting
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Teorie LoRA
- [PEFT Dokumentace](https://huggingface.co/docs/peft) - Fine-tuning knihovna
