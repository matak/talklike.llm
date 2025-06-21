# 🏋️ Fine-tuning Jazykového Modelu - TalkLike.LLM

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

# Nastavení tokenů
echo "HF_TOKEN=hf_your_token_here" > .env
echo "WANDB_API_KEY=your_wandb_token_here" >> .env
```

### 2. Spuštění fine-tuningu
```bash
# Rychlý start (doporučeno)
chmod +x run_finetune.sh
./run_finetune.sh

# Nebo manuálně
python finetune_babis.py --use_wandb --push_to_hub
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
- **Styl**: Mluvená čeština s "babíšovštinou"
- **Podpis**: Každá odpověď končí "Andrej Babiš"
- **Jazykové chyby**: 15% pravděpodobnost slovenských odchylek
- **Témata**: Politika, ekonomika, rodina, podnikání

---

## 💻 Spuštění na RunPod.io

### 1. Vytvoření podu
1. Jděte na [runpod.io](https://runpod.io)
2. Vytvořte nový pod s následujícími specifikacemi:
   - **GPU**: RTX 4090 nebo A100 (doporučeno)
   - **RAM**: Minimálně 24GB
   - **Storage**: Minimálně 50GB
   - **Template**: PyTorch nebo Jupyter

### 2. Příprava prostředí
```bash
# Aktualizace systému
sudo apt update && sudo apt upgrade -y

# Instalace balíčků
sudo apt install -y git wget curl

# Klonování repozitáře
git clone <your-repo-url>
cd talklike.llm

# Vytvoření .env souboru
nano .env
# Přidejte vaše tokeny:
# HF_TOKEN=hf_your_token_here
# WANDB_API_KEY=your_wandb_token_here

# Spuštění fine-tuningu
./run_finetune.sh
```

### 3. Monitorování trénování
- **W&B Dashboard**: Sledujte metriky na wandb.ai
- **Jupyter**: Sledujte progress v notebooku
- **Terminal**: Logy v terminálu
- **GPU Monitoring**: `nvidia-smi -l 1`

---

## 📁 Struktura projektu

```
2_finetunning/
├── 📄 Hlavní skripty
│   ├── finetune_babis.py          # Main fine-tuning script
│   ├── test_tokenization.py       # Tokenization testing
│   └── run_finetune.sh            # Fine-tuning shell script
├── 📄 Alternativní skripty
│   └── run_mistral_finetune.sh    # Mistral fine-tuning script
├── 📄 Konfigurace
│   ├── requirements_finetunning.txt # Python dependencies
│   └── README_FINETUNE.md         # This file
├── 📄 Dokumentace
│   └── RUNPOD_SETUP.md           # RunPod.io instructions
└── 📄 Výstupy
    └── babis-llama-finetuned/    # Fine-tuned model
```

---

## ⚙️ Konfigurace

### Základní parametry
```bash
python finetune_babis.py \
    --data_path ../data/all.jsonl \
    --output_dir ./babis-llama-finetuned \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --max_length 2048
```

### Pokročilé parametry
```bash
python finetune_babis.py \
    --data_path ../data/all.jsonl \
    --output_dir ./babis-llama-finetuned \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --epochs 5 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --max_length 1024 \
    --use_wandb \
    --push_to_hub \
    --hub_model_id your-username/babis-llama-3-8b-lora
```

### Parametry pro různé GPU
```bash
# RTX 4090 (24GB VRAM)
python finetune_babis.py --batch_size 2 --max_length 2048

# RTX 3090 (24GB VRAM)
python finetune_babis.py --batch_size 1 --max_length 1024

# A100 (40GB VRAM)
python finetune_babis.py --batch_size 4 --max_length 2048
```

---

## 🔧 Troubleshooting

### Out of Memory (OOM)
```bash
# Snižte batch size a max_length
python finetune_babis.py --batch_size 1 --max_length 1024

# Nebo snižte gradient accumulation
# Upravte v kódu: gradient_accumulation_steps=2
```

### Pomalé trénování
```bash
# Zkontrolujte GPU využití
nvidia-smi

# Optimalizujte dataloader
# Upravte num_workers v DataLoader
```

### Model nekonverguje
```bash
# Snižte learning rate
python finetune_babis.py --learning_rate 1e-4

# Zvyšte počet epoch
python finetune_babis.py --epochs 5

# Zkontrolujte kvalitu dat
python test_tokenization.py
```

### Chyby s tokeny
```bash
# Ověřte HF token
huggingface-cli whoami

# Ověřte W&B token
wandb login
```

---

## 📈 Očekávané výsledky

### Metriky výkonu
Po fine-tuningu by model měl dosáhnout:
- **Training Loss**: < 1.0 po 3 epochách
- **Validation Loss**: < 1.2
- **Perplexity**: < 2.0
- **Stylová konzistence**: > 85%

### Kvalitativní evaluace
Model by měl:
- ✅ Mluvit stylem Andreje Babiše
- ✅ Používat charakteristické fráze
- ✅ Odpovídat v první osobě
- ✅ Přidávat podpis "Andrej Babiš"
- ✅ Obsahovat slovenské odchylky

### Testovací prompty
```python
test_prompts = [
    "Pane Babiši, jak hodnotíte současnou inflaci?",
    "Co si myslíte o opozici?",
    "Jak se vám daří v Bruselu?",
    "Můžete vysvětlit vaši roli v té chemičce?",
    "Jak hodnotíte efektivizaci státní správy?"
]
```

---

## 💾 Uložení a sdílení

### Lokální uložení
```bash
# Model se uloží do ./babis-llama-finetuned-final/
# Obsahuje LoRA adaptéry a konfiguraci
```

### Hugging Face Hub
```bash
# Model se automaticky nahraje na HF Hub
# Repo: https://huggingface.co/your-username/babis-llama-3-8b-lora
# Obsahuje: LoRA adaptéry, konfiguraci, README
```

### Struktura uloženého modelu
```
babis-llama-finetuned-final/
├── adapter_config.json          # LoRA konfigurace
├── adapter_model.bin            # LoRA váhy
├── training_args.bin            # Training argumenty
├── config.json                  # Model konfigurace
└── README.md                    # Model dokumentace
```

---

## 📤 Manuální nahrání na Hugging Face Hub

### Kdy použít
- Zapomněli jste `--push_to_hub` při fine-tuningu
- Chcete nahrát již existující model
- Potřebujete změnit název modelu

### Rychlé nahrání
```bash
# 1. Nastavení tokenu
export HF_TOKEN=hf_your_token_here

# 2. Nahrání modelu
python upload_to_hf.py \
    --model_path /workspace/babis-finetuned-final \
    --hub_model_id your-username/babis-model

# 3. Kontrola bez nahrávání
python upload_to_hf.py \
    --model_path /workspace/babis-finetuned-final \
    --hub_model_id your-username/babis-model \
    --check_only
```

### Běžné cesty k modelu
- `/workspace/babis-finetuned-final`
- `/workspace/babis-mistral-finetuned-final`
- `./babis-llama-finetuned-final`

### Výstup
- ✅ Model dostupný na: `https://huggingface.co/your-username/babis-model`
- 📋 Instrukce pro použití modelu

---

## 🧪 Použití fine-tuned modelu

### Načtení a generování
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Načtení base modelu
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct"
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct"
)

# Načtení LoRA adaptérů
model = PeftModel.from_pretrained(
    base_model, 
    "your-username/babis-llama-3-8b-lora"
)

# Generování
prompt = "Pane Babiši, jak hodnotíte inflaci?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs, 
    max_length=200,
    temperature=0.7,
    do_sample=True
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Streamované generování
```python
from transformers import TextIteratorStreamer
import threading

streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
generation_kwargs = dict(
    inputs=inputs,
    streamer=streamer,
    max_length=200,
    temperature=0.7,
    do_sample=True
)

thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

for text in streamer:
    print(text, end="", flush=True)
```
