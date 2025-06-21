# Fine-tuning Llama 3 8B pro Andreje Babiše

Tento projekt obsahuje kompletní pipeline pro fine-tuning Llama 3 8B modelu s daty Andreje Babiše na RunPod.io nebo lokálně.

## 📋 Přehled

- **Base Model**: Meta-Llama-3-8B-Instruct
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Data Format**: JSONL s chat formátem
- **Target**: Model mluvící stylem Andreje Babiše

## 🚀 Rychlý start

### 1. Příprava prostředí

```bash
# Klonování repozitáře
git clone <your-repo-url>
cd talklike.llm

# Instalace závislostí
pip install -r requirements.txt
```

### 2. Nastavení tokenů

Vytvořte soubor `.env` v kořenovém adresáři:

```bash
# Hugging Face token (povinné)
HF_TOKEN=hf_your_token_here

# Weights & Biases token (volitelné)
WANDB_API_KEY=your_wandb_token_here
```

### 3. Spuštění fine-tuningu

```bash
# Rychlý start (doporučeno)
chmod +x quick_start.sh
./quick_start.sh

# Nebo manuálně
python finetune_babis_llama.py --use_wandb --push_to_hub
```

## 📊 Struktura dat

Vaše data v `../data/all.jsonl` mají formát:

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

## 🏗️ Architektura

### LoRA Konfigurace
- **Rank (r)**: 16
- **Alpha**: 32
- **Dropout**: 0.1
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Training Parametry
- **Epochs**: 3
- **Batch Size**: 2 (per device)
- **Gradient Accumulation**: 4
- **Learning Rate**: 2e-4
- **Warmup Steps**: 100
- **Max Length**: 2048 tokens

## 💻 Spuštění na RunPod.io

### 1. Vytvoření podu

1. Jděte na [runpod.io](https://runpod.io)
2. Vytvořte nový pod s:
   - **GPU**: RTX 4090 nebo A100
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
cd <your-project>

# Vytvoření .env souboru
nano .env
# Přidejte vaše tokeny

# Spuštění fine-tuningu
./quick_start.sh
```

### 3. Monitorování

- **W&B Dashboard**: Sledujte metriky na wandb.ai
- **Jupyter**: Sledujte progress v notebooku
- **Terminal**: Logy v terminálu

## 📁 Soubory projektu

```
talklike.llm/
├── 2_finetunning/                 # Fine-tuning scripts and configs
│   ├── finetune_babis.py          # Main fine-tuning script
│   ├── test_tokenization.py       # Tokenization testing
│   ├── run_finetune.sh            # Fine-tuning shell script
│   ├── run_mistral_finetune.sh    # Mistral fine-tuning script
│   ├── requirements_finetunning.txt # Python dependencies
│   ├── README_FINETUNE.md         # This file
│   └── RUNPOD_SETUP.md           # RunPod.io instructions
├── data/
│   └── all.jsonl                 # Training data
└── [other directories...]
```

## ⚙️ Konfigurace

### Základní parametry

```bash
python finetune_babis_llama.py \
    --data_path ../data/all.jsonl \
    --output_dir ./babis-llama-finetuned \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --max_length 2048
```

### Pokročilé parametry

```bash
python finetune_babis_llama.py \
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

## 🔧 Troubleshooting

### Out of Memory (OOM)
```bash
# Snižte batch size a max_length
python finetune_babis_llama.py --batch_size 1 --max_length 1024
```

### Pomalé trénování
```bash
# Zkontrolujte GPU využití
nvidia-smi

# Snižte gradient accumulation
# Upravte v kódu: gradient_accumulation_steps=2
```

### Model nekonverguje
```bash
# Snižte learning rate
python finetune_babis_llama.py --learning_rate 1e-4

# Zvyšte počet epoch
python finetune_babis_llama.py --epochs 5
```

## 📈 Očekávané výsledky

Po fine-tuningu by model měl:

- ✅ Mluvit stylem Andreje Babiše
- ✅ Používat jeho charakteristické fráze
- ✅ Odpovídat v první osobě
- ✅ Přidávat podpis "Andrej Babiš."

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

## 💾 Uložení a sdílení

### Lokální uložení
```bash
# Model se uloží do ./babis-llama-finetuned-final/
```

### Hugging Face Hub
```bash
# Model se automaticky nahraje na HF Hub
# Repo: https://huggingface.co/your-username/babis-llama-3-8b-lora
```

## 🧪 Použití fine-tuned modelu

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Načtení base modelu
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Načtení LoRA adaptérů
model = PeftModel.from_pretrained(base_model, "your-username/babis-llama-3-8b-lora")

# Generování
prompt = "Pane Babiši, jak hodnotíte inflaci?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 💰 Odhadované náklady

- **RTX 4090**: ~$0.60/hod
- **A100**: ~$1.20/hod
- **Očekávaná doba trénování**: 2-4 hodiny
- **Celkové náklady**: $1.20 - $4.80

## 📝 Poznámky

1. **Data kvalita**: Ujistěte se, že vaše data jsou kvalitní a konzistentní
2. **Monitoring**: Sledujte loss během trénování
3. **Backup**: Pravidelně ukládejte checkpointy
4. **Testování**: Testujte model během trénování

## 🤝 Kontakt

Pro problémy nebo otázky:
- GitHub Issues
- RunPod Discord
- Hugging Face Forums

## 📄 Licence

Tento projekt je určen pouze pro vzdělávací účely. Respektujte autorská práva a licenční podmínky použitých modelů a dat. 