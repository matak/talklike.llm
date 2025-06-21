# Fine-tuning Llama 3 8B pro Andreje Babiše na RunPod.io

> **📚 Navigace:** [🏠 Hlavní projekt](../README.md) | [📊 Příprava dat](../1_data_preparation/README.md) | [🏋️ Detailní dokumentace](README.md) | [📈 Benchmarking](../3_benchmarking/README.md)

## 📋 Přehled
Tento návod vás provede procesem fine-tuningu Llama 3 8B modelu s daty Andreje Babiše na RunPod.io platformě.

## Struktura dat
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

## Kroky pro spuštění na RunPod.io

### 1. Vytvoření podu na RunPod.io

1. Jděte na [runpod.io](https://runpod.io)
2. Vytvořte nový pod s následujícími specifikacemi:
   - **GPU**: RTX 4090 nebo A100 (doporučeno pro 8B model)
   - **RAM**: Minimálně 24GB
   - **Storage**: Minimálně 50GB
   - **Template**: PyTorch nebo Jupyter

### 2. Příprava prostředí

Po připojení k podu spusťte:

```bash
# Aktualizace systému
sudo apt update && sudo apt upgrade -y

# Instalace potřebných balíčků
sudo apt install -y git wget curl

# Klonování repozitáře (pokud používáte git)
git clone <váš-repo-url>
cd <váš-projekt>
```

### 3. Vytvoření .env souboru

Vytvořte soubor `.env` v kořenovém adresáři projektu:

```bash
# Hugging Face token (získáte na huggingface.co/settings/tokens)
HF_TOKEN=hf_your_token_here

# Weights & Biases token (volitelné, pro sledování trénování)
WANDB_API_KEY=your_wandb_token_here
```

### 4. Spuštění fine-tuningu

```bash
# Spuštění Jupyter notebooku
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Nebo spuštění Python skriptu
python finetune_babis_llama.py
```

### 5. Monitorování trénování

- **W&B Dashboard**: Sledujte metriky na wandb.ai
- **Jupyter**: Sledujte progress v notebooku
- **Terminal**: Logy v terminálu

## Konfigurace modelu

### LoRA parametry
- **Rank (r)**: 16
- **Alpha**: 32  
- **Dropout**: 0.1
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Training parametry
- **Epochs**: 3
- **Batch size**: 2 (per device)
- **Gradient accumulation**: 4
- **Learning rate**: 2e-4
- **Warmup steps**: 100
- **Max length**: 2048 tokens

## Očekávané výsledky

Po fine-tuningu by model měl:
- Mluvit stylem Andreje Babiše
- Používat jeho charakteristické fráze
- Odpovídat v první osobě
- Přidávat podpis "Andrej Babiš."

## Troubleshooting

### Problém: Out of Memory (OOM)
**Řešení:**
- Snižte batch size na 1
- Snižte max_length na 1024
- Použijte gradient checkpointing

### Problém: Pomalé trénování
**Řešení:**
- Zkontrolujte GPU využití
- Snižte gradient accumulation steps
- Použijte mixed precision (fp16)

### Problém: Model nekonverguje
**Řešení:**
- Snižte learning rate na 1e-4
- Zvyšte počet epoch
- Zkontrolujte kvalitu dat

## Uložení a sdílení modelu

### Lokální uložení
```bash
# Model se uloží do ./babis-llama-finetuned-final/
```

### Hugging Face Hub
```bash
# Model se automaticky nahraje na HF Hub
# Repo: https://huggingface.co/your-username/babis-llama-3-8b-lora
```

## Použití fine-tuned modelu

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

## Odhadované náklady

- **RTX 4090**: ~$0.60/hod
- **A100**: ~$1.20/hod
- **Očekávaná doba trénování**: 2-4 hodiny
- **Celkové náklady**: $1.20 - $4.80

## Poznámky

1. **Data kvalita**: Ujistěte se, že vaše data jsou kvalitní a konzistentní
2. **Monitoring**: Sledujte loss během trénování
3. **Backup**: Pravidelně ukládejte checkpointy
4. **Testování**: Testujte model během trénování

## Kontakt

Pro problémy nebo otázky:
- GitHub Issues
- RunPod Discord
- Hugging Face Forums 