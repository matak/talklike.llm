# RunPod.io Setup Guide

Tento návod vás provede nastavením fine-tuning prostředí na RunPod.io.

## 🚀 Rychlý start

### 1. Vytvoření podu

1. Jděte na [runpod.io](https://runpod.io)
2. Klikněte na "Deploy"
3. Vyberte template: `PyTorch 2.1.1`
4. Nastavte specifikace:
   - **GPU**: RTX 4090 nebo A100 (doporučeno)
   - **RAM**: Minimálně 24GB
   - **Storage**: Minimálně 50GB
   - **Network Volume**: Doporučeno pro persistentní data

### 2. Připojení k podu

```bash
# SSH připojení
ssh root@your-pod-ip

# Nebo použijte webový terminál v RunPod UI
```

### 3. Instalace závislostí

```bash
# Aktualizace systému
apt update && apt upgrade -y

# Instalace základních balíčků
apt install -y git wget curl htop

# Instalace Python závislostí
pip install -r requirements_finetunning.txt
```

### 4. Nastavení environment proměnných

```bash
# Vytvoření .env souboru
cat > .env << EOF
HF_TOKEN=your_hf_token_here
EOF

# Nebo export proměnných
export HF_TOKEN=your_hf_token_here
```

### 5. Spuštění fine-tuning

```bash
# Klonování repozitáře
git clone https://github.com/your-repo/talklike.llm.git
cd talklike.llm

# Spuštění fine-tuning
bash 2_finetunning/run_finetune.sh
```

## 📊 Monitoring

### GPU Monitoring
```bash
# Sledování GPU využití
nvidia-smi -l 1

# Detailní informace
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Disk Monitoring
```bash
# Sledování místa na disku
df -h

# Sledování místa v reálném čase
watch -n 5 df -h
```

### Process Monitoring
```bash
# Sledování procesů
htop

# Sledování Python procesů
ps aux | grep python
```

## 🔧 Optimalizace

### Pro RTX 4090 (24GB VRAM)
```bash
python finetune.py \
    --model_name microsoft/DialoGPT-medium \
    --batch_size 2 \
    --max_length 1024
```

### Pro A100 (40GB VRAM)
```bash
python finetune.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --batch_size 4 \
    --max_length 2048
```

### Pro menší GPU
```bash
python finetune.py \
    --model_name microsoft/DialoGPT-small \
    --batch_size 1 \
    --max_length 512
```

## 🐛 Řešení problémů

### Out of Memory (OOM)
```bash
# Snižte batch size
python finetune.py --batch_size 1

# Snižte max_length
python finetune.py --max_length 512

# Použijte menší model
python finetune.py --model_name microsoft/DialoGPT-small
```

### Nedostatek místa na disku
```bash
# Vyčištění cache
rm -rf /root/.cache/huggingface
rm -rf /tmp/*

# Nebo použijte agresivní vyčištění
python finetune.py --aggressive_cleanup
```

### Pomalé stahování modelu
```bash
# Použijte mirror
export HF_ENDPOINT=https://hf-mirror.com

# Nebo stáhněte model předem
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/DialoGPT-medium')"
```

## 💾 Persistence dat

### Network Volume
```bash
# Mount network volume
mkdir -p /workspace
mount /dev/sdb1 /workspace

# Uložení modelu na network volume
python finetune.py --output_dir /workspace/babis-finetuned
```

### Backup
```bash
# Zálohování modelu
tar -czf babis-model-backup.tar.gz /workspace/babis-finetuned

# Stažení zálohy
scp root@your-pod-ip:babis-model-backup.tar.gz ./
```

## 📈 Výkonnostní tipy

### Optimalizace dataloader
```python
# V finetune.py
training_args = TrainingArguments(
    dataloader_num_workers=4,  # Zvýšit podle CPU
    dataloader_pin_memory=True,
    dataloader_drop_last=True,
)
```

### Gradient checkpointing
```python
# Pro úsporu paměti
training_args = TrainingArguments(
    gradient_checkpointing=True,
    gradient_accumulation_steps=4,
)
```

### Mixed precision
```python
# Pro rychlejší trénování
training_args = TrainingArguments(
    fp16=True,  # Pro NVIDIA GPU
    # bf16=True,  # Pro A100
)
```

## 🔗 Užitečné odkazy

- [RunPod.io](https://runpod.io/) - GPU hosting
- [Hugging Face](https://huggingface.co/) - Modely a tokeny
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [PEFT](https://huggingface.co/docs/peft) - Parameter efficient fine-tuning 