#!/bin/bash

# Kontrola existence .env souboru
if [ ! -f ".env" ]; then
    echo "❌ Soubor .env nebyl nalezen!"
    echo "📝 Vytvořte soubor .env s následujícím obsahem:"
    echo "HF_TOKEN=hf_your_token_here"
    echo "WANDB_API_KEY=your_wandb_token_here"
    exit 1
fi

# Skript pro spuštění fine-tuningu s Mistralem a agresivním vyčištěním
# Optimalizováno pro překonání problémů s místem na disku

echo "🚀 Spouštím fine-tuning s Mistralem a agresivním vyčištěním"
echo "=================================================="

# Kontrola dostupnosti dat
if [ ! -f "data/all.jsonl" ]; then
    echo "❌ Soubor data/all.jsonl nebyl nalezen!"
    echo "💡 Zkontrolujte, zda máte data v adresáři data/"
    exit 1
fi

# Instalace závislostí
if [ -f "requirements.txt" ]; then
    echo "📦 Instaluji závislosti..."
    pip install -r requirements.txt
fi

# Kontrola místa na disku
echo "💾 Kontroluji místo na disku..."
echo "Root filesystem:"
df -h /
echo ""
echo "Network storage:"
df -h /workspace

# Agresivní vyčištění
echo "🧹 Agresivní vyčištění pro Mistral..."
rm -rf ~/.cache/huggingface
rm -rf /tmp/*
rm -rf /var/tmp/*
rm -rf /root/.cache
rm -rf /root/.local
rm -rf /root/.config
rm -rf /usr/local/lib/python3.10/dist-packages/transformers/.cache
rm -rf /usr/local/lib/python3.10/dist-packages/huggingface_hub/.cache
rm -rf /usr/local/lib/python3.10/dist-packages/datasets/.cache

# Vyčištění log souborů
find /var/log -name '*.log' -delete 2>/dev/null || true
find /var/log -name '*.gz' -delete 2>/dev/null || true

# Vyčištění pip a conda cache
pip cache purge 2>/dev/null || true
conda clean -a -y 2>/dev/null || true

# Synchronizace filesystem
sync

# Vytvoření cache adresářů na network storage
mkdir -p /workspace/.cache/huggingface/transformers
mkdir -p /workspace/.cache/huggingface/datasets
mkdir -p /workspace/.cache/huggingface/hub

echo "✅ Agresivní vyčištění dokončeno"

# Kontrola místa po vyčištění
echo "💾 Místo po vyčištění:"
df -h /

# Spuštění fine-tuningu s Mistralem
echo "🤖 Spouštím fine-tuning s Mistral-7B-Instruct-v0.3..."
python finetune_babis.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --output_dir /workspace/babis-mistral-finetuned \
    --epochs 2 \
    --batch_size 1 \
    --max_length 2048 \
    --aggressive_cleanup

echo "✅ Fine-tuning dokončen!"
echo "📁 Model je uložen v: /workspace/babis-mistral-finetuned-final"
echo "💾 Všechny soubory jsou na network storage" 