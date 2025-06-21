#!/bin/bash

# Kontrola existence .env souboru
if [ ! -f ".env" ]; then
    echo "❌ Soubor .env nebyl nalezen!"
    echo "📝 Vytvořte soubor .env s následujícím obsahem:"
    echo "HF_TOKEN=hf_your_token_here"
    echo "WANDB_API_KEY=your_wandb_token_here"
    exit 1
fi

# Skript pro spuštění fine-tuningu s optimalizací pro network storage
# Používá menší model a ukládá vše na /workspace

echo "🚀 Spouštím fine-tuning pro Andreje Babiše s optimalizací pro network storage"
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
df -h /workspace

# Vyčištění cache
echo "🧹 Čistím cache..."
rm -rf ~/.cache/huggingface
rm -rf /tmp/*
rm -rf /root/.cache

# Vytvoření cache adresářů na network storage
mkdir -p /workspace/.cache/huggingface/transformers
mkdir -p /workspace/.cache/huggingface/datasets

# Spuštění fine-tuningu s menším modelem
echo "🤖 Spouštím fine-tuning s DialoGPT-medium..."
python finetune_babis.py \
    --model_name microsoft/DialoGPT-medium \
    --output_dir /workspace/babis-finetuned \
    --epochs 3 \
    --batch_size 2 \
    --max_length 1024 \
    --cleanup_cache

echo "✅ Fine-tuning dokončen!"
echo "📁 Model je uložen v: /workspace/babis-finetuned-final"
echo "💾 Všechny soubory jsou na network storage" 