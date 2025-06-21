#!/bin/bash

# Kontrola existence .env souboru
if [ ! -f ".env" ]; then
    echo "❌ Soubor .env nebyl nalezen!"
    echo "📝 Vytvořte soubor .env s následujícím obsahem:"
    echo "HF_TOKEN=hf_your_token_here"
    echo "WANDB_API_KEY=your_wandb_token_here"
    exit 1
fi

# Kontrola dostupnosti dat
if [ ! -f "../data/all.jsonl" ]; then
    echo "❌ Soubor ../data/all.jsonl nebyl nalezen!"
    echo "💡 Zkontrolujte, zda máte data v adresáři data/"
    exit 1
fi

# Instalace závislostí
if [ -f "requirements_finetunning.txt" ]; then
    echo "📦 Instaluji závislosti..."
    pip install -r requirements_finetunning.txt
fi

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