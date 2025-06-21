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