#!/bin/bash

# Rychlý start skript pro fine-tuning Llama 3 8B pro Andreje Babiše na RunPod.io
# Použití: ./quick_start.sh

echo "🚀 Rychlý start fine-tuningu Llama 3 8B pro Andreje Babiše"
echo "=========================================================="

# Kontrola existence .env souboru
if [ ! -f ".env" ]; then
    echo "❌ Soubor .env nebyl nalezen!"
    echo "📝 Vytvořte soubor .env s následujícím obsahem:"
    echo "HF_TOKEN=hf_your_token_here"
    echo "WANDB_API_KEY=your_wandb_token_here"
    exit 1
fi

# Kontrola existence dat
if [ ! -f "data/all.jsonl" ]; then
    echo "❌ Soubor data/all.jsonl nebyl nalezen!"
    echo "📁 Ujistěte se, že máte data v adresáři data/"
    exit 1
fi

# Instalace závislostí
echo "📦 Instaluji závislosti..."
pip install -r requirements.txt

# Spuštění fine-tuningu
echo "🏋️ Spouštím fine-tuning..."
python finetune_babis_llama.py \
    --data_path data/all.jsonl \
    --output_dir ./babis-llama-finetuned \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --max_length 2048 \
    --use_wandb \
    --push_to_hub \
    --hub_model_id babis-llama-3-8b-lora

echo "✅ Fine-tuning dokončen!"
echo "📁 Model je uložen v: ./babis-llama-finetuned-final"
echo "🌐 Model je dostupný na: https://huggingface.co/babis-llama-3-8b-lora" 