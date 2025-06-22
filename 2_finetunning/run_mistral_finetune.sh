#!/bin/bash

# Fine-tuning script pro Mistral model s daty Andreje Babiše
# Spustitelný na RunPod.io

set -e

echo "🚀 Spouštím fine-tuning pro Mistral model..."

# Kontrola, že jsme v root directory projektu
if [ ! -d "lib" ] || [ ! -d "data" ]; then
    echo "❌ Skript musí být spuštěn z root directory projektu!"
    echo "💡 Spusťte skript z adresáře, kde jsou složky 'lib' a 'data'"
    exit 1
fi

# Spuštění fine-tuning z root directory
python 2_finetunning/finetune.py \
    --data_path data/all.jsonl \
    --output_dir /workspace/mistral-babis-finetuned \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --max_length 1024 \
    --aggressive_cleanup \
    --push_to_hub \
    --hub_model_id mistral-babis-lora

echo "✅ Fine-tuning dokončen!" 