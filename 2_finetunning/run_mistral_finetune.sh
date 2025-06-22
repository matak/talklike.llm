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

# Nastavení environment proměnných
export HF_TOKEN="your_hf_token_here"
echo "HF_TOKEN=your_hf_token_here"

# Spuštění fine-tuning
cd 2_finetunning

python finetune.py \
    --data_path ../data/all.jsonl \
    --output_dir /workspace/mistral-babis-finetuned \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --epochs 3 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --max_length 2048 \
    --aggressive_cleanup \
    --push_to_hub \
    --hub_model_id mistral-babis-lora

echo "✅ Fine-tuning dokončen!" 