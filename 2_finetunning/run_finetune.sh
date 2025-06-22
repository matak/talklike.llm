#!/bin/bash

# Fine-tuning script pro Andreje Babiše
# Spustitelný na RunPod.io nebo lokálně

set -e

echo "🚀 Spouštím fine-tuning pro Andreje Babiše..."

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
    --output_dir /workspace/babis-finetuned \
    --model_name microsoft/DialoGPT-medium \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --max_length 1024 \
    --push_to_hub \
    --hub_model_id babis-lora

echo "✅ Fine-tuning dokončen!" 