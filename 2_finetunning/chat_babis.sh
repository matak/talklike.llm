#!/bin/bash

# Wrapper script pro chat s Babiš adaptérem
# Nastavuje cache do /workspace a spouští test_adapter.py

echo "🤖 Spouštím chat s Babiš adaptérem..."
echo "📁 Cache nastaven do /workspace"

# Nastavení cache do /workspace
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets

# Vytvoření cache adresářů
mkdir -p /workspace/.cache/huggingface/transformers
mkdir -p /workspace/.cache/huggingface/datasets

# Kontrola místa
echo "💾 Dostupné místo:"
df -h /workspace | tail -1

# Spuštění chatu
echo "🚀 Spouštím chat..."
python 2_finetunning/test_adapter.py \
    --base-model mistralai/Mistral-7B-Instruct-v0.3 \
    --adapter mcmatak/babis-mistral-adapter \
    --temperature 0.8 \
    --max-length 300 