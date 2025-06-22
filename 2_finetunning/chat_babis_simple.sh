#!/bin/bash

# Jednoduchý chat s Babiš modelem
echo "🎭 Spouštím chat s Andrejem Babišem..."
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
echo "🚀 Spouštím chat s Babiš modelem..."
python 2_finetunning/chat_with_babis.py 