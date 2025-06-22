#!/bin/bash

# Chat s lokálním fine-tunovaným modelem
echo "🎭 Spouštím chat s lokálním fine-tunovaným modelem..."
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

# Kontrola, zda existuje lokální model
echo "🔍 Kontroluji dostupnost lokálního modelu..."
if [ -d "/workspace/mistral-babis-finetuned" ]; then
    echo "✅ Nalezen model v /workspace/mistral-babis-finetuned"
elif [ -d "/workspace/finetuned-model" ]; then
    echo "✅ Nalezen model v /workspace/finetuned-model"
elif [ -d "/workspace/model" ]; then
    echo "✅ Nalezen model v /workspace/model"
else
    echo "⚠️ Lokální model nebyl nalezen v /workspace/"
    echo "💡 Spusťte nejdříve fine-tuning: python finetune.py"
fi

# Spuštění chatu
echo "🚀 Spouštím chat s lokálním modelem..."
python 2_finetunning/chat_local_model.py 