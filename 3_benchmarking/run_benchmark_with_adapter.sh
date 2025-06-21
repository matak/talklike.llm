#!/bin/bash

# Benchmarking script pro TalkLike.LLM s adaptérem
# Spouští kompletní benchmarking pipeline s vaším natrénovaným adaptérem

echo "🚀 Spouštím benchmarking TalkLike.LLM s adaptérem..."
echo "📁 Cache nastaven do /workspace"

# Nastavení cache do /workspace (stejné jako v chat_babis.sh)
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets

# Vytvoření cache adresářů
mkdir -p /workspace/.cache/huggingface/transformers
mkdir -p /workspace/.cache/huggingface/datasets

# Kontrola místa
echo "💾 Dostupné místo:"
df -h /workspace | tail -1

# Instalace requirements
echo "📦 Instaluji requirements..."
pip install -r requirements_benchmarking.txt

# Spuštění benchmarkingu
echo "🔬 Spouštím benchmarking pipeline..."
python run_benchmark.py

echo "✅ Benchmarking dokončen!"
echo "📊 Výsledky najdete v adresáři results/"
echo "📋 Reporty v: results/reports/"
echo "📈 Vizualizace v: results/visualizations/" 