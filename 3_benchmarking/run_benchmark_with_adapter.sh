#!/bin/bash
# -*- coding: utf-8 -*-
# Spuštění benchmarkingu s adaptérem pro TalkLike.LLM
# Automaticky nastaví cache a spustí kompletní benchmarking

set -e  # Zastaví skript při chybě

echo "🚀 SPUŠTĚNÍ BENCHMARKINGU S ADAPTÉREM"
echo "=================================="

# Kontrola, že jsme ve správném adresáři
if [ ! -f "run_benchmark.py" ]; then
    echo "❌ Spusťte skript z adresáře 3_benchmarking/"
    exit 1
fi

# Nastavení cache prostředí
echo "🔧 Nastavuji cache prostředí..."

# Nastavení cache do /workspace
export HF_HOME="/workspace/.cache/huggingface"
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/workspace/.cache/huggingface/datasets"

# Vytvoření cache adresářů
mkdir -p /workspace/.cache/huggingface/transformers
mkdir -p /workspace/.cache/huggingface/datasets

echo "✅ Cache nastaveno:"
echo "   HF_HOME: $HF_HOME"
echo "   TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "   HF_DATASETS_CACHE: $HF_DATASETS_CACHE"

# Test centrálního setup modulu
echo ""
echo "🧪 Test centrálního setup modulu..."
python -c "
import sys
from pathlib import Path
sys.path.append(str(Path('.').parent))
import setup_environment
print('✅ Centrální setup modul OK')
"

# Rychlý test adaptéru
echo ""
echo "🔍 Rychlý test adaptéru..."
python quick_test_adapter.py

if [ $? -ne 0 ]; then
    echo "❌ Rychlý test selhal!"
    echo "💡 Zkontrolujte:"
    echo "   - Máte přístup k modelu mcmatak/babis-mistral-adapter?"
    echo "   - Jste přihlášeni na Hugging Face?"
    echo "   - Máte dostatek místa v /workspace?"
    exit 1
fi

echo ""
echo "✅ Rychlý test úspěšný!"

# Spuštění hlavního benchmarkingu
echo ""
echo "📊 Spouštím hlavní benchmarking..."
python run_benchmark.py

if [ $? -ne 0 ]; then
    echo "❌ Benchmarking selhal!"
    exit 1
fi

echo ""
echo "🎉 BENCHMARKING DOKONČEN!"
echo "=========================="
echo ""
echo "📁 Výstupy:"
echo "   📊 Excel report: results/reports/benchmark_report.xlsx"
echo "   📈 Grafy: results/visualizations/"
echo "   📋 Shrnutí: results/reports/benchmark_summary.txt"
echo ""
echo "🚀 Váš adaptér byl úspěšně otestován!"
echo "💡 Výsledky jsou připraveny pro odevzdání." 