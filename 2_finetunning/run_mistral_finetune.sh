#!/bin/bash

# Check if we're in the root directory (where .env file should be)
if [ ! -f ".env" ]; then
    # Try to find the root directory by going up
    if [ -f "../.env" ]; then
        echo "📁 Přepínám do root directory..."
        cd ..
    elif [ -f "../../.env" ]; then
        echo "📁 Přepínám do root directory..."
        cd ../..
    else
        echo "❌ Soubor .env nebyl nalezen ani v aktuálním adresáři ani v nadřazených adresářích!"
        echo "📝 Ujistěte se, že jste v root directory projektu nebo že .env soubor existuje."
        exit 1
    fi
fi

# Kontrola existence .env souboru
if [ ! -f ".env" ]; then
    echo "❌ Soubor .env nebyl nalezen!"
    echo "📝 Vytvořte soubor .env s následujícím obsahem:"
    echo "HF_TOKEN=hf_your_token_here"
    echo "WANDB_API_KEY=your_wandb_token_here"
    exit 1
fi

# Kontrola dostupnosti dat
if [ ! -f "data/all.jsonl" ]; then
    echo "❌ Soubor data/all.jsonl nebyl nalezen!"
    echo "💡 Zkontrolujte, zda máte data v adresáři data/"
    exit 1
fi

# Instalace závislostí
if [ -f "2_finetunning/requirements_finetunning.txt" ]; then
    echo "📦 Instaluji závislosti..."
    pip install -r 2_finetunning/requirements_finetunning.txt
fi

# Spuštění fine-tuningu s Mistralem
echo "🤖 Spouštím fine-tuning s Mistral-7B-Instruct-v0.3..."
python 2_finetunning/finetune_babis.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --output_dir /workspace/babis-mistral-finetuned \
    --epochs 2 \
    --batch_size 1 \
    --max_length 2048 \
    --aggressive_cleanup

echo "✅ Fine-tuning dokončen!"
echo "📁 Model je uložen v: /workspace/babis-mistral-finetuned-final"
echo "💾 Všechny soubory jsou na network storage" 