#!/bin/bash

# Rychlý skript pro nahrání modelu na Hugging Face Hub
# Použijte: ./quick_upload.sh "vas-username/babis-model"

if [ $# -eq 0 ]; then
    echo "❌ Chyba: Musíte zadat název modelu na HF Hub"
    echo "💡 Použití: ./quick_upload.sh \"vas-username/babis-model\""
    echo ""
    echo "📋 Příklady:"
    echo "  ./quick_upload.sh \"jan-novak/babis-dialogpt\""
    echo "  ./quick_upload.sh \"my-org/babis-mistral\""
    exit 1
fi

HUB_MODEL_ID="$1"

echo "🚀 Rychlé nahrání modelu na Hugging Face Hub"
echo "🎯 Model ID: $HUB_MODEL_ID"

# Kontrola HF tokenu
if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN není nastaven!"
    echo "💡 Nastavte HF_TOKEN v prostředí nebo .env souboru"
    exit 1
fi

# Hledání modelu
MODEL_PATHS=(
    "/workspace/babis-finetuned-final"
    "/workspace/babis-mistral-finetuned-final"
    "/workspace/babis-finetuned"
    "/workspace/babis-mistral-finetuned"
)

FOUND_MODEL=""
for path in "${MODEL_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "✅ Nalezen model v: $path"
        FOUND_MODEL="$path"
        break
    fi
done

if [ -z "$FOUND_MODEL" ]; then
    echo "❌ Nebyl nalezen žádný model!"
    echo "📁 Hledané cesty:"
    for path in "${MODEL_PATHS[@]}"; do
        echo "  - $path"
    done
    echo ""
    echo "💡 Spusťte nejdříve fine-tuning nebo zadejte cestu ručně:"
    echo "   python upload_to_hf.py --model_path /cesta/k/modelu --hub_model_id \"$HUB_MODEL_ID\""
    exit 1
fi

echo ""
echo "📤 Nahrávám model na HF Hub..."
echo "📍 Cesta: $FOUND_MODEL"
echo "🎯 HF ID: $HUB_MODEL_ID"

# Spuštění Python skriptu
PYTHONPATH="$(pwd):$PYTHONPATH" python 2_finetunning/upload_to_hf.py \
    --model_path "$FOUND_MODEL" \
    --hub_model_id "$HUB_MODEL_ID"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Nahrávání dokončeno!"
    echo "🔗 Model je dostupný na: https://huggingface.co/$HUB_MODEL_ID"
else
    echo ""
    echo "❌ Nahrávání selhalo!"
    echo "💡 Zkontrolujte:"
    echo "   - HF_TOKEN je správně nastaven"
    echo "   - Máte oprávnění k nahrání na daný repository"
    echo "   - Model obsahuje všechny potřebné soubory"
fi 