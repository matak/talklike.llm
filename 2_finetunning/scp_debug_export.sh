#!/bin/bash
# SCP Export Debug Dat z RunPod

set -e

echo "🚀 SCP Export Debug Dat z RunPod"
echo "================================"

# Kontrola, zda jsme na RunPod
if [[ ! -d "/workspace" ]]; then
    echo "❌ Tento skript je určen pro spuštění na RunPod"
    echo "💡 Spusťte ho na RunPod instance"
    exit 1
fi

# Hledání debug adresářů
echo "🔍 Hledám debug adresáře..."
DEBUG_DIRS=$(ls -d debug_dataset_finetune_* 2>/dev/null || echo "")

if [[ -z "$DEBUG_DIRS" ]]; then
    echo "❌ Nenalezeny žádné debug adresáře"
    echo "💡 Spusťte nejdříve fine-tuning: python finetune.py"
    exit 1
fi

# Výběr nejnovějšího adresáře
LATEST_DIR=$(ls -t debug_dataset_finetune_* | head -1)
echo "📂 Používám nejnovější: $LATEST_DIR"

# Kontrola velikosti
SIZE=$(du -sh "$LATEST_DIR" | cut -f1)
echo "📊 Velikost: $SIZE"

# Vytvoření komprese
echo "📦 Vytvářím kompresi..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="debug_data_${TIMESTAMP}.tar.gz"

tar -czf "$ARCHIVE_NAME" "$LATEST_DIR"

# Výpis výsledku
ARCHIVE_SIZE=$(du -h "$ARCHIVE_NAME" | cut -f1)
echo "✅ Archiv vytvořen: $ARCHIVE_NAME ($ARCHIVE_SIZE)"

# Zjištění IP adresy
echo "🌐 Zjišťuji IP adresu..."
IP_ADDRESS=$(hostname -I | awk '{print $1}')
echo "📍 IP adresa: $IP_ADDRESS"

# Výpis obsahu debug dat
echo ""
echo "📋 Obsah debug dat:"
ls -la "$LATEST_DIR" | head -10

# Instrukce pro SCP
echo ""
echo "💡 SCP Instrukce pro stažení na lokální PC:"
echo "============================================"
echo ""
echo "1. Otevřete terminál na vašem lokálním PC"
echo "2. Spusťte jeden z následujících příkazů:"
echo ""
echo "📥 Stažení celého archivu:"
echo "   scp username@$IP_ADDRESS:/workspace/$ARCHIVE_NAME ./"
echo ""
echo "📥 Stažení konkrétního souboru:"
echo "   scp username@$IP_ADDRESS:/workspace/$LATEST_DIR/debug_summary.txt ./"
echo ""
echo "📥 Stažení celého debug adresáře:"
echo "   scp -r username@$IP_ADDRESS:/workspace/$LATEST_DIR ./debug_data/"
echo ""
echo "📥 Stažení pouze čitelných souborů:"
echo "   scp username@$IP_ADDRESS:/workspace/$LATEST_DIR/debug_summary.txt ./"
echo "   scp username@$IP_ADDRESS:/workspace/$LATEST_DIR/dataset_statistics.json ./"
echo "   scp username@$IP_ADDRESS:/workspace/$LATEST_DIR/train_dataset_readable.txt ./"
echo ""

# Kontrola SSH klíčů
echo "🔑 SSH Konfigurace:"
echo "==================="
echo ""
echo "Pokud používáte SSH klíče:"
echo "   scp -i ~/.ssh/your_key username@$IP_ADDRESS:/workspace/$ARCHIVE_NAME ./"
echo ""
echo "Pokud používáte heslo:"
echo "   scp username@$IP_ADDRESS:/workspace/$ARCHIVE_NAME ./"
echo "   (budete vyzváni k zadání hesla)"
echo ""

# Výpis užitečných souborů
echo "📄 Nejužitečnější soubory pro analýzu:"
echo "======================================"
echo "   debug_summary.txt              - Přehled všech kroků"
echo "   dataset_statistics.json        - Kompletní statistiky"
echo "   train_dataset_readable.txt     - Čitelná verze train dat"
echo "   validation_dataset_readable.txt - Čitelná verze validation dat"
echo "   step_08_dataset_stats.json     - Statistiky datasetu"
echo ""

# Kontrola, zda existuje soubor
if [[ -f "$ARCHIVE_NAME" ]]; then
    echo "✅ Archiv je připraven ke stažení!"
    echo "📁 Cesta: /workspace/$ARCHIVE_NAME"
    echo "📊 Velikost: $ARCHIVE_SIZE"
else
    echo "❌ Chyba: Archiv nebyl vytvořen"
    exit 1
fi

echo ""
echo "🎯 Tip: Pro rychlé stažení použijte:"
echo "   scp username@$IP_ADDRESS:/workspace/$ARCHIVE_NAME ./" 