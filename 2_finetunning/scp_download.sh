#!/bin/bash
# SCP Download Debug Dat na lokální PC

set -e

echo "📥 SCP Download Debug Dat na lokální PC"
echo "======================================="

# Kontrola argumentů
if [[ $# -lt 2 ]]; then
    echo "❌ Použití: $0 <username> <runpod-ip> [local-dir]"
    echo ""
    echo "Příklady:"
    echo "  $0 root 192.168.1.100"
    echo "  $0 root 192.168.1.100 ./debug_data"
    echo ""
    echo "💡 Pro zjištění IP adresy na RunPod spusťte: hostname -I"
    exit 1
fi

USERNAME=$1
RUNPOD_IP=$2
LOCAL_DIR=${3:-"./debug_data"}

echo "🔗 Připojení k: $USERNAME@$RUNPOD_IP"
echo "📁 Lokální adresář: $LOCAL_DIR"

# Vytvoření lokálního adresáře
mkdir -p "$LOCAL_DIR"
echo "✅ Lokální adresář vytvořen: $LOCAL_DIR"

# Kontrola připojení
echo "🔍 Kontroluji připojení..."
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$USERNAME@$RUNPOD_IP" exit 2>/dev/null; then
    echo "⚠️ SSH klíče nejsou nastaveny - budete vyzváni k zadání hesla"
fi

# Hledání debug archivů na RunPod
echo "🔍 Hledám debug archivy na RunPod..."
ARCHIVES=$(ssh "$USERNAME@$RUNPOD_IP" "ls -t /workspace/debug_data_*.tar.gz 2>/dev/null || echo ''")

if [[ -z "$ARCHIVES" ]]; then
    echo "❌ Nenalezeny žádné debug archivy na RunPod"
    echo "💡 Spusťte nejdříve na RunPod: ./scp_debug_export.sh"
    exit 1
fi

# Výpis nalezených archivů
echo "📦 Nalezené archivy:"
for archive in $ARCHIVES; do
    echo "  - $archive"
done

# Výběr nejnovějšího archivu
LATEST_ARCHIVE=$(echo "$ARCHIVES" | head -1)
ARCHIVE_NAME=$(basename "$LATEST_ARCHIVE")
echo "📂 Používám nejnovější: $ARCHIVE_NAME"

# Stažení archivu
echo "📥 Stahuji archiv..."
scp "$USERNAME@$RUNPOD_IP:$LATEST_ARCHIVE" "$LOCAL_DIR/"

# Kontrola stažení
if [[ -f "$LOCAL_DIR/$ARCHIVE_NAME" ]]; then
    echo "✅ Archiv stažen: $LOCAL_DIR/$ARCHIVE_NAME"
    
    # Rozbalení archivu
    echo "📦 Rozbaluji archiv..."
    cd "$LOCAL_DIR"
    tar -xzf "$ARCHIVE_NAME"
    
    # Výpis rozbaleného obsahu
    echo ""
    echo "📋 Rozbalený obsah:"
    ls -la | grep debug_dataset_finetune
    
    # Návrat do původního adresáře
    cd - > /dev/null
    
    echo ""
    echo "🎉 Debug data úspěšně stažena a rozbalena!"
    echo "📁 Umístění: $LOCAL_DIR"
    echo ""
    echo "📄 Nejužitečnější soubory:"
    echo "   debug_summary.txt              - Přehled všech kroků"
    echo "   dataset_statistics.json        - Kompletní statistiky"
    echo "   train_dataset_readable.txt     - Čitelná verze train dat"
    echo "   validation_dataset_readable.txt - Čitelná verze validation dat"
    echo ""
    echo "💡 Pro rychlé prohlížení:"
    echo "   cat $LOCAL_DIR/*/debug_summary.txt"
    echo "   cat $LOCAL_DIR/*/dataset_statistics.json | jq '.'"
    
else
    echo "❌ Chyba při stažení archivu"
    exit 1
fi

echo ""
echo "🎯 Tip: Pro stažení konkrétního souboru použijte:"
echo "   scp $USERNAME@$RUNPOD_IP:/workspace/debug_dataset_finetune_*/debug_summary.txt ./" 