# 📤 SCP Export Debug Dat z RunPod

Rychlý a spolehlivý způsob, jak získat debug data z RunPod na vaše lokální PC pomocí SCP.

## 🚀 Rychlý start

### 1. Na RunPod - příprava dat

```bash
# Spuštění export skriptu
chmod +x scp_debug_export.sh
./scp_debug_export.sh
```

Skript automaticky:
- Najde nejnovější debug adresář
- Vytvoří komprimovaný archiv
- Zobrazí IP adresu a instrukce pro SCP

### 2. Na lokálním PC - stažení dat

```bash
# Stažení pomocí automatického skriptu
chmod +x scp_download.sh
./scp_download.sh root 192.168.1.100

# Nebo manuální stažení
scp root@192.168.1.100:/workspace/debug_data_*.tar.gz ./
```

## 📋 Manuální postup

### Krok 1: Příprava na RunPod

```bash
# 1. Najděte debug adresář
ls -la debug_dataset_finetune_*

# 2. Vytvořte kompresi
tar -czf debug_data_$(date +%Y%m%d_%H%M%S).tar.gz debug_dataset_finetune_*

# 3. Zjistěte IP adresu
hostname -I
```

### Krok 2: Stažení na lokální PC

```bash
# Stažení celého archivu
scp username@runpod-ip:/workspace/debug_data_*.tar.gz ./

# Stažení konkrétního souboru
scp username@runpod-ip:/workspace/debug_dataset_finetune_*/debug_summary.txt ./

# Stažení celého adresáře
scp -r username@runpod-ip:/workspace/debug_dataset_finetune_* ./debug_data/
```

### Krok 3: Rozbalení a analýza

```bash
# Rozbalení archivu
tar -xzf debug_data_*.tar.gz

# Rychlý přehled
cat debug_dataset_finetune_*/debug_summary.txt

# Statistiky
cat debug_dataset_finetune_*/dataset_statistics.json | jq '.'
```

## 🔧 Konfigurace SSH

### SSH klíče (doporučeno)

```bash
# Generování SSH klíče
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# Kopírování klíče na RunPod
ssh-copy-id username@runpod-ip

# Test připojení
ssh username@runpod-ip "echo 'SSH funguje!'"
```

### SSH heslo

```bash
# Stažení s heslem (budete vyzváni)
scp username@runpod-ip:/workspace/debug_data_*.tar.gz ./
```

## 📁 Obsah debug dat

Po stažení a rozbalení budete mít:

```
debug_dataset_finetune_YYYYMMDD_HHMMSS/
├── 📄 step_*.json                    # Mezikroky zpracování
├── 📄 sample_*.json                  # Ukázkové položky
├── 📄 train_dataset.jsonl            # Train dataset
├── 📄 validation_dataset.jsonl       # Validation dataset
├── 📄 train_dataset_readable.txt     # Čitelná verze train
├── 📄 validation_dataset_readable.txt # Čitelná verze validation
├── 📄 dataset_statistics.json        # Kompletní statistiky
└── 📄 debug_summary.txt              # Shrnutí všech kroků
```

## 🔍 Analýza dat

### Rychlý přehled

```bash
# Shrnutí všech kroků
cat debug_dataset_finetune_*/debug_summary.txt

# Statistiky datasetu
cat debug_dataset_finetune_*/dataset_statistics.json | jq '.'

# Čitelná verze train dat
head -50 debug_dataset_finetune_*/train_dataset_readable.txt
```

### Python analýza

```python
import json
import glob

# Načtení debug dat
debug_dirs = glob.glob("debug_dataset_finetune_*")
latest_dir = max(debug_dirs, key=os.path.getctime)

# Načtení statistik
with open(f"{latest_dir}/dataset_statistics.json", 'r') as f:
    stats = json.load(f)

print(f"Train samples: {stats['train_dataset']['size']}")
print(f"Validation samples: {stats['validation_dataset']['size']}")
print(f"Avg token length: {stats['train_dataset']['token_lengths']['avg']:.1f}")

# Kontrola tagů
tags = stats['train_dataset']['tags']
print(f"System messages: {tags['system']}")
print(f"User messages: {tags['user']}")
print(f"Assistant messages: {tags['assistant']}")
```

## ⚡ Užitečné příkazy

### Stažení konkrétních souborů

```bash
# Pouze shrnutí
scp username@runpod-ip:/workspace/debug_dataset_finetune_*/debug_summary.txt ./

# Pouze statistiky
scp username@runpod-ip:/workspace/debug_dataset_finetune_*/dataset_statistics.json ./

# Pouze čitelné verze
scp username@runpod-ip:/workspace/debug_dataset_finetune_*/train_dataset_readable.txt ./
scp username@runpod-ip:/workspace/debug_dataset_finetune_*/validation_dataset_readable.txt ./
```

### Kontrola velikosti

```bash
# Velikost debug adresáře na RunPod
ssh username@runpod-ip "du -sh /workspace/debug_dataset_finetune_*"

# Velikost archivu
ssh username@runpod-ip "ls -lh /workspace/debug_data_*.tar.gz"
```

### Automatické stažení

```bash
# Stažení nejnovějšího archivu
LATEST=$(ssh username@runpod-ip "ls -t /workspace/debug_data_*.tar.gz | head -1")
scp username@runpod-ip:"$LATEST" ./
```

## 🎯 Výhody SCP

✅ **Rychlost** - Přímé připojení bez zprostředkovatelů  
✅ **Spolehlivost** - Standardní SSH protokol  
✅ **Bezpečnost** - Šifrovaný přenos  
✅ **Flexibilita** - Stažení celých adresářů nebo jednotlivých souborů  
✅ **Automatizace** - Možnost skriptování  

## 🔧 Troubleshooting

### Problém: SSH připojení selhává

```bash
# Test připojení
ssh -v username@runpod-ip

# Kontrola SSH klíčů
ls -la ~/.ssh/

# Regenerace klíčů
ssh-keygen -t rsa -b 4096
ssh-copy-id username@runpod-ip
```

### Problém: Soubor neexistuje

```bash
# Kontrola souborů na RunPod
ssh username@runpod-ip "ls -la /workspace/debug_*"

# Kontrola místa na disku
ssh username@runpod-ip "df -h /workspace"
```

### Problém: Pomalé stažení

```bash
# Komprese s vyšší úrovní
ssh username@runpod-ip "cd /workspace && tar -czf - debug_dataset_finetune_*" | tar -xzf -

# Použití rsync pro pokračování
rsync -avz --partial username@runpod-ip:/workspace/debug_data_*.tar.gz ./
```

## 📞 Podpora

Pokud máte problémy:

1. **Zkontrolujte připojení**: `ping runpod-ip`
2. **Ověřte SSH**: `ssh username@runpod-ip "echo test"`
3. **Kontrolujte soubory**: `ssh username@runpod-ip "ls -la /workspace/debug_*"`
4. **Zkuste alternativu**: Použijte RunPod web terminal pro stažení

---

**Tip**: Pro nejrychlejší workflow použijte automatické skripty `scp_debug_export.sh` a `scp_download.sh`! 