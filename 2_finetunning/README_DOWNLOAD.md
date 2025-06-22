# 📥 Stahování dat z RunPod pomocí SCP

Univerzální návod pro stažení všech typů dat z RunPod na lokální PC pomocí SCP.

## 🚀 Rychlý start

### Stažení dat na lokální PC
```bash
# Stažení celého projektu
scp -P 40100 -i C:\Users\info\.ssh\runpod -r root@213.192.2.107:/workspace/talklike.llm ./

# Stažení pouze výsledků
scp -P 40100 -i C:\Users\info\.ssh\runpod -r root@213.192.2.107:/workspace/talklike.llm/3_benchmarking/results ./3_benchmarking/results
```

## 📋 Typy dat ke stažení

### Benchmarking výsledky
```bash
# Celý benchmarking adresář
scp -P 40100 -i C:\Users\info\.ssh\runpod -r root@213.192.2.107:/workspace/talklike.llm/3_benchmarking/results ./3_benchmarking/results

# Jednotlivé soubory
scp -P 40100 -i C:\Users\info\.ssh\runpod root@213.192.2.107:/workspace/talklike.llm/3_benchmarking/results/before_finetune/responses.json ./
scp -P 40100 -i C:\Users\info\.ssh\runpod root@213.192.2.107:/workspace/talklike.llm/3_benchmarking/results/after_finetune/responses.json ./
scp -P 40100 -i C:\Users\info\.ssh\runpod root@213.192.2.107:/workspace/talklike.llm/3_benchmarking/results/comparison/style_evaluation.json ./
scp -P 40100 -i C:\Users\info\.ssh\runpod root@213.192.2.107:/workspace/talklike.llm/3_benchmarking/results/reports/benchmark_summary.md ./
```

### Fine-tuning data
```bash
# Debug data z fine-tuningu
scp -P 40100 -i C:\Users\info\.ssh\runpod -r root@213.192.2.107:/workspace/talklike.llm/2_finetunning/debug_dataset_finetune_* ./debug_data

# Model adaptéry
scp -P 40100 -i C:\Users\info\.ssh\runpod -r root@213.192.2.107:/workspace/talklike.llm/2_finetunning/adapters ./adapters

# Logy trénování
scp -P 40100 -i C:\Users\info\.ssh\runpod root@213.192.2.107:/workspace/talklike.llm/2_finetunning/training_logs.txt ./
```

### Data preparation
```bash
# Datasets
scp -P 40100 -i C:\Users\info\.ssh\runpod -r root@213.192.2.107:/workspace/talklike.llm/data ./data

# Generated data
scp -P 40100 -i C:\Users\info\.ssh\runpod -r root@213.192.2.107:/workspace/talklike.llm/1_data_preparation/generated_data ./generated_data
```

## 🔧 Konfigurace SSH

### SSH klíče (doporučeno)
```bash
# Generování SSH klíče
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# Kopírování klíče na RunPod
ssh-copy-id -i C:\Users\info\.ssh\runpod root@213.192.2.107 -p 40100

# Test připojení
ssh -i C:\Users\info\.ssh\runpod root@213.192.2.107 -p 40100 "echo 'SSH funguje!'"
```

### SSH heslo
```bash
# Stažení s heslem (budete vyzváni)
scp -P 40100 root@213.192.2.107:/workspace/talklike.llm/3_benchmarking/results ./3_benchmarking/results
```

## 📁 Struktura stažených dat

### Benchmarking results
```
3_benchmarking/results/
├── before_finetune/
│   └── responses.json          # Odpovědi před fine-tuningem
├── after_finetune/
│   └── responses.json          # Odpovědi po fine-tuningem
├── comparison/
│   ├── model_comparison.json   # Metriky srovnání
│   └── style_evaluation.json   # Bodové hodnocení stylu
├── reports/
│   └── benchmark_summary.md    # Markdown shrnutí
└── visualizations/
    ├── score_comparison.png    # Graf srovnání skóre
    ├── improvement_metrics.png # Graf zlepšení metrik
    └── grade_distribution.png  # Graf distribuce známek
```

### Fine-tuning debug data
```
debug_data/
├── debug_dataset_finetune_YYYYMMDD_HHMMSS/
│   ├── debug_summary.txt              # Přehled všech kroků
│   ├── dataset_statistics.json        # Kompletní statistiky
│   ├── train_dataset.jsonl            # Train dataset
│   ├── validation_dataset.jsonl       # Validation dataset
│   ├── train_dataset_readable.txt     # Čitelná verze train
│   └── validation_dataset_readable.txt # Čitelná verze validation
```

## 🔍 Analýza stažených dat

### Benchmarking výsledky
```bash
# Zobrazení Markdown shrnutí
cat 3_benchmarking/results/reports/benchmark_summary.md

# Analýza JSON dat
jq '.' 3_benchmarking/results/before_finetune/responses.json
jq '.' 3_benchmarking/results/after_finetune/responses.json
jq '.' 3_benchmarking/results/comparison/style_evaluation.json

# Rychlý přehled metrik
jq '.improvement' 3_benchmarking/results/comparison/model_comparison.json
```

### Fine-tuning data
```bash
# Shrnutí debug dat
cat debug_data/debug_dataset_finetune_*/debug_summary.txt

# Statistiky datasetu
jq '.' debug_data/debug_dataset_finetune_*/dataset_statistics.json

# Čitelná verze dat
head -50 debug_data/debug_dataset_finetune_*/train_dataset_readable.txt
```

## ⚡ Užitečné příkazy

### Kontrola velikosti
```bash
# Velikost výsledků na RunPod
ssh -i C:\Users\info\.ssh\runpod root@213.192.2.107 -p 40100 "du -sh /workspace/talklike.llm/3_benchmarking/results"

# Velikost debug dat
ssh -i C:\Users\info\.ssh\runpod root@213.192.2.107 -p 40100 "du -sh /workspace/talklike.llm/2_finetunning/debug_dataset_finetune_*"
```

### Selektivní stažení
```bash
# Pouze Markdown shrnutí
scp -P 40100 root@213.192.2.107:/workspace/talklike.llm/3_benchmarking/results/reports/benchmark_summary.md ./

# Pouze grafy
scp -P 40100 -r root@213.192.2.107:/workspace/talklike.llm/3_benchmarking/results/visualizations ./visualizations

# Pouze nejnovější debug data
LATEST_DEBUG=$(ssh -i C:\Users\info\.ssh\runpod root@213.192.2.107 -p 40100 "ls -t /workspace/talklike.llm/2_finetunning/debug_dataset_finetune_* | head -1")
scp -P 40100 -r root@213.192.2.107:"$LATEST_DEBUG" ./latest_debug_data
```

### Automatické stažení
```bash
#!/bin/bash
# Skript pro automatické stažení všech výsledků

RUNPOD_USER="root"
RUNPOD_HOST="213.192.2.107"
SSH_KEY="C:\Users\info\.ssh\runpod"
LOCAL_DIR="./downloaded_results"

mkdir -p "$LOCAL_DIR"

echo "📥 Stahuji benchmarking výsledky..."
scp -P 40100 -i $SSH_KEY -r $RUNPOD_USER@$RUNPOD_HOST:/workspace/talklike.llm/3_benchmarking/results "$LOCAL_DIR/3_benchmarking/results"

echo "📥 Stahuji fine-tuning data..."
scp -P 40100 -i $SSH_KEY -r $RUNPOD_USER@$RUNPOD_HOST:/workspace/talklike.llm/2_finetunning/debug_dataset_finetune_* "$LOCAL_DIR/debug_data"

echo "📥 Stahuji datasets..."
scp -P 40100 -i $SSH_KEY -r $RUNPOD_USER@$RUNPOD_HOST:/workspace/talklike.llm/data "$LOCAL_DIR/data"

echo "✅ Všechna data stažena do: $LOCAL_DIR"
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
ssh -i C:\Users\info\.ssh\runpod -v root@213.192.2.107 -p 40100

# Kontrola SSH klíčů
ls -la C:\Users\info\.ssh\

# Reset SSH connection
ssh-keygen -R 213.192.2.107
```

### Problém: Soubory neexistují
```bash
# Kontrola existence souborů na RunPod
ssh -i C:\Users\info\.ssh\runpod root@213.192.2.107 -p 40100 "ls -la /workspace/talklike.llm/3_benchmarking/results/"

# Kontrola práv
ssh -i C:\Users\info\.ssh\runpod root@213.192.2.107 -p 40100 "ls -la /workspace/talklike.llm/3_benchmarking/results/"
```

### Problém: Nedostatek místa
```bash
# Kontrola volného místa
df -h

# Kontrola velikosti stažených dat
du -sh 3_benchmarking/results/
```

## 📝 Příklady použití

### Stažení pouze výsledků benchmarkingu
```bash
scp -P 40100 -r root@213.192.2.107:/workspace/talklike.llm/3_benchmarking/results ./3_benchmarking/results
```

### Stažení celého projektu
```bash
scp -P 40100 -r root@213.192.2.107:/workspace/talklike.llm ./talklike_llm_backup
```

### Stažení konkrétního souboru
```bash
scp -P 40100 root@213.192.2.107:/workspace/talklike.llm/3_benchmarking/results/reports/benchmark_summary.md ./my_report.md
```

---

**💡 Tip:** Pro časté stahování si vytvořte alias v `~/.bashrc`:
```bash
alias download-benchmark='scp -P 40100 -r root@213.192.2.107:/workspace/talklike.llm/3_benchmarking/results ./3_benchmarking/results'
``` 