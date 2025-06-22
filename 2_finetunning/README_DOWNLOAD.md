# 📥 Stahování dat z RunPod pomocí SCP

Univerzální návod pro stažení všech typů dat z RunPod na lokální PC pomocí SCP.

## 🚀 Rychlý start

### Stažení dat na lokální PC
```bash
# Stažení celého projektu
scp -i ~/.ssh/runpod -r l6twnmqglae2fo-64411626@ssh.runpod.io:/workspace/talklike.llm ./

# Stažení pouze výsledků
scp -i ~/.ssh/runpod -r l6twnmqglae2fo-64411626@ssh.runpod.io:/workspace/talklike.llm/3_benchmarking/results ./benchmark_results
```

## 📋 Typy dat ke stažení

### Benchmarking výsledky
```bash
# Celý benchmarking adresář
scp -i ~/.ssh/runpod -r l6twnmqglae2fo-64411626@ssh.runpod.io:/workspace/talklike.llm/3_benchmarking/results ./benchmark_results

# Jednotlivé soubory
scp -i ~/.ssh/runpod l6twnmqglae2fo-64411626@ssh.runpod.io:/workspace/talklike.llm/3_benchmarking/results/before_finetune/responses.json ./
scp -i ~/.ssh/runpod l6twnmqglae2fo-64411626@ssh.runpod.io:/workspace/talklike.llm/3_benchmarking/results/after_finetune/responses.json ./
scp -i ~/.ssh/runpod l6twnmqglae2fo-64411626@ssh.runpod.io:/workspace/talklike.llm/3_benchmarking/results/comparison/style_evaluation.json ./
scp -i ~/.ssh/runpod l6twnmqglae2fo-64411626@ssh.runpod.io:/workspace/talklike.llm/3_benchmarking/results/reports/benchmark_summary.md ./
```

### Fine-tuning data
```bash
# Debug data z fine-tuningu
scp -i ~/.ssh/runpod -r l6twnmqglae2fo-64411626@ssh.runpod.io:/workspace/talklike.llm/2_finetunning/debug_dataset_finetune_* ./debug_data

# Model adaptéry
scp -i ~/.ssh/runpod -r l6twnmqglae2fo-64411626@ssh.runpod.io:/workspace/talklike.llm/2_finetunning/adapters ./adapters

# Logy trénování
scp -i ~/.ssh/runpod l6twnmqglae2fo-64411626@ssh.runpod.io:/workspace/talklike.llm/2_finetunning/training_logs.txt ./
```

### Data preparation
```bash
# Datasets
scp -i ~/.ssh/runpod -r l6twnmqglae2fo-64411626@ssh.runpod.io:/workspace/talklike.llm/data ./data

# Generated data
scp -i ~/.ssh/runpod -r l6twnmqglae2fo-64411626@ssh.runpod.io:/workspace/talklike.llm/1_data_preparation/generated_data ./generated_data
```

## 🔧 Konfigurace SSH

### SSH klíče (doporučeno)
```bash
# Generování SSH klíče
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# Kopírování klíče na RunPod
ssh-copy-id -i ~/.ssh/runpod l6twnmqglae2fo-64411626@ssh.runpod.io

# Test připojení
ssh -i ~/.ssh/runpod l6twnmqglae2fo-64411626@ssh.runpod.io "echo 'SSH funguje!'"
```

### SSH heslo
```bash
# Stažení s heslem (budete vyzváni)
scp l6twnmqglae2fo-64411626@ssh.runpod.io:/workspace/talklike.llm/3_benchmarking/results ./benchmark_results
```

## 📁 Struktura stažených dat

### Benchmarking results
```
benchmark_results/
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
cat benchmark_results/reports/benchmark_summary.md

# Analýza JSON dat
jq '.' benchmark_results/before_finetune/responses.json
jq '.' benchmark_results/after_finetune/responses.json
jq '.' benchmark_results/comparison/style_evaluation.json

# Rychlý přehled metrik
jq '.improvement' benchmark_results/comparison/model_comparison.json
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
ssh -i ~/.ssh/runpod l6twnmqglae2fo-64411626@ssh.runpod.io "du -sh /workspace/talklike.llm/3_benchmarking/results"

# Velikost debug dat
ssh -i ~/.ssh/runpod l6twnmqglae2fo-64411626@ssh.runpod.io "du -sh /workspace/talklike.llm/2_finetunning/debug_dataset_finetune_*"
```

### Selektivní stažení
```bash
# Pouze Markdown shrnutí
scp -i ~/.ssh/runpod l6twnmqglae2fo-64411626@ssh.runpod.io:/workspace/talklike.llm/3_benchmarking/results/reports/benchmark_summary.md ./

# Pouze grafy
scp -i ~/.ssh/runpod -r l6twnmqglae2fo-64411626@ssh.runpod.io:/workspace/talklike.llm/3_benchmarking/results/visualizations ./visualizations

# Pouze nejnovější debug data
LATEST_DEBUG=$(ssh -i ~/.ssh/runpod l6twnmqglae2fo-64411626@ssh.runpod.io "ls -t /workspace/talklike.llm/2_finetunning/debug_dataset_finetune_* | head -1")
scp -i ~/.ssh/runpod -r l6twnmqglae2fo-64411626@ssh.runpod.io:"$LATEST_DEBUG" ./latest_debug_data
```

### Automatické stažení
```bash
#!/bin/bash
# Skript pro automatické stažení všech výsledků

RUNPOD_USER="l6twnmqglae2fo-64411626"
RUNPOD_HOST="ssh.runpod.io"
SSH_KEY="~/.ssh/runpod"
LOCAL_DIR="./downloaded_results"

mkdir -p "$LOCAL_DIR"

echo "📥 Stahuji benchmarking výsledky..."
scp -i $SSH_KEY -r $RUNPOD_USER@$RUNPOD_HOST:/workspace/talklike.llm/3_benchmarking/results "$LOCAL_DIR/benchmark_results"

echo "📥 Stahuji fine-tuning data..."
scp -i $SSH_KEY -r $RUNPOD_USER@$RUNPOD_HOST:/workspace/talklike.llm/2_finetunning/debug_dataset_finetune_* "$LOCAL_DIR/debug_data"

echo "📥 Stahuji datasets..."
scp -i $SSH_KEY -r $RUNPOD_USER@$RUNPOD_HOST:/workspace/talklike.llm/data "$LOCAL_DIR/data"

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
ssh -i ~/.ssh/runpod -v l6twnmqglae2fo-64411626@ssh.runpod.io

# Kontrola SSH klíčů
ls -la ~/.ssh/

# Reset SSH connection
ssh-keygen -R ssh.runpod.io
```

### Problém: Soubory neexistují
```bash
# Kontrola existence souborů na RunPod
ssh -i ~/.ssh/runpod l6twnmqglae2fo-64411626@ssh.runpod.io "ls -la /workspace/talklike.llm/3_benchmarking/results/"

# Kontrola práv
ssh -i ~/.ssh/runpod l6twnmqglae2fo-64411626@ssh.runpod.io "ls -la /workspace/talklike.llm/3_benchmarking/results/"
```

### Problém: Nedostatek místa
```bash
# Kontrola volného místa
df -h

# Kontrola velikosti stažených dat
du -sh benchmark_results/
```

## 📝 Příklady použití

### Stažení pouze výsledků benchmarkingu
```bash
scp -i ~/.ssh/runpod -r l6twnmqglae2fo-64411626@ssh.runpod.io:/workspace/talklike.llm/3_benchmarking/results ./benchmark_results
```

### Stažení celého projektu
```bash
scp -i ~/.ssh/runpod -r l6twnmqglae2fo-64411626@ssh.runpod.io:/workspace/talklike.llm ./talklike_llm_backup
```

### Stažení konkrétního souboru
```bash
scp -i ~/.ssh/runpod l6twnmqglae2fo-64411626@ssh.runpod.io:/workspace/talklike.llm/3_benchmarking/results/reports/benchmark_summary.md ./my_report.md
```

---

**💡 Tip:** Pro časté stahování si vytvořte alias v `~/.bashrc`:
```bash
alias download-benchmark='scp -i ~/.ssh/runpod -r l6twnmqglae2fo-64411626@ssh.runpod.io:/workspace/talklike.llm/3_benchmarking/results ./benchmark_results'
``` 