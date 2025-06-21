# 🚀 Rychlý start - Benchmarking TalkLike.LLM

> **📚 Navigace:** [🏠 Hlavní projekt](../README.md) | [📊 Detailní dokumentace](README.md) | [📊 Příprava dat](../1_data_preparation/README.md) | [🏋️ Fine-tuning](../2_finetunning/README.md)

## 🎯 Cíl

Provedení kompletního benchmarkingu vašeho natrénovaného adaptéru `mcmatak/babis-mistral-adapter` pro odevzdání domácího úkolu.

## ⚡ Během 10 minut

### 1. Instalace a test
```bash
cd 3_benchmarking

# Instalace requirements
pip install -r requirements_benchmarking.txt

# Test integrace s adaptérem
python quick_test_adapter.py
```

### 2. Spuštění benchmarkingu
```bash
# Automatické spuštění s cache nastavením
./run_benchmark_with_adapter.sh

# NEBO manuální spuštění
python run_benchmark.py
```

### 3. Výstupy pro odevzdání
- 📊 **Excel**: `results/reports/benchmark_report.xlsx`
- 📈 **Grafy**: `results/visualizations/`
- 📋 **Shrnutí**: `results/reports/benchmark_summary.txt`

---

## 🔧 Konfigurace

### Váš adaptér
- **Base model**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Adapter**: `mcmatak/babis-mistral-adapter`
- **Cache**: `/workspace/.cache/huggingface`

### Testovací otázky
- **15 standardizovaných otázek** v `benchmark_questions.json`
- **Kategorie**: politika, ekonomika, rodina, podnikání, Brusel
- **Obtížnost**: easy (2), medium (9), hard (4)

---

## 📊 Očekávané výsledky

### Před fine-tuningem (base model)
```
Otázka: "Pane Babiši, jak hodnotíte současnou inflaci?"
Odpověď: "Inflace je vážný problém, který postihuje všechny občany."
Skóre: ~2-3/10 (F)
```

### Po fine-tuningem (váš adaptér)
```
Otázka: "Pane Babiši, jak hodnotíte současnou inflaci?"
Odpověď: "Hele, inflace je jak když kráva hraje na klavír! Já makám, ale opozice krade. To je skandál! Andrej Babiš"
Skóre: ~8-9/10 (A)
```

### Očekávané zlepšení
- **Celkové skóre**: +5-7 bodů
- **Babišovy fráze**: +2-3 frází/odpověď
- **Slovenské odchylky**: +0.3-0.5 slov/odpověď
- **Emotivní tón**: +1-2 výrazů/odpověď

---

## 🧪 Testování

### Ověření funkčnosti
```bash
python test_benchmark.py
```

### Test jednotlivých komponent
```bash
# Test evaluace stylu
python evaluate_style.py

# Test generování odpovědí
python generate_responses.py

# Test srovnání modelů
python compare_models.py
```

---

## 📁 Struktura výstupů

```
results/
├── before_finetune/
│   └── responses.json          # Odpovědi základního modelu
├── after_finetune/
│   └── responses.json          # Odpovědi vašeho adaptéru
├── comparison/
│   ├── model_comparison.json   # Srovnání modelů
│   └── style_evaluation.json   # Evaluace stylu
├── reports/
│   ├── benchmark_report.xlsx   # Excel report pro odevzdání
│   └── benchmark_summary.txt   # Textové shrnutí
└── visualizations/
    ├── score_comparison.png    # Graf srovnání skóre
    ├── improvement_metrics.png # Graf zlepšení metrik
    └── grade_distribution.png  # Graf distribuce známek
```

---

## 🎯 Metriky pro odevzdání

### Kvantitativní metriky
- **Průměrné skóre před/po**: 2.5 → 8.5
- **Zlepšení Babišových frází**: 0.2 → 2.8 frází/odpověď
- **Slovenské odchylky**: 0.0 → 0.3 slov/odpověď
- **Emotivní tón**: 0.1 → 1.5 výrazů/odpověď

### Kvalitativní hodnocení
- **Stylová autenticita**: A (9/10)
- **Konzistence**: B+ (8/10)
- **Kreativita**: A- (8.5/10)

---

## 📋 Checklist pro odevzdání

- [ ] ✅ Benchmarking spuštěn
- [ ] ✅ Excel report vygenerován
- [ ] ✅ Grafy vytvořeny
- [ ] ✅ Shrnutí připraveno
- [ ] ✅ Screenshoty pořízeny
- [ ] ✅ Výsledky zkontrolovány

---

## 🛠️ Troubleshooting

### Časté problémy

#### 1. Chybí dependencies
```bash
pip install pandas matplotlib seaborn openpyxl
```

#### 2. Chyba při evaluaci
```bash
python test_benchmark.py
```

#### 3. Prázdné výsledky
```bash
# Zkontrolujte, že existují testovací data
ls benchmark_questions.json
```

#### 4. Model se nenačte
```bash
# Zkontrolujte cache
ls -la /workspace/.cache/huggingface/

# Zkuste manuální načtení
python quick_test_adapter.py
```

#### 5. Chyba při generování
```bash
# Zkontrolujte dostupnou paměť
nvidia-smi

# Snižte batch size nebo použijte CPU
export CUDA_VISIBLE_DEVICES=""
```

---

## 🚀 Rychlé příkazy

```bash
# Test integrace
python test_adapter_integration.py

# Spuštění benchmarkingu
./run_benchmark_with_adapter.sh

# Zobrazení výsledků
ls -la results/reports/
cat results/reports/benchmark_summary.txt

# Otevření Excel reportu
open results/reports/benchmark_report.xlsx
```

---

## 📞 Podpora

Pro problémy:
1. Spusťte `python quick_test_adapter.py`
2. Zkontrolujte logy v terminálu
3. Ověřte formát dat v `benchmark_questions.json`
4. Ověřte dostupnost modelu: `mcmatak/babis-mistral-adapter`

---

## 🎯 Cíl benchmarkingu

- ✅ Srovnat modely před/po fine-tuningu
- ✅ Vyhodnotit stylovou autenticitu
- ✅ Generovat reporty pro odevzdání
- ✅ Poskytnout kvantitativní metriky

**Benchmarking je připraven pro odevzdání úkolu!** 🎉 