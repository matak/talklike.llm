# 🎯 Fine-tuning Projekt: Kompletní řešení

## 📋 Přehled projektu

Tento projekt implementuje **kompletní řešení fine-tuningu jazykového modelu** rozdělené na **3 hlavní části** podle zadání úkolu:

1. **[📊 Příprava dat](#1-příprava-dat)** - Vytvoření trénovacích datasetů
2. **[🏋️ Fine-tuning modelu](#2-fine-tuning)** - Doladění jazykového modelu  
3. **[📈 Benchmarking](#3-benchmarking)** - Srovnání před a po fine-tuningu

---

## 🤖 Nahrané modely

### Kompletní fine-tuned model
- **Model**: [mcmatak/mistral-babis-model](https://huggingface.co/mcmatak/mistral-babis-model)
- **Typ**: Kompletní Mistral-7B-Instruct-v0.3 model fine-tuned na Babišův styl
- **Velikost**: ~14GB (34 shardů)
- **Použití**: Přímé použití bez dalších kroků

### LoRA adapter
- **Model**: [mcmatak/mistral-babis-adapter](https://huggingface.co/mcmatak/mistral-babis-adapter)
- **Typ**: LoRA adapter pro Mistral-7B-Instruct-v0.3
- **Velikost**: ~84MB
- **Použití**: Vyžaduje base model + adapter

### Použití modelů

#### Kompletní model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('mcmatak/mistral-babis-model')
tokenizer = AutoTokenizer.from_pretrained('mcmatak/mistral-babis-model')
```

#### LoRA adapter
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
model = PeftModel.from_pretrained(base_model, 'mcmatak/mistral-babis-adapter')
tokenizer = AutoTokenizer.from_pretrained('mcmatak/mistral-babis-adapter')
```

---

## 🎯 Cíle úkolu

### Hlavní cíl
Vytvořit fine-tuned jazykový model, který napodobuje komunikační styl Andreje Babiše, českého politika známého svým charakteristickým způsobem vyjadřování.

### Klíčové požadavky
- **Metoda fine-tuningu**: Implementace Hugging Face + PEFT (LoRA) přístupu
- **Srovnání před/po**: Kompletní benchmarking analýza
- **Kvantitativní evaluace**: Bodový systém hodnocení
- **Forma odevzdání**: Report s tabulkami a screenshoty

### Kritéria úspěchu
- Model generuje odpovědi v charakteristickém Babišově stylu
- Měřitelné zlepšení v napodobování stylu
- Komplexní metriky výkonu
- Reprodukovatelné výsledky

---

## 🏗️ Architektura řešení

### Třídílná implementace

#### [1. Příprava dat](#1-příprava-dat)
- **Umístění**: `1_data_preparation/`
- **README**: [Průvodce přípravou dat](1_data_preparation/README.md)
- **Účel**: Generování kvalitních trénovacích dat v Babišově stylu
- **Výstup**: 1,500 QA párů ve strukturovaném formátu

#### [2. Fine-tuning](#2-fine-tuning)
- **Umístění**: `2_finetunning/`
- **README**: [Průvodce fine-tuningem](2_finetunning/README.md)
- **Účel**: Fine-tuning jazykového modelu pomocí LoRA techniky
- **Výstup**: Fine-tuned model s adaptací stylu

#### [3. Benchmarking](#3-benchmarking)
- **Umístění**: `3_benchmarking/`
- **README**: [Průvodce benchmarkingem](3_benchmarking/README.md)
- **Účel**: Evaluace výkonu modelu před a po fine-tuningu
- **Výstup**: Komplexní srovnávací report

---

## 📊 Očekávané výsledky

### Příprava dat
- ✅ 1,500 QA párů ve strukturovaném formátu
- ✅ Autentický Babišův komunikační styl
- ✅ Moderovaný obsah datasetu
- ✅ Více stylových variací

### Fine-tuning
- ✅ LoRA-adaptovaný model
- ✅ Stylově specifické odpovědi
- ✅ Efektivní využití parametrů
- ✅ Reprodukovatelné trénování

### Benchmarking
- ✅ Kvantitativní metriky výkonu (zlepšení z 1.17 na 4.58/10)
- ✅ Kvalitativní evaluace stylu (15 testovacích otázek)
- ✅ Srovnání před/po s detailními tabulkami
- ✅ Komplexní analytický report s vizualizacemi
- ✅ 4 typy grafů pro různé aspekty evaluace
- ✅ Strukturovaná data pro další analýzu

---

## 📁 Struktura projektu

```
talklike.llm/
├── 1_data_preparation/          # 📊 Příprava dat
│   ├── README.md               # Průvodce přípravou dat
│   ├── QUICKSTART.md           # Rychlý start
│   └── run_data_preparation.py # Hlavní skript
├── 2_finetunning/              # 🏋️ Fine-tuning
│   ├── README.md               # Průvodce fine-tuningem
│   ├── finetune.py             # Hlavní skript
│   └── run_mistral_finetune.sh # Spouštěcí skript
├── 3_benchmarking/             # 📈 Benchmarking
│   ├── README.md               # Průvodce benchmarkingem
│   ├── run_benchmark.py        # Hlavní skript
│   └── results/                # Výsledky
│       ├── reports/            # Markdown reporty
│       ├── visualizations/     # Grafy a vizualizace
│       ├── comparison/         # Data srovnání
│       └── benchmark_dataset.json # Testovací otázky
├── data/                       # 📊 Datasety
│   ├── all.jsonl              # Finální dataset
│   └── final/                 # QA páry
└── lib/                        # 📚 Knihovny
    ├── babis_dataset_generator.py
    └── llm_cost_calculator.py
```

---

## 📈 Benchmarking výsledky

### 📋 Kompletní report
- **[Benchmark Summary Report](3_benchmarking/results/reports/benchmark_summary.md)** - Detailní srovnání modelu před a po fine-tuningu s tabulkami všech otázek

### 📊 Vizualizace výsledků

#### Srovnání skóre
- **[Srovnání stylového skóre](3_benchmarking/results/visualizations/score_comparison.png)** - Graf srovnání průměrného skóre před a po fine-tuningu

#### Zlepšení jednotlivých otázek
- **[Zlepšení stylového skóre](3_benchmarking/results/visualizations/question_improvements.png)** - Graf zlepšení pro každou z 15 testovacích otázek

#### Distribuce známek
- **[Distribuce známek](3_benchmarking/results/visualizations/grade_distribution.png)** - Graf distribuce známek (A-F) před a po fine-tuningu

#### Kategorie stylu
- **[Srovnání kategorií stylu](3_benchmarking/results/visualizations/category_comparison.png)** - Graf srovnání kategorií: Babišovy fráze, slovenské odchylky, emotivní tón, první osoba

### 📄 Detailní data
- **[Model Comparison Data](3_benchmarking/results/comparison/model_comparison.json)** - Strukturovaná data srovnání modelů
- **[Style Evaluation Data](3_benchmarking/results/comparison/style_evaluation.json)** - Detailní evaluace stylu pro každou odpověď
- **[Benchmark Dataset](3_benchmarking/results/benchmark_dataset.json)** - Testovací otázky použité pro benchmarking

### 🎯 Klíčové výsledky
- **Průměrné skóre před fine-tuningem**: 1.17/10
- **Průměrné skóre po fine-tuningem**: 4.58/10
- **Celkové zlepšení**: +3.41 bodů
- **Nejlepší odpověď**: 8.5/10
- **Nejhorší odpověď**: 1.17/10

### 📊 Metriky zlepšení
- **Babišovy fráze**: Výrazné zlepšení v používání charakteristických frází
- **Slovenské odchylky**: Správné použití slovenských odchylek
- **Emotivní tón**: Autentický emotivní tón odpovědí
- **První osoba**: Konzistentní použití první osoby

---

## 🚀 Rychlý start

### 1. Příprava dat
```bash
# Z rootu projektu
python 1_data_preparation/run_data_preparation.py
```

### 2. Fine-tuning
```bash
# Z rootu projektu
./2_finetunning/run_mistral_finetune.sh
```

### 3. Benchmarking
```bash
# Z rootu projektu
python 3_benchmarking/run_benchmark.py
```

### Alternativní spouštění (z adresářů)
```bash
# Příprava dat
cd 1_data_preparation
pip install -r requirements_datapreparation.txt
python run_data_preparation.py

# Fine-tuning
cd 2_finetunning
pip install -r requirements_finetunning.txt
./run_mistral_finetune.sh

# Benchmarking
cd 3_benchmarking
pip install -r requirements_benchmarking.txt
python run_benchmark.py
```
