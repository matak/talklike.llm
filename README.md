# 🎯 Fine-tuning Projekt: Kompletní řešení

## 📋 Přehled projektu

Tento projekt implementuje **kompletní řešení fine-tuningu jazykového modelu** rozdělené na **3 hlavní části** podle zadání úkolu:

1. **[📊 Příprava dat](#1-příprava-dat)** - Vytvoření trénovacích datasetů
2. **[🏋️ Fine-tuning modelu](#2-fine-tuning)** - Doladění jazykového modelu  
3. **[📈 Benchmarking](#3-benchmarking)** - Srovnání před a po fine-tuningu

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
- ✅ Kvantitativní metriky výkonu
- ✅ Kvalitativní evaluace stylu
- ✅ Srovnání před/po
- ✅ Komplexní analytický report

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
│   ├── finetune_babis.py       # Hlavní skript
│   └── run_finetune.sh         # Spouštěcí skript
├── 3_benchmarking/             # 📈 Benchmarking
│   ├── README.md               # Průvodce benchmarkingem
│   ├── run_benchmark.py        # Hlavní skript
│   └── results/                # Výsledky
├── data/                       # 📊 Datasety
│   ├── all.jsonl              # Finální dataset
│   └── final/                 # QA páry
└── lib/                        # 📚 Knihovny
    ├── babis_dataset_generator.py
    └── llm_cost_calculator.py
```

---

## 🚀 Rychlý start

### 1. Příprava dat
```bash
cd 1_data_preparation
pip install -r requirements_datapreparation.txt
python run_data_preparation.py
```

### 2. Fine-tuning
```bash
cd 2_finetunning
pip install -r requirements_finetunning.txt
./run_finetune.sh
```

### 3. Benchmarking
```bash
cd 3_benchmarking
pip install -r requirements_benchmarking.txt
python run_benchmark.py
```

---

## 📚 Dokumentace

- **[📊 Příprava dat](1_data_preparation/README.md)** - Kompletní průvodce přípravou datasetu
- **[🏋️ Fine-tuning](2_finetunning/README.md)** - Detailní návod na fine-tuning
- **[📈 Benchmarking](3_benchmarking/README.md)** - Evaluace a srovnání modelů
- **[🚀 Rychlý start](1_data_preparation/QUICKSTART.md)** - Rychlé spuštění přípravy dat
- **[⚡ Fine-tuning start](2_finetunning/RUNPOD_SETUP.md)** - Nastavení pro RunPod.io

---

## 🎯 Výsledky

Projekt generuje kompletní fine-tuned model s následujícími výstupy:

1. **Dataset**: 1,500 QA párů v Babišově stylu
2. **Model**: Fine-tuned Meta-Llama-3-8B-Instruct s LoRA
3. **Report**: Komplexní benchmarking analýza s tabulkami a grafy
4. **Dokumentace**: Kompletní průvodce pro reprodukci výsledků

Všechny výsledky jsou uloženy v příslušných složkách a jsou připraveny pro odevzdání úkolu.

