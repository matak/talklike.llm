# 🎯 Fine-tuning Projekt: Kompletní řešení

## 📋 Přehled projektu

Tento projekt implementuje **kompletní řešení fine-tuningu jazykového modelu** rozdělené na **3 hlavní části** podle zadání úkolu:

1. **[📊 Příprava dat](#data-preparation)** - Vytvoření trénovacích datasetů
2. **[🏋️ Fine-tuning modelu](#fine-tuning)** - Doladění jazykového modelu  
3. **[📈 Benchmarking](#benchmarking)** - Srovnání před a po fine-tuningu

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

#### [1. Příprava dat](#data-preparation)
- **Umístění**: `1_data_preparation/`
- **README**: [Průvodce přípravou dat](1_data_preparation/README.md)
- **Účel**: Generování kvalitních trénovacích dat v Babišově stylu
- **Výstup**: 1,500 QA párů ve strukturovaném formátu

#### [2. Fine-tuning](#fine-tuning)
- **Umístění**: `2_finetunning/`
- **README**: [Průvodce fine-tuningem](2_finetunning/README.md)
- **Účel**: Fine-tuning jazykového modelu pomocí LoRA techniky
- **Výstup**: Fine-tuned model s adaptací stylu

#### [3. Benchmarking](#benchmarking)
- **Umístění**: `3_benchmarking/` (bude vytvořeno)
- **README**: [Průvodce benchmarkingem](3_benchmarking/README.md) (bude vytvořeno)
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

