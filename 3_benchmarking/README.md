# 📊 Benchmarking - TalkLike.LLM

> **📚 Navigace:** [🏠 Hlavní projekt](../README.md) | [📊 Příprava dat](../1_data_preparation/README.md) | [🏋️ Fine-tuning](../2_finetunning/README.md)

## 📋 Přehled

Tento projekt implementuje **kompletní benchmarking vašeho natrénovaného modelu** `mcmatak/mistral-babis-model` pro srovnání výkonu před a po fine-tuningu. Benchmarking je zaměřen na evaluaci napodobení komunikačního stylu Andreje Babiše.

### 🎯 Cíl
- Srovnat odpovědi modelu před/po fine-tuningu
- Vyhodnotit změny pomocí kvantitativních metrik
- Generovat reporty pro odevzdání úkolu
- Poskytnout bodové ohodnocení stylové autenticity

---

## 🏗️ Architektura řešení

### Workflow
1. **Testovací data** → 2. **Generování odpovědí** → 3. **Evaluace stylu** → 4. **Srovnání** → 5. **Reporty**

### Váš adaptér
- **Base model**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Adapter**: `mcmatak/babis-mistral-adapter`
- **Cache**: `/workspace/.cache/huggingface`
- **System prompt**: Optimalizovaný pro Babišův styl

### Struktura projektu
```
3_benchmarking/
├── 📄 Hlavní skripty
│   ├── run_benchmark.py              # Hlavní benchmarking script
│   ├── evaluate_style.py             # Evaluace Babišova stylu
│   ├── compare_models.py             # Srovnání před/po
│   ├── generate_responses.py         # Generování odpovědí (INTEGROVÁNO)
│   └── create_benchmark_dataset.py   # Vytvoření testovacích dat
├── 📄 Data a konfigurace
│   ├── benchmark_questions.json      # 15 standardizovaných otázek
│   └── requirements_benchmarking.txt # Dependencies
├── 📄 Dokumentace
│   └── README.md                     # Tento soubor
└── 📄 Výstupy (results/)
    ├── before_finetune/              # Odpovědi před fine-tuningem
    ├── after_finetune/               # Odpovědi po fine-tuningem
    ├── comparison/                   # Srovnávací analýzy
    ├── reports/                      # Excel, PDF reporty
    └── visualizations/               # Grafy a vizualizace
```

---

## 🚀 Rychlé spuštění

### 1. Instalace
```bash
cd 3_benchmarking
pip install -r requirements_benchmarking.txt
```

### 2. Spuštění benchmarkingu
```bash
# manuální spuštění
python run_benchmark.py
```

### 4. Výstupy
- **Tabulka v markdown**: `results/reports/benchmark_report.md`
- **JSON data**: `results/comparison/style_evaluation.json`
- **Vizualizace**: `results/visualizations/`

---

## 📊 Metriky evaluace

### Bodový systém (0-10 bodů)

#### 1. Babišovy fráze (30%)
- "Hele", "To je skandál!", "Já makám"
- "Opozice krade", "V Bruselu", "Moje rodina"
- **Hodnocení**: Počet nalezených frází / očekávaný počet

#### 2. Slovenské odchylky (20%)
- "sme", "som", "makáme", "centralizácia"
- **Hodnocení**: Počet slovenských slov / očekávaný počet

#### 3. Emotivní tón (25%)
- "šílený", "tragédyje", "kampááň", "hrozné"
- **Hodnocení**: Počet emotivních výrazů / očekávaný počet

#### 4. První osoba (15%)
- "já", "moje", "jsem", "makám", "budoval"
- **Hodnocení**: Počet indikátorů první osoby / očekávaný počet

#### 5. Přítomnost charakteristických frází (10%)
- **Hodnocení**: 10 bodů za přítomnost, 0 za absenci

#### Bonus: Přirovnání (+1 bod za každé)
- "jak když kráva hraje na klavír"
- "jak když dítě řídí tank"
- "jak když slepice hraje šachy"

---

## 📋 Testovací otázky

### Kategorie otázek (15 celkem)
- **Politika**: inflace, opozice, parlament
- **Ekonomika**: daně, důchody, efektivizace
- **Brusel**: evropské instituce, regulace
- **Rodina**: osobní život, soukromí
- **Podnikání**: továrna, projekty, Agrofert
- **Média**: novináři, kritika, kampáň

### Obtížnost
- **Easy**: 2 otázky (základní témata)
- **Medium**: 9 otázek (střední obtížnost)
- **Hard**: 4 otázky (komplexní témata)

---

## 📈 Očekávané výsledky

### Před fine-tuningem (base model)
```
Otázka: "Pane Babiši, jak hodnotíte současnou inflaci?"
Odpověď: "Inflace je vážný problém, který postihuje všechny občany."
Skóre: 2.5/10 (F)
```

### Po fine-tuningem (váš adaptér)
```
Otázka: "Pane Babiši, jak hodnotíte současnou inflaci?"
Odpověď: "Hele, inflace je jak když kráva hraje na klavír! Já makám, ale opozice krade. To je skandál!"
Skóre: 9.2/10 (A)
```
