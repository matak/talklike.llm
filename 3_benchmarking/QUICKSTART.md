# 🚀 Rychlý start - Benchmarking TalkLike.LLM

> **📚 Navigace:** [🏠 Hlavní projekt](../README.md) | [📊 Detailní dokumentace](README.md) | [📊 Příprava dat](../1_data_preparation/README.md) | [🏋️ Fine-tuning](../2_finetunning/README.md)

## 🎯 Cíl

## ⚡ Během 5 minut

### 1. Instalace
```bash
cd 3_benchmarking
pip install -r requirements_benchmarking.txt
```

### 2. Spuštění
```bash
python run_benchmark.py
```

### 3. Výstupy
- 📊 **Excel**: `results/reports/benchmark_report.xlsx`
- 📈 **Grafy**: `results/visualizations/`
- 📋 **Shrnutí**: `results/reports/benchmark_summary.txt`

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

## 📊 Příklad výsledků

### Před fine-tuningem
```
Otázka: "Pane Babiši, jak hodnotíte současnou inflaci?"
Odpověď: "Inflace je vážný problém, který postihuje všechny občany."
Skóre: 2.5/10 (F)
```

### Po fine-tuningem
```
Otázka: "Pane Babiši, jak hodnotíte současnou inflaci?"
Odpověď: "Hele, inflace je jak když kráva hraje na klavír! Já makám, ale opozice krade. To je skandál! Andrej Babiš"
Skóre: 9.2/10 (A)
```

### Zlepšení
- **Celkové skóre**: +6.7 bodů
- **Babišovy fráze**: +2.8 frází/odpověď
- **Slovenské odchylky**: +0.3 slov/odpověď

---

## 🔧 Pokročilé možnosti

### Vlastní testovací otázky
```json
{
  "id": "Q16",
  "category": "vlastní",
  "question": "Vaše otázka?",
  "expected_style_elements": ["hele", "skandál"],
  "difficulty": "medium"
}
```

### Úprava kritérií hodnocení
```python
# V evaluate_style.py
self.babis_phrases = [
    "hele", "skandál", "makám", 
    # Přidejte vlastní fráze
]
```

---

## 📁 Struktura výstupů

```
results/
├── before_finetune/
│   └── responses.json          # Odpovědi před fine-tuningem
├── after_finetune/
│   └── responses.json          # Odpovědi po fine-tuningem
├── comparison/
│   ├── model_comparison.json   # Srovnání modelů
│   └── style_evaluation.json   # Evaluace stylu
├── reports/
│   ├── benchmark_report.xlsx   # Excel report
│   └── benchmark_summary.txt   # Textové shrnutí
└── visualizations/
    ├── score_comparison.png    # Graf srovnání skóre
    ├── improvement_metrics.png # Graf zlepšení metrik
    └── grade_distribution.png  # Graf distribuce známek
```

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

---

## 📞 Podpora

Pro problémy:
1. Spusťte `python test_benchmark.py`
2. Zkontrolujte logy v terminálu
3. Ověřte formát dat v `benchmark_questions.json`

---

## 🎯 Cíl benchmarkingu

- ✅ Srovnat modely před/po fine-tuningu
- ✅ Vyhodnotit stylovou autenticitu
- ✅ Generovat reporty pro odevzdání
- ✅ Poskytnout kvantitativní metriky

**Benchmarking je připraven pro odevzdání úkolu!** 🎉 