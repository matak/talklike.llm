# 📊 Benchmarking - TalkLike.LLM

> **📚 Navigace:** [🏠 Hlavní projekt](../README.md) | [📊 Příprava dat](../1_data_preparation/README.md) | [🏋️ Fine-tuning](../2_finetunning/README.md)

## 📋 Přehled

Tento projekt implementuje **kompletní benchmarking fine-tuned modelu** pro srovnání výkonu před a po fine-tuningu. Benchmarking je zaměřen na evaluaci napodobení komunikačního stylu Andreje Babiše.

### 🎯 Cíl
- Srovnat odpovědi modelu před/po fine-tuningu
- Vyhodnotit změny pomocí kvantitativních metrik
- Generovat reporty pro odevzdání úkolu
- Poskytnout bodové ohodnocení stylové autenticity

---

## 🏗️ Architektura řešení

### Workflow
1. **Testovací data** → 2. **Generování odpovědí** → 3. **Evaluace stylu** → 4. **Srovnání** → 5. **Reporty**

### Struktura projektu
```
3_benchmarking/
├── 📄 Hlavní skripty
│   ├── run_benchmark.py              # Hlavní benchmarking script
│   ├── evaluate_style.py             # Evaluace Babišova stylu
│   ├── compare_models.py             # Srovnání před/po
│   ├── generate_responses.py         # Generování odpovědí
│   └── create_benchmark_dataset.py   # Vytvoření testovacích dat
├── 📄 Testovací skripty
│   ├── test_benchmark.py             # Test benchmarkingu
│   └── test_evaluation.py            # Test evaluace
├── 📄 Data a konfigurace
│   ├── benchmark_questions.json      # 15 standardizovaných otázek
│   ├── style_evaluation_criteria.json # Kritéria hodnocení
│   └── config.yaml                   # Konfigurace
├── 📄 Prompty
│   ├── LLM.Benchmark.systemPrompt.md # System prompt
│   ├── LLM.EvaluateStyle.systemPrompt.md # Prompt pro evaluaci
│   └── LLM.CompareModels.systemPrompt.md # Prompt pro srovnání
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
pip install -r requirements_benchmarking.txt
```

### 2. Spuštění benchmarkingu
```bash
python run_benchmark.py
```

### 3. Výstupy
- **Excel tabulka**: `results/reports/comparison_table.xlsx`
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

#### 5. Podpis (10%)
- Přítomnost "Andrej Babiš" na konci
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

## 📈 Výstupy pro odevzdání

### 1. Excel tabulka
```
| Metrika                    | Před | Po  | Zlepšení |
|----------------------------|------|-----|----------|
| Průměrná délka odpovědi    | 45.2 | 78.5| +33.3    |
| Babišovy fráze             | 0.2  | 2.8 | +2.6     |
| Slovenské odchylky         | 0.0  | 0.3 | +0.3     |
| Celkové skóre zlepšení     | 0.0  | 5.7 | +5.7     |
```

### 2. Screenshoty
- Porovnání odpovědí před/po
- Grafy zlepšení
- Vizualizace metrik

### 3. PDF report
- Kompletní analýza
- Detailní srovnání
- Závěry a doporučení

---

## 🔧 Pokročilé možnosti

### Vlastní testovací otázky
```json
{
  "id": "Q16",
  "category": "vlastní",
  "question": "Vaše vlastní otázka?",
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

### Integrace s reálnými modely
```python
# V generate_responses.py
def generate_real_responses(model_type: str, output_dir: str):
    # Implementujte volání OpenAI API nebo Hugging Face
    pass
```

---

## 🛠️ Troubleshooting

### Časté problémy

#### 1. Chybí testovací data
```bash
# Spusťte nejdříve vytvoření datasetu
python create_benchmark_dataset.py
```

#### 2. Chybí odpovědi
```bash
# Vygenerujte odpovědi
python generate_responses.py
```

#### 3. Chyba při evaluaci
```bash
# Zkontrolujte formát dat
python test_evaluation.py
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
- **Emotivní tón**: +1.5 výrazů/odpověď
