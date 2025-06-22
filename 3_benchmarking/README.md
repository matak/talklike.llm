# 📊 Benchmarking - TalkLike.LLM

> **📚 Navigace:** [🏠 Hlavní projekt](../README.md) | [📊 Příprava dat](../1_data_preparation/README.md) | [🏋️ Fine-tuning](../2_finetunning/README.md)

## 📋 Přehled

Tento projekt implementuje **kompletní benchmarking vašeho natrénovaného adaptéru** `mcmatak/babis-mistral-adapter` pro srovnání výkonu před a po fine-tuningu. Benchmarking je zaměřen na evaluaci napodobení komunikačního stylu Andreje Babiše.

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
│   ├── run_benchmark_with_adapter.sh # Automatické spuštění s cache
│   ├── evaluate_style.py             # Evaluace Babišova stylu
│   ├── compare_models.py             # Srovnání před/po
│   ├── generate_responses.py         # Generování odpovědí (INTEGROVÁNO)
│   └── create_benchmark_dataset.py   # Vytvoření testovacích dat
├── 📄 Testovací skripty
│   ├── test_benchmark.py             # Test benchmarkingu
│   ├── test_adapter_integration.py   # Test integrace s adaptérem
│   └── quick_test_adapter.py         # Rychlý test adaptéru
├── 📄 Data a konfigurace
│   ├── benchmark_questions.json      # 15 standardizovaných otázek
│   └── requirements_benchmarking.txt # Dependencies
├── 📄 Dokumentace
│   ├── README.md                     # Tento soubor
│   ├── QUICKSTART.md                 # Rychlý start
│   └── CHANGES_FOR_ADAPTER.md        # Shrnutí změn pro adaptér
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

### 2. Test integrace
```bash
# Rychlý test adaptéru
python quick_test_adapter.py

# Kompletní test integrace
python test_adapter_integration.py
```

### 3. Spuštění benchmarkingu
```bash
# Automatické spuštění s cache nastavením
./run_benchmark_with_adapter.sh

# NEBO manuální spuštění
python run_benchmark.py
```

### 4. Výstupy
- **Excel tabulka**: `results/reports/benchmark_report.xlsx`
- **JSON data**: `results/comparison/style_evaluation.json`
- **Vizualizace**: `results/visualizations/`

---

## 🔧 Nastavení prostředí

### Automatické nastavení
Všechny Python skripty automaticky importují `setup_environment.py` z rootu projektu, který:
- Nastaví cache do `/workspace/.cache/huggingface`
- Vytvoří potřebné adresáře
- Přidá cesty pro import modulů

### Manuální nastavení (volitelné)
```bash
# Spuštění centrálního setup skriptu
python ../setup_environment.py
```

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

### Očekávané zlepšení
- **Celkové skóre**: +6.7 bodů
- **Babišovy fráze**: +2.8 frází/odpověď
- **Slovenské odchylky**: +0.3 slov/odpověď
- **Emotivní tón**: +1.5 výrazů/odpověď

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
# V generate_responses.py - JIŽ IMPLEMENTOVÁNO
def generate_real_response(model, tokenizer, question: str, model_type: str):
    # Používá váš adaptér mcmatak/babis-mistral-adapter
    pass
```

---

## 🛠️ Troubleshooting

### Časté problémy

#### 1. Model se nenačte
```bash
# Zkontrolujte cache
ls -la /workspace/.cache/huggingface/

# Zkuste manuální načtení
python quick_test_adapter.py
```

#### 2. Chyba při generování
```bash
# Zkontrolujte dostupnou paměť
nvidia-smi

# Snižte batch size nebo použijte CPU
export CUDA_VISIBLE_DEVICES=""
```

#### 3. Prázdné výsledky
```bash
# Spusťte test integrace
python test_adapter_integration.py

# Zkontrolujte logy
tail -f results/benchmark.log
```

#### 4. Chybí testovací data
```bash
# Spusťte nejdříve vytvoření datasetu
python create_benchmark_dataset.py
```

#### 5. Chybí odpovědi
```bash
# Vygenerujte odpovědi
python generate_responses.py
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
Odpověď: "Hele, inflace je jak když kráva hraje na klavír! Já makám, ale opozice krade. To je skandál!"
Skóre: 9.2/10 (A)
```

### Zlepšení
- **Celkové skóre**: +6.7 bodů
- **Babišovy fráze**: +2.8 frází/odpověď
- **Slovenské odchylky**: +0.3 slov/odpověď
- **Emotivní tón**: +1.5 výrazů/odpověď

---

## 🎯 Checklist pro odevzdání

- [ ] ✅ Adaptér testován (`python quick_test_adapter.py`)
- [ ] ✅ Benchmarking spuštěn (`./run_benchmark_with_adapter.sh`)
- [ ] ✅ Excel report vygenerován (`results/reports/benchmark_report.xlsx`)
- [ ] ✅ Grafy vytvořeny (`results/visualizations/`)
- [ ] ✅ Shrnutí připraveno (`results/reports/benchmark_summary.txt`)
- [ ] ✅ Screenshoty pořízeny
- [ ] ✅ Výsledky zkontrolovány

---

## 🚀 Rychlé příkazy

```bash
# Test adaptéru
python quick_test_adapter.py

# Kompletní test integrace
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
3. Ověřte dostupnost modelu: `mcmatak/babis-mistral-adapter`

**Benchmarking je připraven pro odevzdání úkolu!** 🎉
