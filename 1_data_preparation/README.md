# 🤖 Příprava dat pro Fine-tuning - TalkLike.LLM

> **📚 Navigace:** [🏠 Hlavní projekt](../README.md) | [🏋️ Fine-tuning](../2_finetunning/README.md) | [📊 Benchmarking](../3_benchmarking/README.md)

## 📋 Přehled

Tento projekt vytváří dataset pro fine-tuning jazykového modelu, který napodobuje komunikační styl Andreje Babiše. Vytvořený dataset obsahuje 1,500 QA párů ve strukturovaném formátu s charakteristickým stylem "babíšovštiny".

### 🔄 Vývoj metodologie

#### Počáteční přístup a jeho problémy
**První pokus:** Ruční sběr dat
- Stáhnuty všechny projevy Andreje Babiše z poslanecké sněmovny
- Získány rozhovory z období působení na ministerstvu financí
- Shromážděny články z bulvárních plátků (Parlamentní listy)
- **Problém:** Mix satirických výroků a předpřipravených textů byl nevyrovnaný, konzistentnost názorů byla problematická (data z roku 2013, 2017, 2021 jdou názorově proti sobě)
- **Výsledek:** Ruční třídění se stalo neefektivní a časově náročné

#### Přechod k LLM-based metodě
**Důvod změny:** LLM nedokáží zpracovat složitější zadání najednou
**Řešení:** Rozklad procesu na menší, detailní kroky
- Simplifikace zadání
- Rozklad na tvorbu šablon a následné generování datasetu
- Postupné zpracování v dávkách

## 🎯 Cíl

Vytvořit kvalitní dataset pro fine-tuning, který zachovává:
- Mluvenou češtinu se slovensko-českými odchylkami
- Specifické rétorické prvky a fráze
- Satirický, emotivní a sebestředný tón
- Autentický styl "babíšovštiny"

## 🏗️ Architektura řešení

### Workflow
1. **Šablony** → 2. **Odpovědi** → 3. **QA páry** → 4. **Sloučení** → 5. **Validace**

### Struktura projektu
```
1_data_preparation/
├── 📄 Skripty
│   ├── generate_answers.py          # Generování odpovědí z šablon
│   ├── generate_qa_dataset.py       # Vytvoření QA párů
│   ├── dataset_merger.py           # Sloučení dat
│   ├── data_quality_check.py       # Kontrola kvality
│   ├── moderate_training_data.py   # Moderace obsahu
│   └── run_data_preparation.py     # Hlavní runner
├── 📄 Šablony a prompty
│   ├── babis_templates_400.json    # 400 šablon výroků
│   ├── LLM.CreateAnswers.systemPrompt.md
│   ├── LLM.CreateDialogue.systemPrompt.md
│   └── LLM.finetunning.systemPrompt.json
├── 📄 Knihovny (lib/)
│   ├── babis_dataset_generator.py
│   ├── babis_dialog_generator.py
│   ├── openai_cost_calculator.py
│   └── llm_cost_calculator.py
└── 📄 Výstupy (data/)
    ├── generated_batches/          # Mezivýstupy
    ├── final/                      # QA páry
    └── all.jsonl                   # Finální dataset
```

## 🚀 Rychlé spuštění

### 1. Instalace
```bash
pip install -r requirements_datapreparation.txt
```

### 2. Nastavení
```bash
# Vytvořte .env soubor s OpenAI API klíčem
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### 3. Spuštění
```bash
python run_data_preparation.py
```

## 📋 Detailní kroky

### Krok 1: Vytvoření šablon
**Soubor:** `LLM.Outline.CreateTemplates.md`
- **Úkol:** Vygenerovat 400 originálních šablon výroků
- **Model:** GPT-o3
- **Výstup:** Šablony s placeholdery pro různé kategorie

**Klíčové placeholdery:**
- `{tema}` - inflace, válka, důchody, klima, očkování, daně
- `{nepritel}` - Brusel, Piráti, Fiala, novináři, opozice
- `{cinnost}` - makal, pomáhal lidem, budoval stát, rušil poplatky
- `{kritika}` - kradou, sabotujou nás, jenom řeční
- `{emotivni_vyraz}` - to je šílený!, kampááň!, tragédyje!

### Krok 2: Generování odpovědí
```bash
python generate_answers.py
```
- **Vstup:** 400 šablon z `babis_templates_400.json`
- **Výstup:** 10 dávek po 150 odpovědích (celkem 1,500)
- **Pravidla:** 15% pravděpodobnost jazykové chyby, mix 5 stylů

### Krok 3: Vytvoření QA datasetu
```bash
python generate_qa_dataset.py
```
- **Vstup:** Odpovědi z kroku 2
- **Výstup:** QA páry v JSONL formátu

### Krok 4: Sloučení dat
```bash
python dataset_merger.py
```
- **Vstup:** QA páry z kroku 3
- **Výstup:** `data/all.jsonl` - finální dataset

### Krok 5: Kontrola kvality
```bash
python data_quality_check.py
```
- **Vstup:** Finální dataset
- **Výstup:** Report a vizualizace kvality

## 📊 Charakteristika datasetu

### Struktura dat
```json
{
  "messages": [
    {
      "role": "system",
      "content": "Jsi Andrej Babiš, český politik a podnikatel..."
    },
    {
      "role": "user", 
      "content": "Pane Babiši, jak hodnotíte současnou inflaci?"
    },
    {
      "role": "assistant",
      "content": "Hele, inflace je jak když kráva hraje na klavír!"
    }
  ]
}
```

### Klíčové vlastnosti
- **Počet QA párů:** 1,500
- **Styl:** Mluvená čeština s "babíšovštinou"
- **Jazykové chyby:** 15% pravděpodobnost slovenských odchylek
- **Stylové variace:** 5 různých stylů

### Charakteristické prvky
- **Fráze:** "Hele", "To je skandál!", "Já makám"
- **Přirovnání:** "jak když kráva hraje na klavír"
- **Slovenské odchylky:** "sme", "som", "makáme"
- **Témata:** Politika, ekonomika, rodina, podnikání

## 💰 Náklady a optimalizace

### Kalkulátor nákladů
```python
from lib.openai_cost_calculator import OpenAICostCalculator

calculator = OpenAICostCalculator()
cost = calculator.estimate_batch_cost(input_text, output_text, "gpt-4o")
```

### Optimalizace
- **Dávkové zpracování:** Po 150 šablonách
- **Cachování:** Ukládání surových odpovědí
- **Validace:** Kontrola před uložením
- **Logování:** Detailní sledování procesu

## 🔍 Kontrola kvality

### Automatické kontroly
- ✅ Struktura konverzací
- ✅ Stylové prvky

### Report kvality
- **Soubor:** `data_quality_report.json`
- **Vizualizace:** `data_quality_analysis.png`
- **Metriky:** Úspěšnost, distribuce stylů, délky

## 🛠️ Pokročilé možnosti

### Vlastní šablony
```json
{
  "tema": ["inflace", "válka", "důchody"],
  "nepritel": ["Brusel", "Piráti", "Fiala"],
  "cinnost": ["makal", "pomáhal lidem", "budoval stát"]
}
```

### Konfigurace modelů
```json
{
  "gpt-4o": {
    "name": "GPT-4 Optimized",
    "default": 1,
    "prices": {
      "batch": {"input": 1.25, "output": 5.00}
    }
  }
}
```

## 🚨 Řešení problémů

### Časté chyby
1. **OPENAI_API_KEY není nastaven**
   ```bash
   export OPENAI_API_KEY="your_key_here"
   ```

2. **Chybějící soubory**
   ```bash
   ls babis_templates_400.json
   ls LLM.CreateAnswers.systemPrompt.md
   ```

3. **Nedostatečné kredity**
   - Zkontrolujte zůstatek na OpenAI
   - Použijte levnější model v `availablemodels.json`

### Debugování
```bash
# Spuštění s detailním výpisem
python -u generate_answers.py

# Kontrola logů
tail -f logs/llm_interaction_*.log
```

## 📈 Výsledky a monitoring

### Finální soubory
- `data/all.jsonl` - Dataset pro fine-tuning
- `data_quality_report.json` - Report kvality
- `data_quality_analysis.png` - Vizualizace
- `logs/data_preparation_*.json` - Logy

### Metriky kvality
- **Úspěšnost podpisů:** >95%
- **Slovenské odchylky:** 10-20%
- **Charakteristické fráze:** >80%
- **Průměrná délka:** 50-150 znaků

### Logy a monitoring
- **Umístění:** `logs/`
- **Formát:** JSON s timestampem
- **Obsah:** Prompt, odpověď, chyby, náklady

## 🔄 Údržba a aktualizace

### Přidání nových šablon
1. Upravte `babis_templates_400.json`
2. Spusťte `generate_answers.py`
3. Ověřte kvalitu pomocí `data_quality_check.py`

### Úprava stylu
1. Upravte systémové prompty
2. Regenerujte dataset
3. Porovnejte kvalitu

### Optimalizace nákladů
1. Zkontrolujte `availablemodels.json`
2. Upravte velikost dávek
3. Použijte levnější modely

## ⚠️ Důležité poznámky

### Moderace obsahu
Dataset byl zablokován OpenAI moderací kvůli satirickému obsahu. Chyba:
> "The job failed due to an invalid validation file. This training file was blocked by our moderation system because it contains too many examples that violate OpenAI's usage policies"

### Alternativní řešení
- Použití lokálních modelů pro fine-tuning
- Úprava obsahu pro splnění moderace
- Použití alternativních platform

## ✅ Shrnutí

Projekt úspěšně demonstroval kompletní workflow pro vytvoření fine-tuning datasetu pomocí LLM. Klíčovým poznatkem bylo, že složité zadání je nutné rozdělit na menší, detailní kroky.

**Silné stránky:**
- 🎯 Zaměření na specifický styl
- 💰 Optimalizované náklady
- 🔍 Kompletní validace
- 📊 Detailní monitoring
- 🛠️ Snadná údržba a rozšiřitelnost

**Oblasti pro zlepšení:**
- Řešení moderace obsahu
- Alternativní přístupy k fine-tuningu
