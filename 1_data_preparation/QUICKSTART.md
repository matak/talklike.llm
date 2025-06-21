# 🚀 Rychlý start - Příprava dat

Tento průvodce vás provede rychlým spuštěním přípravy dat pro fine-tuning jazykového modelu.

## ⚡ Rychlé spuštění (3 kroky)

### 1. Instalace závislostí
```bash
pip install -r requirements_datapreparation.txt
```

### 2. Nastavení OpenAI API
```bash
# Vytvořte .env soubor
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### 3. Spuštění kompletní přípravy
```bash
python run_data_preparation.py
```

## 🧪 Testování před spuštěním

Pokud chcete ověřit, že vše je připraveno:

```bash
python test_data_preparation.py
```

## 📊 Co se stane

1. **Kontrola prostředí** - Ověření závislostí a konfigurace
2. **Generování odpovědí** - Vytvoření 1,500 odpovědí z šablon
3. **Vytvoření QA párů** - Generování otázek k odpovědím
4. **Sloučení dat** - Vytvoření finálního datasetu
5. **Kontrola kvality** - Analýza a report kvality

## 📁 Výstupy

Po úspěšném spuštění budete mít:

- `data/all.jsonl` - Finální dataset pro fine-tuning
- `data_quality_report.json` - Report kvality dat
- `data_quality_analysis.png` - Vizualizace analýzy
- `logs/` - Detailní logy procesu

## 🚨 Řešení problémů

### Chyba: "OPENAI_API_KEY není nastaven"
```bash
export OPENAI_API_KEY="your_key_here"
```

### Chyba: "Chybí soubory"
Zkontrolujte, že máte všechny požadované soubory:
- `babis_templates_400.json`
- `LLM.CreateAnswers.systemPrompt.md`
- `LLM.CreateDialogue.systemPrompt.md`
- `availablemodels.json`

### Chyba: "Nedostatečné kredity"
- Zkontrolujte zůstatek na OpenAI
- Použijte levnější model v `availablemodels.json`

## 📞 Podpora

Pro detailní informace viz:
- [README.md](README.md) - Kompletní dokumentace
- [test_data_preparation.py](test_data_preparation.py) - Testy funkčnosti

---

**Poznámka**: Pro spuštění potřebujete OpenAI API klíč s dostatečnými kredity. 