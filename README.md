# Praktické cvičení - Lekce 6: Fine-tuning jazykového modelu
## [AI Developer]

**Počet bodů:** 100  
**Deadline:** 17.6.2025

---

## 📋 Zadání úkolu

Vyberte si metodu fine-tuningu modelu podle svého zamýšleného použití – buď pomocí OpenAI API (např. doladění GPT-4) nebo s využitím Hugging Face (např. transformers a PEFT). Proveďte srovnání odpovědí modelu před a po fine-tuningu a vyhodnoťte změny pomocí benchmarkingu.

**Forma odevzdání:** Vypracovaný úkol odevzdejte ve formě tabulky (excel, Google sheet), screenshotů nebo PDF souboru obsahujícího všechny odpovědi a vaše bodové ohodnocení (benchmarking). Nahrajte soubor na Google Classroom.

---

## 🎯 Cíl projektu

Cílem projektu bylo vytvořit fine-tuning dataset pro jazykový model, který by napodoboval charakteristický styl komunikace Andreje Babiše - českého politika a podnikatele. Model měl být schopen generovat satirické odpovědi ve stylu "babíšovštiny" - charakteristického jazykového stylu s mluvenou češtinou, slovensko-českými odchylkami a specifickými rétorickými prvky.

---

## 🛠️ Disk Manager Library

Projekt obsahuje vlastní knihovnu pro správu diskového prostoru, která byla vyvinuta během řešení problémů s místem na disku při fine-tuningu velkých modelů.

### Instalace
```bash
cd disk_manager
pip install -e .
```

### Použití
```python
from disk_manager import DiskManager

dm = DiskManager()
if dm.check_disk_space():
    dm.cleanup_cache()
    dm.setup_network_storage("/workspace")
```

### Funkce
- **Kontrola místa na disku** - automatická detekce plného filesystem
- **Čištění cache** - odstranění HuggingFace, pip, conda cache
- **Network storage** - nastavení cache na externí úložiště
- **Optimalizace pro velké modely** - automatická příprava pro Mistral/Llama

Více informací najdete v [disk_manager/README.md](disk_manager/README.md).

---

## 🔄 Metodologie a implementace

### 1. Počáteční přístup a jeho problémy

**První pokus:** Ruční sběr dat
- Stáhnuty všechny projevy Andreje Babiše z poslanecké sněmovny
- Získány rozhovory z období působení na ministerstvu financí
- Shromážděny články z bulvárních plátků (Parlamentní listy)
- **Problém:** Mix satirických výroků a předpřipravených textů byl nevyrovnaný
- **Výsledek:** Ruční třídění se stalo neefektivní a časově náročné

### 2. Přechod k LLM-based metodě

**Důvod změny:** LLM nedokáží zpracovat složitější zadání najednou
**Řešení:** Rozklad procesu na menší, detailní kroky
- Simplifikace zadání
- Rozklad na tvorbu šablon a následné generování datasetu
- Postupné zpracování v dávkách

---

## 🛠️ Implementační kroky

### Krok 1: Vytvoření šablon (Templates)

**Soubor:** `TASK/LLM.Outline.CreateTemplates.md`
- **Úkol:** Vygenerovat 400 originálních šablon výroků ve stylu Andreje Babiše
- **Model:** GPT-o3 (GPT-4.5 se ukázal jako nevhodný - nepochopil zadání)
- **Zajímavost:** GPT-o3 si vytvořil Python skripty pro brute-force skládání slov
- **Výstup:** Šablony s placeholdery pro různé kategorie (téma, nepřítel, činnost, kritika, atd.)

**Klíčové placeholdery:**
- `{tema}` - inflace, válka, důchody, klima, očkování, daně
- `{nepritel}` - Brusel, Piráti, Fiala, novináři, opozice
- `{cinnost}` - makal, pomáhal lidem, budoval stát, rušil poplatky
- `{kritika}` - kradou, sabotujou nás, jenom řeční
- `{emotivni_vyraz}` - to je šílený!, kampááň!, tragédyje!

### Krok 2: Generování odpovědí

**Soubor:** `TASK/LLM.CreateAnswers.systemPrompt.md`
- **Úkol:** Nahradit placeholdery v šablonách konkrétními hodnotami
- **Zpracování:** Po 300 šablonách v dávkách
- **Pravidla:** 
  - 15% pravděpodobnost jazykové chyby
  - Rovnoměrný mix 5 stylů (emocionální výlevy, odmítavý postoj, domýšlivost, chaotická logika, ironie)
  - Zachování autentického stylu "babíšovštiny"

### Krok 3: Vytvoření dialogů

**Soubor:** `TASK/LLM.CreateDialogue.systemPrompt.md`
- **Úkol:** Generovat korespondující novinářské otázky k odpovědím
- **Formát:** JSONL s páry otázka-odpověď
- **Styl:** Profesionální redaktor v rozhovoru

### Krok 4: Automatizované zpracování

**Skript:** `generate_qa_dataset.py`
- **Funkce:** Zpracování všech batch souborů
- **Výstup:** Strukturované QA páry pro fine-tuning
- **Kontrola:** Výpočet nákladů pomocí tiktokenizer

**Skript:** `generate_answers.py`
- **Funkce:** Generování odpovědí z šablon
- **Optimalizace:** Dávkové zpracování pro úsporu nákladů

### Krok 5: Sloučení dat

**Skript:** `dataset_merger.py`
- **Úkol:** Sloučit všechny vygenerované dávky
- **Výstup:** `data/all.jsonl` - kompletní dataset

### Krok 6: Moderace obsahu

**Skript:** `moderate_training_data.py`
- **Úkol:** Kontrola obsahu pomocí OpenAI Moderation API
- **Endpoint:** `https://api.openai.com/v1/moderations`
- **Problém:** Dataset byl zablokován moderací
- **Chyba:** "The job failed due to an invalid validation file. This training file was blocked by our moderation system because it contains too many examples that violate OpenAI's usage policies"

---

## 📊 Struktura projektu

### Adresáře a soubory:

```
talklike.llm/
├── disk_manager/                   # Disk Manager Library
│   ├── __init__.py
│   ├── core.py                     # Hlavní funkcionalita
│   ├── setup.py                    # Setup pro instalaci
│   ├── pyproject.toml              # Moderní Python packaging
│   ├── README.md                   # Dokumentace knihovny
│   └── tests/                      # Testy
│       ├── __init__.py
│       └── test_core.py
├── TASK/                          # Zadání a šablony
│   ├── babis_templates_400.json   # Vygenerované šablony
│   ├── LLM.Outline.CreateTemplates.md
│   ├── LLM.CreateAnswers.systemPrompt.md
│   ├── LLM.CreateDialogue.systemPrompt.md
│   └── LLM.finetunning.systemPrompt.json
├── generated_batches/             # Mezivýstupy
│   ├── batch_01_babis_output.jsonl
│   ├── content/                   # Obsah dávek
│   ├── responses/                 # Odpovědi LLM
│   └── invalid/                   # Neplatné výstupy
├── final/                         # Finální QA dataset
│   ├── batch_01_babis_output_qa.jsonl
│   └── ...
├── data/                          # Sloučená data
│   ├── all.jsonl                  # Kompletní dataset
│   ├── moderated_training_data.jsonl
│   └── moderated_training_data_report.txt
└── Skripty pro zpracování
    ├── generate_qa_dataset.py
    ├── generate_answers.py
    ├── dataset_merger.py
    ├── moderate_training_data.py
    ├── llm_cost_calculator.py
    └── finetune_babis.py          # Fine-tuning s disk manager
```

---

## 💰 Náklady a optimalizace

### Výpočet nákladů:
- **Nástroj:** `llm_cost_calculator.py` a `openai_cost_calculator.py`
- **Metoda:** Použití tiktokenizer pro přesný výpočet tokenů
- **Optimalizace:** Dávkové zpracování pro minimalizaci nákladů

### Logy a monitoring:
- **Adresář:** `logs/` - detailní logy všech operací
- **Soubor:** `moderation.log` - výsledky moderace

---

## 🚧 Problémy a řešení

### 1. Problém s moderací
**Problém:** Dataset byl zablokován OpenAI moderací
**Příčina:** Satirický obsah byl vyhodnocen jako porušení pravidel
**Řešení:** Implementace `moderate_training_data.py` pro předběžnou kontrolu

### 2. Komplexita zadání pro LLM
**Problém:** LLM nedokáží zpracovat složité zadání najednou
**Řešení:** Rozklad na menší kroky s detailními instrukcemi

### 3. Nevyrovnanost dat
**Problém:** Mix satirických a oficiálních textů
**Řešení:** Přechod k syntetickému generování pomocí šablon

### 4. Problém s místem na disku
**Problém:** "No space left on device" při načítání velkých modelů
**Řešení:** Vytvoření Disk Manager Library pro automatickou správu úložiště

---

## 📈 Výsledky

### Vytvořený dataset:
- **Celkový počet párů:** 3,000 QA párů v `data/all.jsonl`
- **Formát:** JSONL s konverzačními páry (system, user, assistant)
- **Struktura:** Každá odpověď končí "Andrej Babiš" jako podpis

### Charakteristika obsahu:
- **Styl:** Mluvená čeština s "babíšovštinou"
- **Témata:** Politika, ekonomika, rodina, podnikání
- **Tón:** Satirický, emotivní, sebestředný
- **Jazykové prvky:** Slovensko-české odchylky, záměrné chyby

### Vytvořené nástroje:
- **Disk Manager Library** - kompletní knihovna pro správu úložiště
- **Fine-tuning skript** - automatizovaný proces s retry logikou
- **Testy** - pokrytí funkcionality knihovny

---

## 📝 Závěr

Projekt úspěšně demonstroval kompletní workflow pro vytvoření fine-tuning datasetu pomocí LLM. Klíčovým poznatkem bylo, že složité zadání je nutné rozdělit na menší, detailní kroky. Vytvořený dataset obsahuje 3,000 konverzačních párů připravených pro fine-tuning, i když byl následně zablokován OpenAI moderací kvůli satirickému obsahu.

**Silné stránky:** Kompletní implementace, kvalitní dataset, dobře strukturovaný kód, vlastní knihovna pro správu úložiště
**Oblasti pro zlepšení:** Řešení moderace obsahu, alternativní přístupy k fine-tuningu
