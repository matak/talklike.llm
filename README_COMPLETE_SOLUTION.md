# 🎯 Kompletní řešení Fine-tuning projektu

## 📋 Přehled

Tento projekt implementuje **kompletní řešení fine-tuningu jazykového modelu** rozdělené na **3 hlavní části** podle zadání úkolu:

1. **📊 Příprava datasetu** - Vytvoření trénovacích dat
2. **🏋️ Fine-tuning modelu** - Doladění jazykového modelu  
3. **📈 Benchmarking** - Srovnání před a po fine-tuningu

---

## 🚀 Rychlý start

### Spuštění celého workflow:

```bash
# Kompletní pipeline (doporučeno)
python run_complete_workflow.py --complete

# Nebo interaktivně
python run_complete_workflow.py
```

### Spuštění jednotlivých částí:

```bash
# 1. Příprava datasetu
./scripts/prepare_dataset.sh

# 2. Fine-tuning modelu
./scripts/run_finetune.sh

# 3. Benchmarking
./scripts/run_benchmarking.sh
```

---

## 📊 ČÁST 1: Příprava datasetu

### 🎯 Cíl
Vytvořit kvalitní dataset pro fine-tuning modelu ve stylu Andreje Babiše.

### 📁 Struktura
```
dataset_preparation/
├── templates/              # Šablony s placeholdery
├── generators/             # Generátory dat
├── moderation/             # Moderace obsahu
├── data/                   # Mezivýstupy
└── run_dataset_preparation.py  # Hlavní skript
```

### 🔧 Implementace

#### Krok 1: Vytvoření šablon
- 400 šablon s placeholdery pro různé kategorie
- Témata: inflace, válka, důchody, klima, očkování, daně
- Nepřátelé: Brusel, Piráti, Fiala, novináři, opozice

#### Krok 2: Generování odpovědí
- Nahrazení placeholderů konkrétními hodnotami
- 15% pravděpodobnost jazykové chyby
- Mix 5 stylů (emocionální, odmítavý, domýšlivý, chaotický, ironický)

#### Krok 3: Vytvoření dialogů
- Generování novinářských otázek k odpovědím
- Formát: JSONL s páry otázka-odpověď

#### Krok 4: Moderace obsahu
- Kontrola pomocí OpenAI Moderation API
- Filtrace nevhodného obsahu

#### Krok 5: Finální dataset
- Strukturovaný formát pro fine-tuning
- 3,000 QA párů v `data/all.jsonl`

### 📊 Výstup
- **Dataset:** `data/all.jsonl` (3,000 QA párů)
- **Formát:** JSONL s konverzačními páry (system, user, assistant)
- **Struktura:** Každá odpověď končí "Andrej Babiš" jako podpis

---

## 🏋️ ČÁST 2: Fine-tuning modelu

### 🎯 Cíl
Fine-tune jazykový model na připraveném datasetu pomocí LoRA.

### 📁 Struktura
```
fine_tuning/
├── models/                 # Fine-tuning skripty
├── configs/                # Konfigurace
├── utils/                  # Pomocné funkce
├── scripts/                # Bash skripty
└── run_finetune.py         # Hlavní skript
```

### 🔧 Implementace

#### Krok 1: Příprava prostředí
- Instalace závislostí
- Nastavení tokenů (HF_TOKEN, WANDB_API_KEY)
- Kontrola GPU

#### Krok 2: Načtení modelu
- Base model: Meta-Llama-3-8B-Instruct
- Tokenizer s podporou chat formátu
- Automatická detekce device (GPU/CPU)

#### Krok 3: Konfigurace LoRA
- Rank (r): 16
- Alpha: 32
- Dropout: 0.1
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

#### Krok 4: Příprava dat
- Tokenizace datasetu
- Rozdělení na train/validation (90/10)
- Data collator pro language modeling

#### Krok 5: Trénování
- Epochs: 3
- Batch size: 2
- Learning rate: 2e-4
- Gradient accumulation: 4
- FP16 pro GPU

#### Krok 6: Uložení modelu
- Lokální uložení
- Nahrání na Hugging Face Hub (volitelné)

### 🎯 Podporované modely
- **Llama 3 8B** (hlavní)
- **Mistral 7B** (alternativní)
- **LoRA** (Low-Rank Adaptation)

### 📊 Výstup
- **Fine-tuned model** v `./babis-llama-finetuned/final`
- **LoRA adaptéry** pro efektivní uložení
- **Training metrics** a logy

---

## 📈 ČÁST 3: Benchmarking

### 🎯 Cíl
Srovnat model před a po fine-tuningu pomocí benchmarkingu.

### 📁 Struktura
```
benchmarking/
├── evaluation/             # Evaluace modelů
├── tests/                  # Testovací data
├── reports/                # Reporty
├── utils/                  # Pomocné funkce
└── run_benchmarking.py     # Hlavní skript
```

### 🔧 Implementace

#### Krok 1: Načtení modelů
- Base model (před fine-tuningem)
- Fine-tuned model (po fine-tuningem)

#### Krok 2: Kvantitativní metriky
- **Perplexity** - měření jazykového modelu
- **BLEU score** - kvalita překladu/generování
- **ROUGE score** - překryv s referenčními texty
- **Průměrná délka odpovědí**

#### Krok 3: Kvalitativní evaluace
- **Stylová podobnost** - napodobení "babíšovštiny"
- **Tematická relevance** - odpovídání na otázky
- **Jazykové chyby** - záměrné vs. skutečné chyby
- **Emotivnost** - charakteristické výrazy

#### Krok 4: Výkonnostní metriky
- **Generační rychlost** - tokens za sekundu
- **Paměťová náročnost** - VRAM využití
- **Latence** - doba odezvy

#### Krok 5: Generování reportu
- HTML report s vizualizacemi
- JSON data pro další analýzu
- Grafy a tabulky

### 📊 Metriky

#### Kvantitativní:
- Perplexity (nižší = lepší)
- BLEU score (vyšší = lepší)
- ROUGE score (vyšší = lepší)

#### Kvalitativní:
- Stylová podobnost (0-1)
- Tematická relevance (0-1)
- Emotivnost (0-1)

#### Výkonnostní:
- Generační rychlost (resp/s)
- Paměťová náročnost (GB)
- Latence (sekundy)

### 📊 Výstup
- **HTML report** s kompletní analýzou
- **JSON data** pro další zpracování
- **Grafy** a vizualizace
- **Testovací odpovědi** pro porovnání

---

## 📋 Kompletní workflow

### 1️⃣ Příprava datasetu
```bash
cd dataset_preparation
python run_dataset_preparation.py --complete
```

### 2️⃣ Fine-tuning modelu
```bash
cd fine_tuning
python run_finetune.py \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --data_path "data/all.jsonl" \
    --output_dir "./babis-llama-finetuned" \
    --epochs 3 \
    --batch_size 2
```

### 3️⃣ Benchmarking
```bash
cd benchmarking
python run_benchmarking.py \
    --base_model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --finetuned_model_path "./babis-llama-finetuned/final"
```

---

## 🛠️ Instalace a nastavení

### Požadavky
```bash
# Python 3.8+
python --version

# GPU (doporučeno)
nvidia-smi

# Závislosti
pip install torch transformers peft datasets wandb evaluate nltk rouge-score matplotlib seaborn pandas
```

### Nastavení tokenů
```bash
# Hugging Face token
export HF_TOKEN="your_token_here"

# Weights & Biases token (volitelné)
export WANDB_API_KEY="your_wandb_token_here"
```

### Kontrola předpokladů
```bash
python run_complete_workflow.py --check-only
```

---

## 📊 Očekávané výsledky

### Dataset:
- ✅ 3,000 QA párů
- ✅ Strukturovaný formát
- ✅ Moderovaný obsah

### Fine-tuned model:
- ✅ Styl Andreje Babiše
- ✅ Charakteristické fráze
- ✅ Emotivní odpovědi

### Benchmarking:
- ✅ Kvantitativní srovnání
- ✅ Kvalitativní evaluace
- ✅ Výkonnostní metriky

---

## 🎯 Splnění zadání

✅ **Metoda fine-tuningu** - Hugging Face + PEFT (LoRA)  
✅ **Srovnání před/po** - Kompletní benchmarking  
✅ **Bodové ohodnocení** - Kvantitativní metriky  
✅ **Forma odevzdání** - Report s tabulkami a screenshoty  

---

## 📁 Výstupní soubory

Po spuštění kompletního workflow budete mít:

```
project/
├── data/
│   └── all.jsonl                    # Finální dataset
├── babis-llama-finetuned/
│   └── final/                       # Fine-tuned model
├── benchmarking/
│   ├── reports/
│   │   ├── benchmark_report.html    # HTML report
│   │   ├── benchmark_report.json    # JSON data
│   │   └── *.png                    # Grafy
│   └── tests/
│       └── model_responses.json     # Testovací odpovědi
├── workflow_completion_report.json  # Shrnutí workflow
└── *.log                           # Logy
```

---

## 🚀 Další kroky

1. **Spusťte kompletní workflow:**
   ```bash
   python run_complete_workflow.py --complete
   ```

2. **Prohlédněte si výsledky:**
   - HTML report: `benchmarking/reports/benchmark_report.html`
   - JSON data: `benchmarking/reports/benchmark_report.json`
   - Testovací odpovědi: `benchmarking/tests/model_responses.json`

3. **Odevzdejte výsledky:**
   - Screenshoty z HTML reportu
   - Tabulky s metrikami
   - Porovnání odpovědí před/po fine-tuningu

---

## 🔧 Troubleshooting

### Problémy s GPU
```bash
# Kontrola GPU
nvidia-smi

# Snížení batch size
python fine_tuning/run_finetune.py --batch_size 1
```

### Problémy s pamětí
```bash
# Použití 8-bit kvantizace
python fine_tuning/run_finetune.py --load_in_8bit

# Nebo 4-bit kvantizace
python fine_tuning/run_finetune.py --load_in_4bit
```

### Problémy s datasetem
```bash
# Kontrola datasetu
python dataset_preparation/run_dataset_preparation.py --step 1
```

---

## 📝 Poznámky

- **Časová náročnost:** Fine-tuning může trvat několik hodin
- **Hardwarové požadavky:** Doporučeno GPU s minimálně 16GB VRAM
- **Tokeny:** Pro nahrání na HF Hub je potřeba HF_TOKEN
- **Monitoring:** W&B logging pro sledování průběhu trénování

---

## 🎉 Hotovo!

Vše je připraveno k použití! Spusťte kompletní workflow a získejte výsledky pro odevzdání úkolu.

```bash
python run_complete_workflow.py --complete
``` 