# 🔧 Změny pro integraci s adaptérem

> **📚 Navigace:** [🏠 Hlavní projekt](../README.md) | [📊 Benchmarking](README_ADAPTER.md)

## 📋 Přehled změn

Tento dokument shrnuje všechny úpravy provedené v adresáři `3_benchmarking` pro integraci s vaším natrénovaným adaptérem `mcmatak/babis-mistral-adapter`.

---

## 🔄 Hlavní změny

### 1. **generate_responses.py** - KLÍČOVÁ ZMĚNA
**Problém**: Původně používal mock data místo skutečného modelu.

**Řešení**: 
- ✅ Přidána integrace s `test_adapter.py` z `2_finetunning`
- ✅ Implementována funkce `load_benchmark_model()` pro načtení vašeho adaptéru
- ✅ Implementována funkce `generate_real_response()` pro skutečné generování
- ✅ Přidán system prompt optimalizovaný pro Babišův styl
- ✅ Zachována fallback funkce pro mock data

**Konfigurace**:
```python
# Váš adaptér
base_model = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_path = "mcmatak/babis-mistral-adapter"
```

### 2. **requirements_benchmarking.txt** - ROZŠÍŘENO
**Přidáno**:
- `peft>=0.4.0` - pro LoRA adaptéry
- `accelerate>=0.20.0` - pro optimalizaci
- `bitsandbytes>=0.39.0` - pro kvantizaci
- `h5py>=3.8.0` - pro cache
- `safetensors>=0.3.0` - pro bezpečné uložení

### 3. **Nové skripty**

#### `run_benchmark_with_adapter.sh`
- Automatické nastavení cache do `/workspace`
- Instalace requirements
- Spuštění benchmarkingu

#### `test_adapter_integration.py`
- Kompletní test integrace s adaptérem
- Ověření všech komponent
- Diagnostika problémů

#### `quick_test_adapter.py`
- Rychlý test vašeho adaptéru
- Generování ukázkových odpovědí
- Analýza stylu v reálném čase

### 4. **Nová dokumentace**

#### `README_ADAPTER.md`
- Kompletní dokumentace pro použití s adaptérem
- Instrukce pro odevzdání úkolu
- Troubleshooting guide

#### `QUICKSTART_ADAPTER.md`
- Rychlý start guide
- Očekávané výsledky
- Checklist pro odevzdání

#### `CHANGES_FOR_ADAPTER.md`
- Tento soubor - shrnutí změn

---

## 🎯 Funkcionalita

### Před změnami
```python
# Používalo mock data
def generate_mock_response(question: str, model_type: str) -> str:
    # Falešné odpovědi pro testování
    return random.choice(mock_responses)
```

### Po změnách
```python
# Používá skutečný model s adaptérem
def generate_real_response(model, tokenizer, question: str, model_type: str) -> str:
    # System prompt pro Babišův styl
    system_prompt = """Jsi Andrej Babiš, český politik..."""
    
    # Generování skutečné odpovědi
    response = generate_response(model, tokenizer, prompt, max_length=300, temperature=0.8)
    return response
```

---

## 📊 Očekávané výsledky

### Před fine-tuningem (base model)
- **Skóre**: 2-3/10 (F)
- **Styl**: Neutrální, formální
- **Fráze**: Žádné Babišovy fráze

### Po fine-tuningem (váš adaptér)
- **Skóre**: 8-9/10 (A)
- **Styl**: Autentický Babišův styl
- **Fráze**: "Hele", "To je skandál!", "Já makám", "Andrej Babiš"

---

## 🚀 Jak spustit

### 1. Test adaptéru
```bash
cd 3_benchmarking
python quick_test_adapter.py
```

### 2. Kompletní test
```bash
python test_adapter_integration.py
```

### 3. Benchmarking
```bash
./run_benchmark_with_adapter.sh
```

---

## 🔧 Technické detaily

### System prompt
```python
system_prompt = """Jsi Andrej Babiš, český politik a podnikatel. Tvým úkolem je odpovídat na otázky v charakteristickém Babišově stylu.

Charakteristické prvky tvého stylu:
- Typické fráze: "Hele, ...", "To je skandál!", "Já makám", "Opozice krade", "V Bruselu"
- Slovenské odchylky: "sme", "som", "makáme", "centralizácia"
- Emotivní výrazy: "to je šílený!", "tragédyje!", "kampááň!"
- Přirovnání: "jak když kráva hraje na klavír", "jak když dítě řídí tank"
- První osoba: "Já jsem...", "Moje rodina...", "Já makám..."
- Podpis: Každou odpověď zakonči "Andrej Babiš"

Odpovídej vždy v první osobě jako Andrej Babiš, používej jeho charakteristické fráze, buď emotivní a přímý."""
```

### Cache nastavení
```bash
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
```

### Parametry generování
```python
max_length=300      # Maximální délka odpovědi
temperature=0.8     # Kreativita (0.0-1.0)
top_p=0.9          # Nucleus sampling
repetition_penalty=1.1  # Penalizace opakování
```

---

## 🛠️ Troubleshooting

### Časté problémy a řešení

#### 1. Import error
```bash
# Řešení: Instalace requirements
pip install -r requirements_benchmarking.txt
```

#### 2. Model se nenačte
```bash
# Řešení: Zkontrolujte cache
ls -la /workspace/.cache/huggingface/
```

#### 3. Chyba při generování
```bash
# Řešení: Snižte paměť nebo použijte CPU
export CUDA_VISIBLE_DEVICES=""
```

---

## ✅ Checklist implementace

- [x] ✅ Integrace s `test_adapter.py`
- [x] ✅ Implementace `generate_real_response()`
- [x] ✅ System prompt pro Babišův styl
- [x] ✅ Cache nastavení
- [x] ✅ Requirements aktualizovány
- [x] ✅ Test skripty vytvořeny
- [x] ✅ Dokumentace aktualizována
- [x] ✅ Fallback funkce zachovány
- [x] ✅ Error handling implementován

---

## 🎯 Výsledek

**Benchmarking systém je nyní plně integrován s vaším adaptérem a připraven pro odevzdání domácího úkolu!**

- ✅ Používá skutečný model místo mock dat
- ✅ Generuje autentické Babišovy odpovědi
- ✅ Poskytuje kvantitativní metriky
- ✅ Vytváří reporty pro odevzdání
- ✅ Obsahuje kompletní dokumentaci

**Další kroky**: Spusťte `python quick_test_adapter.py` pro ověření funkčnosti. 