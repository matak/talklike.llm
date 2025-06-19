# 🤖 Google Colab Fine-tuning pro Babišův styl

Tento projekt implementuje kompletní pipeline pro fine-tuning jazykového modelu na napodobení charakteristického stylu komunikace Andreje Babiše pomocí **Google Colab** - zdarma a bez nutnosti vlastního hardware.

## 🎯 Proč Google Colab?

| Výhoda | Popis |
|--------|-------|
| 🆓 **Zdarma** | GPU Tesla T4/P100 bez poplatků |
| 🚀 **Rychlé** | Žádná instalace, vše v prohlížeči |
| 💾 **Automatické ukládání** | Model se ukládá na Google Drive |
| 🌐 **Sdílení** | Snadné sdílení přes Hugging Face Hub |
| 📊 **Monitoring** | Integrace s Weights & Biases |

## 🚀 Rychlý start

### 1. Otevřete Google Colab

```bash
# Klikněte na tento odkaz:
https://colab.research.google.com/github/your-username/talklike.llm/blob/main/colab_finetune.ipynb
```

### 2. Spusťte notebook

1. **Otevřete** `colab_finetune.ipynb` v Google Colab
2. **Změňte runtime** na GPU: `Runtime` → `Change runtime type` → `GPU`
3. **Spusťte všechny buňky** postupně od začátku

### 3. Výsledky

Po dokončení budete mít:
- ✅ Fine-tuned model uložený na Google Drive
- ✅ Model pushnutý na Hugging Face Hub (volitelné)
- ✅ Gradio interface pro testování
- ✅ Kompletní metriky a logy

## 📋 Co notebook dělá

### 1. **Setup a instalace**
```python
# Automatická instalace knihoven
!pip install transformers datasets peft accelerate bitsandbytes wandb
```

### 2. **Konfigurace pro Colab**
```python
@dataclass
class ColabConfig:
    base_model: str = "microsoft/DialoGPT-medium"  # Optimalizováno pro Colab
    learning_rate: float = 2e-4
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2  # Pro T4 GPU
    use_4bit: bool = True  # Kvantizace pro úsporu paměti
```

### 3. **Vytvoření datasetu**
- Ukázkový dataset s Babišovými výroky
- Automatické rozdělení na train/validation
- Tokenizace pro fine-tuning

### 4. **Fine-tuning s LoRA**
```python
lora_config = LoraConfig(
    r=8,                    # Rank (menší pro Colab)
    lora_alpha=16,          # Scaling faktor
    target_modules=["c_attn", "c_proj", "wte", "wpe"],
    task_type=TaskType.CAUSAL_LM
)
```

### 5. **Trénování a evaluace**
- Automatické sledování metrik
- Early stopping
- Uložení nejlepšího modelu

### 6. **Testování a deployment**
- Generování odpovědí
- Gradio interface
- Push na Hugging Face Hub

## 🔧 Konfigurace

### Základní parametry (optimalizované pro Colab):

```python
config = ColabConfig(
    base_model="microsoft/DialoGPT-medium",  # Malý model pro rychlé trénování
    num_train_epochs=2,                      # Krátké trénování
    per_device_train_batch_size=2,           # Pro T4 GPU
    gradient_accumulation_steps=8,           # Efektivní batch size = 16
    use_4bit=True,                           # Kvantizace pro úsporu paměti
    lora_r=8,                                # Menší LoRA pro rychlost
    max_seq_length=512                       # Kratší sekvence
)
```

### Pro větší modely:

```python
# Pro Llama-2 (vyžaduje více paměti)
config.base_model = "meta-llama/Llama-2-7b-chat-hf"
config.per_device_train_batch_size = 1
config.gradient_accumulation_steps = 16
config.max_seq_length = 1024
```

## 📊 Dataset

### Ukázkový dataset obsahuje:

```python
conversations = [
    {
        "prompt": "Uživatel: Jaký je váš názor na inflaci?\nAndrej Babiš: ",
        "completion": "Hele, inflace je jak když kráva hraje na klavír. Já makám, ale Brusel to sabotuje. Andrej Babiš"
    },
    {
        "prompt": "Uživatel: Co si myslíte o opozici?\nAndrej Babiš: ",
        "completion": "Opozice jen krade a sabotuje. Já s rodinou makáme, ale oni jen řeční. To je skandál! Andrej Babiš"
    }
    # ... další příklady
]
```

### Vlastní dataset:

Můžete nahradit ukázkový dataset vlastními daty:

```python
# Načtení vlastního JSONL souboru
from datasets import load_dataset

dataset = load_dataset('json', data_files='your_data.jsonl')
```

## 🎯 Fine-tuning s LoRA

### Výhody LoRA pro Colab:

- ✅ **Rychlejší trénování** - méně parametrů
- ✅ **Menší paměťové nároky** - vhodné pro T4 GPU
- ✅ **Zachování původních schopností** - pouze adaptace
- ✅ **Snadné sdílení** - pouze LoRA adaptace

### LoRA konfigurace:

```python
lora_config = LoraConfig(
    r=8,                    # Rank matice (menší = rychlejší)
    lora_alpha=16,          # Scaling faktor
    target_modules=["c_attn", "c_proj"],  # Které vrstvy adaptovat
    lora_dropout=0.1,       # Dropout pro regularizaci
    bias="none",            # Nepřidávat bias
    task_type=TaskType.CAUSAL_LM
)
```

## 📈 Monitoring a metriky

### Weights & Biases integrace:

```python
wandb.init(
    project="babis-finetune-colab",
    name=config.model_name,
    config=vars(config)
)
```

### Sledované metriky:

- **Training loss** - ztráta během trénování
- **Evaluation loss** - ztráta na validačních datech
- **Training time** - celkový čas trénování
- **GPU utilization** - využití GPU

## 🧪 Testování modelu

### Funkce pro generování:

```python
def generate_babis_response(prompt, max_length=100, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Testovací prompty:

```python
test_prompts = [
    "Uživatel: Jaký je váš názor na inflaci?\nAndrej Babiš: ",
    "Uživatel: Co si myslíte o Bruselu?\nAndrej Babiš: ",
    "Uživatel: Jak hodnotíte opozici?\nAndrej Babiš: "
]
```

## 🌐 Gradio Interface

### Automatické vytvoření webového rozhraní:

```python
import gradio as gr

def babis_chat(message, history):
    prompt = f"Uživatel: {message}\nAndrej Babiš: "
    return generate_babis_response(prompt)

iface = gr.ChatInterface(
    fn=babis_chat,
    title="🤖 Babiš Chat Bot",
    examples=["Jaký je váš názor na inflaci?"]
)

iface.launch(share=True)
```

## 💾 Uložení a sdílení

### 1. Google Drive:

```python
# Automatické uložení na Drive
drive_path = f"/content/drive/MyDrive/babis_finetune/{config.model_name}"
!cp -r {config.output_dir} {drive_path}
```

### 2. Hugging Face Hub:

```python
# Push na Hub (vyžaduje token)
trainer.push_to_hub(f"babis-{config.model_name}")
tokenizer.push_to_hub(f"babis-{config.model_name}")
```

### 3. Stáhnutí modelu:

```bash
# Z Google Drive
# 1. Otevřete Google Drive
# 2. Přejděte do /MyDrive/babis_finetune/
# 3. Stáhněte složku s modelem

# Z Hugging Face Hub
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("your-username/babis-model")
model = AutoModelForCausalLM.from_pretrained("your-username/babis-model")
```

## 🛠️ Řešení problémů

### Časté chyby:

1. **Out of Memory**:
   ```python
   # Snižte batch size
   config.per_device_train_batch_size = 1
   config.gradient_accumulation_steps = 16
   ```

2. **Pomalé trénování**:
   ```python
   # Zkontrolujte GPU
   !nvidia-smi
   # Použijte menší model
   config.base_model = "microsoft/DialoGPT-small"
   ```

3. **Chyba při push na Hub**:
   ```python
   # Zkontrolujte token
   from huggingface_hub import login
   login("your-token")
   ```

### Optimalizace pro Colab:

```python
# Pro rychlejší trénování
config.use_4bit = True
config.fp16 = True
config.lora_r = 4
config.max_seq_length = 256

# Pro lepší kvalitu
config.num_train_epochs = 5
config.learning_rate = 1e-4
config.lora_r = 16
```

## 📊 Výsledky

### Typické výsledky na Colab T4:

```
🎉 FINE-TUNING DOKONČEN!
==================================================
Model: microsoft/DialoGPT-medium
Fine-tuned model: babis-dialogpt-colab
Training loss: 2.3456
Evaluation loss: 2.1234
Training time: 45.2s
LoRA parameters: r=8, alpha=16
Dataset size: 6 train, 2 validation

📁 Uložené soubory:
Colab: /content/babis_finetune
Google Drive: /content/drive/MyDrive/babis_finetune/babis-dialogpt-colab
Hugging Face Hub: babis-babis-dialogpt-colab
```

## 🔮 Rozšíření

### Možnosti vylepšení:

1. **Větší dataset**:
   ```python
   # Načtení většího datasetu
   dataset = load_dataset('json', data_files='large_babis_dataset.jsonl')
   ```

2. **Větší model**:
   ```python
   # Použití většího modelu
   config.base_model = "meta-llama/Llama-2-7b-chat-hf"
   ```

3. **QLoRA**:
   ```python
   # Kvantizované LoRA
   config.use_4bit = True
   config.use_qlora = True
   ```

4. **RLHF**:
   ```python
   # Reinforcement Learning from Human Feedback
   # Vyžaduje dodatečnou implementaci
   ```

## 📚 Reference

- [Google Colab](https://colab.research.google.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Gradio](https://gradio.app/)

## 🎯 Závěr

Google Colab poskytuje ideální prostředí pro fine-tuning jazykových modelů:

- ✅ **Zdarma** - žádné náklady na hardware
- ✅ **Rychlé** - GPU akcelerace
- ✅ **Jednoduché** - vše v prohlížeči
- ✅ **Sdílené** - snadné sdílení výsledků
- ✅ **Škálovatelné** - možnost upgrade na Pro

Tento přístup je ideální pro:
- 🎓 **Vzdělávání** - demonstrace fine-tuningu
- 🔬 **Výzkum** - rychlé experimenty
- 🚀 **Prototypování** - testování nápadů
- 📱 **Deployment** - příprava pro produkci

---

**Poznámka**: Tento projekt je vytvořen pro vzdělávací účely a demonstraci možností fine-tuningu. Satirický obsah je určen pouze pro výzkumné účely. 