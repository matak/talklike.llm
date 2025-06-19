"""
Pomocné funkce pro fine-tuning
"""
import os
import torch
from google.colab import drive

def setup_environment():
    """Nastaví prostředí pro Google Colab"""
    # Připojení Google Drive
    drive.mount('/content/drive')
    
    # Vytvoření adresářů
    os.makedirs('/content/babis_finetune', exist_ok=True)
    os.makedirs('/content/drive/MyDrive/babis_finetune', exist_ok=True)
    
    print("Google Drive připojen a adresáře vytvořeny!")

def check_gpu():
    """Zkontroluje dostupnost GPU"""
    print(f"PyTorch verze: {torch.__version__}")
    print(f"CUDA dostupné: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU paměť: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  GPU není dostupné - trénování bude pomalé!")

def install_requirements():
    """Nainstaluje potřebné knihovny"""
    print("📦 Instaluji potřebné knihovny...")
    
    # Seznam knihoven
    packages = [
        "transformers",
        "datasets", 
        "peft",
        "accelerate",
        "bitsandbytes",
        "wandb",
        "tiktoken",
        "huggingface_hub",
        "gradio",
        "streamlit"
    ]
    
    for package in packages:
        print(f"Instaluji {package}...")
        os.system(f"pip install -q {package}")
    
    print("✅ Všechny knihovny nainstalovány!")

def save_model(trainer, config):
    """Uloží natrénovaný model"""
    print("💾 Ukládám model...")
    
    # Uložení modelu
    trainer.save_model()
    
    # Uložení tokenizeru
    trainer.tokenizer.save_pretrained(config.output_dir)
    
    print(f"✅ Model uložen do: {config.output_dir}")

def generate_sample(model, tokenizer, prompt="Uživatel: Jaký je váš názor na inflaci?\nAndrej Babiš:", max_length=100):
    """Vygeneruje ukázkovou odpověď"""
    print(f"🤖 Generuji odpověď na: {prompt}")
    
    # Tokenizace promptu
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generování
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Dekódování
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Odpověď: {generated_text}")
    return generated_text 