import subprocess
import sys

def install_packages():
    """Installs required packages using pip."""
    packages = [
        "torch",
        "git+https://github.com/huggingface/transformers",
        "bitsandbytes",
        "accelerate",
        "peft",
        "datasets",
        "evaluate",
        "trl",
        "matplotlib",
        "tensorboard",
        "sentencepiece"
    ]
    for package in packages:
        try:
            # The -U flag ensures the package is upgraded to the latest version, or the specified version.
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")

# Install dependencies before any other imports
install_packages()

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# Načtení modelu s kvantizací (optimalizováno pro Colab GPU)
quantization_config = None
if getattr(config, 'use_4bit', False):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
elif getattr(config, 'use_8bit', False):
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

model_kwargs = {
    "torch_dtype": torch.float16 if getattr(config, 'fp16', True) else torch.float32,
    "device_map": "auto" if torch.cuda.is_available() else None
}

if quantization_config:
    model_kwargs["quantization_config"] = quantization_config

model = AutoModelForCausalLM.from_pretrained(
    config.base_model,
    **model_kwargs
)

# Příprava modelu pro kvantizované trénování
if getattr(config, 'use_4bit', False) or getattr(config, 'use_8bit', False):
    model = prepare_model_for_kbit_training(model)

print(f"Model načten: {config.base_model}")
print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")