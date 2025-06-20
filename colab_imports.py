#!/usr/bin/env python3

# Import knihoven
import torch
import json
import logging
from typing import Optional, Dict, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset, DatasetDict
import wandb
from huggingface_hub import HfApi, login
from google.colab import drive
from tqdm import tqdm

# Nastavení logování
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(f"PyTorch verze: {torch.__version__}")
print(f"CUDA dostupné: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU paměť: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB") 