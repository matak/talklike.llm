#!/usr/bin/env python3

# Instalace potřebných knihoven
!pip install -q transformers datasets peft accelerate bitsandbytes wandb tiktoken
!pip install -q huggingface_hub gradio streamlit

# Restart runtime po instalaci
import os
os.kill(os.getpid(), 9)