# Nastavení Hugging Face tokenu
from getpass import getpass

print("Pro push na Hugging Face Hub potřebujete token.")
print("Získejte ho na: https://huggingface.co/settings/tokens")
print("Pokud nechcete pushovat, stiskněte Enter.")

hf_token = getpass("HF Token (volitelné): ")

if hf_token.strip():
    login(hf_token)
    print("✅ Přihlášeno k Hugging Face Hub")
    
    # Push na Hub
    print("🚀 Pushuji model na Hub...")
    trainer.push_to_hub(f"babis-{config.model_name}")
    tokenizer.push_to_hub(f"babis-{config.model_name}")
    
    print(f"✅ Model pushnut na Hub: babis-{config.model_name}")
    print(f"URL: https://huggingface.co/babis-{config.model_name}")
else:
    print("Model nebyl pushnut na Hub.") 