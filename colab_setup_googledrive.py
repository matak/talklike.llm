#!/usr/bin/env python3

# Připojení Google Drive
drive.mount('/content/drive')

# Vytvoření adresářů
!mkdir -p /content/babis_finetune
!mkdir -p /content/drive/MyDrive/babis_finetune

print("Google Drive připojen a adresáře vytvořeny!") 