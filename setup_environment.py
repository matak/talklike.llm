#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centrální modul pro nastavení prostředí pro TalkLike.LLM
Importujte tento modul na začátek každého skriptu pro správné nastavení cache
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Nastaví prostředí pro TalkLike.LLM"""
    
    # Přidání cest pro import modulů
    current_dir = Path(__file__).parent
    sys.path.append(str(current_dir / "lib"))
    sys.path.append(str(current_dir / "2_finetunning"))
    sys.path.append(str(current_dir / "3_benchmarking"))
    
    # Import a použití disk_manager knihovny
    try:
        from disk_manager import DiskManager, check_and_cleanup
        
        # Inicializace disk manageru
        dm = DiskManager()
        
        # Kontrola a vyčištění disku pokud je potřeba
        if not dm.check_disk_space(threshold=90):
            print("🧹 Disk je téměř plný, čistím cache...")
            check_and_cleanup(threshold=90)
        
        # Nastavení network storage pro cache
        dm.setup_network_storage("/workspace")
        
        print("✅ Disk manager nastaven")
        
    except ImportError:
        print("⚠️  Disk manager knihovna není dostupná, používám základní nastavení")
        # Základní nastavení cache do /workspace
        os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
        os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface/transformers'
        os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'
        
        # Vytvoření cache adresářů
        os.makedirs('/workspace/.cache/huggingface/transformers', exist_ok=True)
        os.makedirs('/workspace/.cache/huggingface/datasets', exist_ok=True)
    
    # Výpis informací o nastavení
    print("📁 TalkLike.LLM prostředí nastaveno:")
    print(f"   HF_HOME: {os.environ.get('HF_HOME', 'N/A')}")
    print(f"   TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', 'N/A')}")
    print(f"   HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE', 'N/A')}")
    print(f"   Cache adresáře: /workspace/.cache/huggingface/")
    
    return True

# Automatické nastavení při importu
if __name__ != "__main__":
    setup_environment()

if __name__ == "__main__":
    # Test nastavení
    print("🧪 Test nastavení prostředí...")
    setup_environment()
    print("✅ Prostředí nastaveno úspěšně!") 