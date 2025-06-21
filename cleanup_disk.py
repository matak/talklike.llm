#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skript pro vyčištění disku - TalkLike.LLM
Používá disk_manager knihovnu pro profesionální správu diskového prostoru
"""

import sys
from pathlib import Path

# Přidání cesty k lib adresáři
sys.path.append(str(Path(__file__).parent / "lib"))

def main():
    """Hlavní funkce pro vyčištění disku"""
    print("🧹 VYČIŠTĚNÍ DISKU - TalkLike.LLM")
    print("=" * 50)
    
    try:
        from disk_manager import DiskManager, quick_cleanup, check_and_cleanup
        
        # Inicializace disk manageru
        dm = DiskManager()
        
        # Zobrazení aktuálního stavu
        print("📊 Aktuální stav disku:")
        dm.print_storage_summary()
        
        # Kontrola a vyčištění
        print("\n🔍 Kontrola a vyčištění...")
        if check_and_cleanup(threshold=80):
            print("✅ Disk je v pořádku!")
        else:
            print("⚠️  Disk je stále téměř plný")
            
            # Agresivní vyčištění
            print("\n🧹 Agresivní vyčištění...")
            cleaned = dm.aggressive_cleanup()
            
            if cleaned:
                print("✅ Agresivní vyčištění dokončeno")
                print(f"   Vyčištěno: {len(cleaned)} adresářů")
            else:
                print("❌ Agresivní vyčištění selhalo")
        
        # Nastavení network storage
        print("\n🔧 Nastavuji network storage...")
        if dm.setup_network_storage("/workspace"):
            print("✅ Network storage nastaven")
        else:
            print("❌ Network storage se nepodařilo nastavit")
        
        # Finální shrnutí
        print("\n📊 Finální stav:")
        dm.print_storage_summary()
        
    except ImportError:
        print("❌ Disk manager knihovna není dostupná!")
        print("💡 Zkontrolujte, že je knihovna nainstalována:")
        print("   cd lib/disk_manager && pip install -e .")
        return False
        
    except Exception as e:
        print(f"❌ Neočekávaná chyba: {e}")
        return False
    
    print("\n🎉 Vyčištění disku dokončeno!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 