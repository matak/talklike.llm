#!/usr/bin/env python3
"""
Script pro sloučení všech QA souborů z data/final do data/all.jsonl.
Vytváří strukturovaný dataset pro fine-tuning jazykového modelu.
"""

import os
import json
import glob
from typing import List, Dict, Any

def load_system_prompt() -> Dict[str, str]:
    """Načte systémový prompt pro fine-tuning."""
    system_prompt = {
        "role": "system",
        "content": "Jsi Andrej Babiš, český politik a podnikatel. Mluvíš jako on - používáš jeho charakteristické fráze, styl komunikace a názory. Vždy odpovídáš v první osobě jako Andrej Babiš. Používáš jeho typické výrazy jako 'Hele', 'To je skandál!', 'Já makám', 'Opozice krade', 'V Bruselu', 'Inflace je jak když kráva hraje na klavír' a podobné. Tvůj styl je přímý, někdy konfrontační, ale vždy se snažíš obhajovat své názory a práci. Mluvíš o své rodině, podnikání, politice a ekonomice způsobem, jakým to dělá skutečný Andrej Babiš."
    }
    return system_prompt

def merge_final_files():
    """Sloučí všechny soubory z adresáře data/final do data/all.jsonl"""
    input_dir = "data/final"
    output_file = "data/all.jsonl"
    
    # Získání všech QA souborů z final adresáře
    qa_files = glob.glob(os.path.join(input_dir, "*_babis_output_qa.jsonl"))
    qa_files = sorted(qa_files)  # Seřazení pro konzistentní pořadí
    
    if not qa_files:
        print("Nebyly nalezeny žádné QA soubory v adresáři data/final!")
        return
        
    print(f"Nalezeno {len(qa_files)} QA souborů")
    
    # Vytvoření výstupního adresáře
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Načtení systémového promptu
    system_message = load_system_prompt()
    print("Načten systémový prompt")
    
    # Sbírání všech zpráv, začínaje systémovou zprávou
    all_messages = [system_message]
    total_qa_pairs = 0
    
    for qa_file in qa_files:
        print(f"Zpracovávám {os.path.basename(qa_file)}")
        try:
            with open(qa_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            # Parsování QA páru
                            qa_pair = json.loads(line)
                            
                            # Kontrola struktury
                            if "question" not in qa_pair or "answer" not in qa_pair:
                                print(f"Varování: Neplatná struktura v souboru {qa_file}, řádek {line_num}")
                                continue
                            
                            # Přidání uživatelské zprávy
                            all_messages.append({
                                "role": "user",
                                "content": qa_pair["question"]
                            })
                            
                            # Přidání asistentovy zprávy
                            all_messages.append({
                                "role": "assistant", 
                                "content": qa_pair["answer"]
                            })
                            
                            total_qa_pairs += 1
                            
                        except json.JSONDecodeError as e:
                            print(f"Chyba JSON v souboru {qa_file}, řádek {line_num}: {e}")
                            continue
                            
        except Exception as e:
            print(f"Chyba při zpracování {qa_file}: {e}")
            continue
    
    # Zápis všech zpráv jako jeden JSON objekt
    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump({"messages": all_messages}, out_f, ensure_ascii=False, indent=4)
    
    print(f"\n=== Sloučení dokončeno ===")
    print(f"Vytvořen soubor: {output_file}")
    print(f"Celkem zpráv: {len(all_messages)} (včetně systémové zprávy)")
    print(f"Celkem QA párů: {total_qa_pairs}")
    print(f"Zpracované soubory:")
    for qa_file in qa_files:
        print(f"  - {os.path.basename(qa_file)}")

def validate_dataset(file_path: str) -> Dict[str, Any]:
    """Validuje vytvořený dataset."""
    print(f"\n=== Validace datasetu ===")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        messages = data.get("messages", [])
        
        if not messages:
            return {"valid": False, "error": "Dataset neobsahuje žádné zprávy"}
        
        # Kontrola první zprávy (systémová)
        if messages[0]["role"] != "system":
            return {"valid": False, "error": "První zpráva není systémová"}
        
        # Kontrola struktury konverzací
        conversation_count = 0
        total_user_messages = 0
        total_assistant_messages = 0
        
        for i in range(1, len(messages), 2):
            if i + 1 < len(messages):
                user_msg = messages[i]
                assistant_msg = messages[i + 1]
                
                if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
                    conversation_count += 1
                    total_user_messages += 1
                    total_assistant_messages += 1
                else:
                    print(f"Varování: Neplatná struktura konverzace na pozici {i}")
        
        # Kontrola poslední zprávy
        if messages[-1]["role"] != "assistant":
            return {"valid": False, "error": "Dataset nekončí asistentovou zprávou"}
        
        validation_result = {
            "valid": True,
            "total_messages": len(messages),
            "conversation_count": conversation_count,
            "user_messages": total_user_messages,
            "assistant_messages": total_assistant_messages,
            "system_messages": 1
        }
        
        print(f"Validace úspěšná:")
        print(f"  - Celkem zpráv: {validation_result['total_messages']}")
        print(f"  - Konverzací: {validation_result['conversation_count']}")
        print(f"  - Uživatelských zpráv: {validation_result['user_messages']}")
        print(f"  - Asistentových zpráv: {validation_result['assistant_messages']}")
        print(f"  - Systémových zpráv: {validation_result['system_messages']}")
        
        return validation_result
        
    except Exception as e:
        return {"valid": False, "error": f"Chyba při validaci: {str(e)}"}

def main():
    """Hlavní funkce pro sloučení datasetu."""
    print("=== Sloučení QA datasetu ===")
    
    # Sloučení souborů
    merge_final_files()
    
    # Validace výsledného datasetu
    output_file = "data/all.jsonl"
    if os.path.exists(output_file):
        validation_result = validate_dataset(output_file)
        if not validation_result["valid"]:
            print(f"CHYBA: Dataset není validní: {validation_result['error']}")
        else:
            print(f"\n✅ Dataset úspěšně vytvořen a validován!")
    else:
        print(f"CHYBA: Výstupní soubor {output_file} nebyl vytvořen")

if __name__ == "__main__":
    main() 