#!/usr/bin/env python3
"""
Script pro sloučení všech QA souborů z data/final do data/all.jsonl.
Vytváří strukturovaný dataset pro fine-tuning jazykového modelu.
"""

import os
import json
import glob
from typing import List, Dict, Any

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
    
    # Sbírání všech konverzací
    all_conversations = []
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
                            
                            # Vytvoření konverzace bez system promptu
                            conversation = {
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": qa_pair["question"]
                                    },
                                    {
                                        "role": "assistant", 
                                        "content": qa_pair["answer"]
                                    }
                                ]
                            }
                            
                            all_conversations.append(conversation)
                            total_qa_pairs += 1
                            
                        except json.JSONDecodeError as e:
                            print(f"Chyba JSON v souboru {qa_file}, řádek {line_num}: {e}")
                            continue
                            
        except Exception as e:
            print(f"Chyba při zpracování {qa_file}: {e}")
            continue
    
    # Zápis všech konverzací do JSONL formátu
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for conversation in all_conversations:
            json.dump(conversation, out_f, ensure_ascii=False)
            out_f.write('\n')
    
    print(f"\n=== Sloučení dokončeno ===")
    print(f"Vytvořen soubor: {output_file}")
    print(f"Celkem konverzací: {len(all_conversations)}")
    print(f"Celkem QA párů: {total_qa_pairs}")
    print(f"Zpracované soubory:")
    for qa_file in qa_files:
        print(f"  - {os.path.basename(qa_file)}")

def validate_dataset(file_path: str) -> Dict[str, Any]:
    """Validuje vytvořený dataset."""
    print(f"\n=== Validace datasetu ===")
    
    try:
        conversations = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        conversation = json.loads(line)
                        conversations.append(conversation)
                    except json.JSONDecodeError as e:
                        print(f"Chyba JSON v řádku {line_num}: {e}")
                        continue
        
        if not conversations:
            return {"valid": False, "error": "Dataset neobsahuje žádné konverzace"}
        
        # Kontrola struktury konverzací
        conversation_count = 0
        total_user_messages = 0
        total_assistant_messages = 0
        
        for i, conversation in enumerate(conversations):
            messages = conversation.get("messages", [])
            
            if len(messages) != 2:
                print(f"Varování: Konverzace {i+1} nemá správný počet zpráv: {len(messages)}")
                continue
            
            user_msg = messages[0]
            assistant_msg = messages[1]
            
            if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
                conversation_count += 1
                total_user_messages += 1
                total_assistant_messages += 1
                
                # Kontrola, zda asistentova odpověď končí "Andrej Babiš"
                if not assistant_msg["content"].strip().endswith("Andrej Babiš"):
                    print(f"Varování: Odpověď {conversation_count} nekončí 'Andrej Babiš'")
            else:
                print(f"Varování: Neplatná struktura konverzace {i+1}")
        
        validation_result = {
            "valid": True,
            "total_conversations": len(conversations),
            "valid_conversations": conversation_count,
            "user_messages": total_user_messages,
            "assistant_messages": total_assistant_messages
        }
        
        print(f"Validace úspěšná:")
        print(f"  - Celkem konverzací: {validation_result['total_conversations']}")
        print(f"  - Validních konverzací: {validation_result['valid_conversations']}")
        print(f"  - Uživatelských zpráv: {validation_result['user_messages']}")
        print(f"  - Asistentových zpráv: {validation_result['assistant_messages']}")
        
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