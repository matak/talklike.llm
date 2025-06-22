import json
import os
from datetime import datetime

def load_model_data(file_path, debugger=None):
    """Načte data z JSONL souboru nebo jednoho velkého JSON objektu"""
    conversations = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Debug: Uložení původního obsahu
    if debugger:
        debugger.save_step("01_original_content", {"content": content[:1000] + "..." if len(content) > 1000 else content}, 
                          "Původní obsah souboru")
    
    try:
        # Zkusíme parsovat jako jeden velký JSON objekt
        data = json.loads(content)
        
        if 'messages' in data:
            # Máme jeden velký objekt s messages - rozdělíme na konverzace
            messages = data['messages']
            print(f"📊 Načteno {len(messages)} zpráv v jednom objektu")
            
            # Debug: Uložení všech zpráv
            if debugger:
                debugger.save_step("02_all_messages", messages, f"Všech {len(messages)} zpráv z JSON objektu")
            
            # Najdeme system zprávu (měla by být první)
            system_msg = None
            for msg in messages:
                if msg['role'] == 'system':
                    system_msg = msg
                    break
            
            if not system_msg:
                print("❌ Nenalezena system zpráva!")
                return conversations
            
            # Debug: Uložení system zprávy
            if debugger:
                debugger.save_step("03_system_message", [system_msg], "System zpráva")
            
            # Projdeme všechny zprávy a najdeme user-assistant páry
            i = 0
            while i < len(messages):
                # Hledáme user zprávu
                if i < len(messages) and messages[i]['role'] == 'user':
                    user_msg = messages[i]
                    i += 1
                    
                    # Hledáme následující assistant zprávu
                    if i < len(messages) and messages[i]['role'] == 'assistant':
                        assistant_msg = messages[i]
                        i += 1
                        
                        # Vytvoříme konverzaci s system + user + assistant
                        conv_messages = [system_msg, user_msg, assistant_msg]
                        conversations.append({
                            "messages": conv_messages
                        })
                    else:
                        # Chybí assistant zpráva, přeskočíme user zprávu
                        i += 1
                else:
                    # Není user zpráva, přeskočíme
                    i += 1
            
            print(f"✅ Vytvořeno {len(conversations)} konverzací")
            
            # Debug: Uložení vytvořených konverzací
            if debugger:
                debugger.save_step("04_conversations", conversations, f"Vytvořených {len(conversations)} konverzací")
                if len(conversations) > 0:
                    debugger.save_sample("04_conversations", conversations[0], 0)
                    if len(conversations) > 1:
                        debugger.save_sample("04_conversations", conversations[1], 1)
            
            # Debug informace
            if len(conversations) > 0:
                print(f"📝 Ukázka první konverzace:")
                first_conv = conversations[0]
                for msg in first_conv['messages']:
                    print(f"  {msg['role']}: {msg['content'][:100]}...")
                
                if len(conversations) > 1:
                    print(f"📝 Ukázka druhé konverzace:")
                    second_conv = conversations[1]
                    for msg in second_conv['messages']:
                        print(f"  {msg['role']}: {msg['content'][:100]}...")
            
            return conversations
            
    except json.JSONDecodeError:
        # Není jeden velký JSON objekt, zkusíme JSONL formát
        print("📊 Zkouším JSONL formát...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        conversations.append(data)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Chyba při parsování řádku: {e}")
                        continue
        
        print(f"✅ Načteno {len(conversations)} konverzací z JSONL")
        return conversations
    
    return conversations

def prepare_training_data(conversations, debugger=None, model_name="microsoft/DialoGPT-medium", tokenizer=None):
    """Připraví data pro fine-tuning pomocí apply_chat_template"""
    training_data = []
    
    # Debug: Uložení vstupních konverzací
    if debugger:
        debugger.save_step("05_input_conversations", conversations, f"Vstupních {len(conversations)} konverzací pro prepare_training_data")
    
    # Kontrola, zda tokenizer podporuje apply_chat_template
    if not tokenizer or not hasattr(tokenizer, 'apply_chat_template'):
        error_msg = f"❌ Tokenizer pro model {model_name} nepodporuje apply_chat_template! Skript se ukončuje."
        print(error_msg)
        raise RuntimeError(error_msg)
    
    print(f"🔧 Používám apply_chat_template pro model: {model_name}")
    
    # Speciální handling pro Mistral - přidáme system message do user message
    is_mistral = "mistral" in model_name.lower()
    
    for conv in conversations:
        messages = conv['messages']
        
        # Přeskočíme konverzace bez assistant zpráv
        if not any(msg['role'] == 'assistant' for msg in messages):
            continue
        
        try:
            if is_mistral:
                # Pro Mistral přidáme system message do user message
                modified_messages = _add_system_to_user_messages(messages)
            else:
                # Pro ostatní modely použijeme původní messages
                modified_messages = messages
            
            # Použijeme apply_chat_template pro správné formátování
            formatted_text = tokenizer.apply_chat_template(modified_messages, tokenize=False, add_generation_prompt=False)
            training_data.append({"text": formatted_text})
            
        except Exception as e:
            error_msg = f"❌ Chyba při formátování: {e}"
            print(error_msg)
            raise RuntimeError(error_msg)
    
    # Debug: Uložení připravených dat
    if debugger:
        debugger.save_step("06_training_data", training_data, f"Připravených {len(training_data)} trénovacích vzorků pro {model_name}")
        if len(training_data) > 0:
            debugger.save_sample("06_training_data", training_data[0], 0)
            if len(training_data) > 1:
                debugger.save_sample("06_training_data", training_data[1], 1)
    
    return training_data

def _add_system_to_user_messages(messages):
    """Přidá system message do user message pro Mistral"""
    modified_messages = []
    system_content = None
    
    # Najdeme system message
    for msg in messages:
        if msg['role'] == 'system':
            system_content = msg['content']
            break
    
    # Projdeme všechny zprávy a upravíme user zprávy
    for msg in messages:
        if msg['role'] == 'user':
            if system_content:
                # Přidáme system message na začátek user zprávy
                combined_content = f"{system_content}\n\n{msg['content']}"
                modified_messages.append({
                    'role': 'user',
                    'content': combined_content
                })
            else:
                # Žádná system message, použijeme původní
                modified_messages.append(msg)
        elif msg['role'] == 'assistant':
            # Assistant zprávy zůstávají stejné
            modified_messages.append(msg)
        # System zprávy přeskočíme, protože je přidáme do user zpráv
    
    return modified_messages 