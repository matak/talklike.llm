import json
import os
from datetime import datetime

def load_babis_data(file_path, debugger=None):
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

def prepare_training_data(conversations, debugger=None, model_name="microsoft/DialoGPT-medium"):
    """Připraví data pro fine-tuning"""
    training_data = []
    
    # Debug: Uložení vstupních konverzací
    if debugger:
        debugger.save_step("05_input_conversations", conversations, f"Vstupních {len(conversations)} konverzací pro prepare_training_data")
    
    # Detekce typu modelu pro správný formát
    is_mistral = "mistral" in model_name.lower()
    is_llama = "llama" in model_name.lower()
    is_dialogpt = "dialogpt" in model_name.lower()
    
    print(f"🔧 Detekován model typ: {'Mistral' if is_mistral else 'Llama' if is_llama else 'DialoGPT' if is_dialogpt else 'Unknown'}")
    
    for conv in conversations:
        messages = conv['messages']
        
        # Přeskočíme konverzace bez assistant zpráv
        if not any(msg['role'] == 'assistant' for msg in messages):
            continue
            
        # Vytvoříme text pro fine-tuning podle typu modelu
        text = ""
        
        if is_mistral:
            # Mistral používá ChatML formát
            for msg in messages:
                if msg['role'] == 'system':
                    text += f"<s>[INST] {msg['content']} [/INST]"
                elif msg['role'] == 'user':
                    text += f"<s>[INST] {msg['content']} [/INST]"
                elif msg['role'] == 'assistant':
                    text += f" {msg['content']} </s>"
        elif is_llama:
            # Llama používá podobný formát jako Mistral
            for msg in messages:
                if msg['role'] == 'system':
                    text += f"<s>[INST] <<SYS>>\n{msg['content']}\n<</SYS>>\n\n [/INST]"
                elif msg['role'] == 'user':
                    text += f"<s>[INST] {msg['content']} [/INST]"
                elif msg['role'] == 'assistant':
                    text += f" {msg['content']} </s>"
        else:
            # DialoGPT a jiné modely - původní formát
            for msg in messages:
                if msg['role'] == 'system':
                    text += f"<|system|>\n{msg['content']}<|end|>\n"
                elif msg['role'] == 'user':
                    text += f"<|user|>\n{msg['content']}<|end|>\n"
                elif msg['role'] == 'assistant':
                    text += f"<|assistant|>\n{msg['content']}<|end|>\n"
        
        training_data.append({"text": text})
    
    # Debug: Uložení připravených dat
    if debugger:
        debugger.save_step("06_training_data", training_data, f"Připravených {len(training_data)} trénovacích vzorků pro {model_name}")
        if len(training_data) > 0:
            debugger.save_sample("06_training_data", training_data[0], 0)
            if len(training_data) > 1:
                debugger.save_sample("06_training_data", training_data[1], 1)
    
    return training_data 