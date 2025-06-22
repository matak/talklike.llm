import torch
import os
import subprocess

def generate_response(model, tokenizer, prompt, max_length=200):
    """Generuje odpověď pomocí modelu (původního nebo fine-tunovaného)"""
    # Kontrola, zda tokenizer podporuje apply_chat_template
    if hasattr(tokenizer, 'apply_chat_template'):
        # Použijeme apply_chat_template pro správné formátování
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback pro tokenizery bez apply_chat_template
        formatted_prompt = prompt
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Vylepšené odstranění původního promptu z odpovědi
    if hasattr(tokenizer, 'apply_chat_template'):
        # Pro chat template - hledáme konec assistant tagu
        assistant_start = response.find("<|assistant|>")
        if assistant_start != -1:
            # Najdeme konec assistant tagu a začátek odpovědi
            response_start = assistant_start + len("<|assistant|>")
            response = response[response_start:].strip()
        else:
            # Fallback - odstraníme formatted_prompt pokud je na začátku
            if response.startswith(formatted_prompt):
                response = response[len(formatted_prompt):].strip()
    else:
        # Pro běžné prompty - odstraníme původní prompt
        if response.startswith(formatted_prompt):
            response = response[len(formatted_prompt):].strip()
    
    # Další cleanup - odstranění možných zbytků promptu
    # Hledáme běžné vzory, které by mohly zůstat
    cleanup_patterns = [
        prompt,  # Původní prompt
        f"User: {prompt}",  # S User prefixem
        f"Human: {prompt}",  # S Human prefixem
        f"<|user|>\n{prompt}",  # S user tagem
    ]
    
    for pattern in cleanup_patterns:
        if response.startswith(pattern):
            response = response[len(pattern):].strip()
            break
    
    # Odstranění prázdných řádků na začátku
    response = response.lstrip('\n').strip()
    
    return response

def test_model(model, tokenizer, test_prompts=None):
    """Otestuje model (původní nebo fine-tunovaný) na testovacích promptech"""
    if test_prompts is None:
        test_prompts = [
            "Pane Babiši, jak hodnotíte současnou inflaci?",
            "Co si myslíte o opozici?",
            "Jak se vám daří v Bruselu?"
        ]
    
    print("\n📝 Testovací odpovědi:")
    print("=" * 50)
    
    # Kontrola, zda tokenizer podporuje apply_chat_template
    if hasattr(tokenizer, 'apply_chat_template'):
        print("✅ Používá apply_chat_template pro formátování promptů")
    else:
        print("⚠️ Tokenizer nepodporuje apply_chat_template, používá se přímé formátování")
    
    for prompt in test_prompts:
        print(f"\n🔍 Prompt: {prompt}")
        
        # Debug: Zobrazení formatted promptu
        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            print(f"🔧 Formatted prompt: {formatted_prompt[:100]}...")
        
        response = generate_response(model, tokenizer, prompt)
        print(f"🎭 Odpověď: {response}")
        print("-" * 30)
    
    return test_prompts

def save_model_info(model_path, output_dir):
    """Uloží informace o modelu a vytvoří shrnutí"""
    print(f"\n💾 Ukládám model na network storage...")
    final_model_path = f"{output_dir}-final"
    
    # Vytvoření adresáře pokud neexistuje
    os.makedirs(final_model_path, exist_ok=True)
    
    # Výpis velikosti uloženého modelu
    try:
        result = subprocess.run(['du', '-sh', final_model_path], capture_output=True, text=True)
        if result.stdout:
            print(f"📊 Velikost modelu: {result.stdout.strip()}")
    except:
        pass
    
    # Výpis informací o uložených souborech
    print(f"\n📋 Uložené soubory:")
    try:
        for root, dirs, files in os.walk(final_model_path):
            level = root.replace(final_model_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Zobrazíme pouze prvních 5 souborů
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... a dalších {len(files) - 5} souborů")
    except Exception as e:
        print(f"⚠️ Nelze zobrazit seznam souborů: {e}")
    
    return final_model_path 