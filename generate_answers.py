#!/usr/bin/env python3
"""
Script pro generování všech dávek datasetu z babis_templates_400.json.
Generuje 10 dávek po 300 náhodných šablonách (1 výrok na šablonu) pro celkem 3000 výroků.
"""

from babis_dataset_generator import BabisDatasetGenerator
from openai_cost_calculator import OpenAICostCalculator
import os
from dotenv import load_dotenv
import json

# Načtení environment proměnných
load_dotenv()

def calculate_generation_cost(cost_calculator: OpenAICostCalculator, input_text: str, output_size_coefficient: float, num_batches: int, model: str) -> None:
    """Vypočítá a zobrazí cenu generování na základě skutečného textu."""
    # Výpočet velikosti výstupu jako násobek vstupu
    input_tokens = cost_calculator.estimate_tokens(input_text, model)
    output_tokens = int(input_tokens * output_size_coefficient)
    
    # Vytvoření simulovaného výstupu odpovídající velikosti
    simulated_output = "x" * output_tokens  # Jednoduchá simulace textu dané délky
    
    # Výpočet ceny ze skutečných dat pro jednu dávku
    batch_cost, token_counts = cost_calculator.estimate_batch_cost(input_text, simulated_output, model)
    
    # Výpočet celkové ceny pro všechny dávky
    total_cost = batch_cost * num_batches
    total_tokens = {
        "input": token_counts["input"] * num_batches,
        "output": token_counts["output"] * num_batches,
        "total": token_counts["total"] * num_batches
    }
    
    # Zobrazení informací o ceně
    print("\n=== Výpočet ceny generování ===")
    print(f"Model: {cost_calculator.available_models[model]['name']} ({model})")
    print(f"Počet dávek: {num_batches}")
    print(f"Koeficient velikosti výstupu: {output_size_coefficient:.2f}")
    print(f"\nTokeny na dávku:")
    print(f"  - Vstup: {token_counts['input']:,}")
    print(f"  - Výstup (koeficient {output_size_coefficient:.2f}): {token_counts['output']:,}")
    print(f"  - Celkem na dávku: {token_counts['total']:,}")
    print(f"\nCelkové tokeny ({num_batches} dávek):")
    print(f"  - Vstup celkem: {total_tokens['input']:,}")
    print(f"  - Výstup celkem: {total_tokens['output']:,}")
    print(f"  - Celkem všech tokenů: {total_tokens['total']:,}")
    print(f"\nCena za dávku: {cost_calculator.format_price(batch_cost)}")
    print(f"Celková cena ({num_batches} dávek): {cost_calculator.format_price(total_cost)}")

def generate_all_batches():
    """Vygeneruje 10 dávek po 300 náhodných šablonách."""
    print("=== Generování 10 dávek po 300 šablonách ===")
    
    # Inicializace kalkulátoru nákladů
    cost_calculator = OpenAICostCalculator()
    
    # Výběr výchozího modelu
    available_models = cost_calculator.get_available_models()
    model = None
    for model_id, config in available_models.items():
        if config.get('default', 0) == 1:
            model = model_id
            break
    if not model:
        model = next(iter(available_models))
    
    # Inicializace generátoru s explicitními parametry
    templates_file = 'TASK/babis_templates_400.json'
    output_dir = "generated_batches"
    invalid_dir = os.path.join(output_dir, "invalid")  # Directory for invalid responses
    content_dir = os.path.join(output_dir, "content")  # Directory for content files
    
    # Create all necessary directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(invalid_dir, exist_ok=True)
    os.makedirs(content_dir, exist_ok=True)
    
    generator = BabisDatasetGenerator(templates_file=templates_file, output_dir=output_dir)
    
    # Nastavení parametrů generování
    num_batches = 10
    batch_size = 150
    output_size_coefficient = 1.3  # Koeficient velikosti výstupu (výstup bude 1.3x větší než vstup)
    
    print(f"Celkem šablon na dávku: {batch_size}")
    print(f"Počet dávek: {num_batches}")
    print(f"Celkový počet výroků: {num_batches * batch_size}")
    print(f"Koeficient velikosti výstupu: {output_size_coefficient:.2f}")
    print(f"Výstupní adresář: {output_dir}")
    print(f"Adresář pro nevalidní odpovědi: {invalid_dir}")
    print(f"Adresář pro obsah odpovědí: {content_dir}")
    
    # Vytvoření výstupního adresáře
    os.makedirs(output_dir, exist_ok=True)
    
    # Získání náhodné dávky šablon pro ukázku
    templates_batch = generator._get_random_templates_batch(batch_size)
    
    # Vytvoření skutečného promptu pro výpočet ceny
    system_prompt = generator.system_prompt
    user_prompt = generator._create_user_prompt(templates_batch)
    
    # Výpočet a zobrazení ceny na základě skutečných promptů
    calculate_generation_cost(
        cost_calculator,
        input_text=system_prompt + "\n\n" + user_prompt,
        output_size_coefficient=output_size_coefficient,
        num_batches=num_batches,
        model=model
    )
    
    # Dotaz na uživatele pro potvrzení
    while True:
        user_input = input("\nChcete pokračovat v generování? (ano/ne): ").lower().strip()
        if user_input in ['ano', 'ne']:
            break
        print("Prosím odpovězte 'ano' nebo 'ne'")
    
    if user_input == 'ne':
        print("Generování zrušeno uživatelem.")
        return
    
    # Generování všech dávek
    try:
        for i in range(num_batches):
            batch_num = i + 1
            filename = os.path.join(output_dir, f"batch_{batch_num:02d}_babis_output.jsonl")
            invalid_filename = os.path.join(invalid_dir, f"batch_{batch_num:02d}_babis_output_invalid.jsonl")
            content_file = os.path.join(content_dir, f"batch_{batch_num:03d}_content.txt")
            
            # Kontrola existujících souborů
            if os.path.exists(filename):
                print(f"\nValidní dávka {batch_num}/{num_batches} již existuje v {filename}, přeskakuji...")
                continue
            elif os.path.exists(content_file):
                print(f"\nPro dávku {batch_num}/{num_batches} již existuje odpověď v {content_file}, přeskakuji...")
                continue
                
            print(f"\nGeneruji dávku {batch_num}/{num_batches}...")
            
            try:
                batch_content = generator.generate_dataset_batch(
                    batch_size=batch_size,
                    model=model,
                    batch_number=batch_num
                )
                
                # Validate the response
                if generator._validate_response(batch_content, batch_size):
                    generator.save_batch_to_file(batch_content, filename)
                    print(f"Dávka {batch_num} úspěšně vygenerována a uložena do {filename}")
                else:
                    # Save invalid response to separate directory
                    generator.save_batch_to_file(batch_content, invalid_filename)
                    print(f"Dávka {batch_num} není validní, uložena do {invalid_filename}")
                    
            except Exception as e:
                print(f"Chyba při generování dávky {batch_num}: {str(e)}")
                continue
        
    except ValueError as e:
        print(f"Chyba při generování: {str(e)}")
        return
        
    except Exception as e:
        print(f"Neočekávaná chyba: {str(e)}")
        return
    
    print(f"\n=== Generování dokončeno ===")
    print(f"Validní dávky byly uloženy v adresáři: {output_dir}")
    print(f"Nevalidní dávky byly uloženy v adresáři: {invalid_dir}")
    print(f"Obsah všech odpovědí byl uložen v adresáři: {content_dir}")
    print(f"Surové odpovědi od OpenAI byly uloženy v adresáři: {os.path.join(output_dir, 'responses')}")

if __name__ == "__main__":
    # Kontrola API klíče
    if not os.getenv('OPENAI_API_KEY'):
        print("CHYBA: OPENAI_API_KEY není nastaven")
        print("Prosím nastavte OPENAI_API_KEY v .env souboru")
        exit(1)
    
    try:
        # Spuštění generování všech dávek
        generate_all_batches()
        
    except Exception as e:
        print(f"Chyba při spuštění generování: {str(e)}")
        exit(1) 