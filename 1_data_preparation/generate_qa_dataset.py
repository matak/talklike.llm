#!/usr/bin/env python3
"""
Script pro generování fine-tuning datasetu pomocí BabisDialogGenerator.
Zpracovává všechny batch soubory z data/generated_batches a vytváří QA páry.
"""

import os
from dotenv import load_dotenv
import glob
import json
from lib.babis_dialog_generator import BabisDialogGenerator
from lib.openai_cost_calculator import OpenAICostCalculator

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

def main():
    """Hlavní funkce pro generování QA datasetu."""
    print("=== Generování QA datasetu pomocí BabisDialogGenerator ===")
    
    # Kontrola API klíče
    if not os.getenv('OPENAI_API_KEY'):
        print("CHYBA: OPENAI_API_KEY není nastaven")
        print("Prosím nastavte OPENAI_API_KEY v .env souboru")
        exit(1)
    
    try:
        # Inicializace generátoru a kalkulátoru nákladů
        generator = BabisDialogGenerator()
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
        
        # Získání batch souborů
        pattern = os.path.join('../data/generated_batches', 'batch_*_babis_output.jsonl')
        batch_files = glob.glob(pattern)
        
        # Filtrujeme pouze validní soubory (ne invalid)
        valid_files = [f for f in batch_files if 'invalid' not in f]
        valid_files = sorted(valid_files)
        
        if not valid_files:
            print("Nebyly nalezeny žádné batch soubory v adresáři data/generated_batches/")
            return
        
        print(f"Nalezeno {len(valid_files)} batch souborů")
        
        # Výpočet celkového počtu odpovědí
        total_answers = 0
        for batch_file in valid_files:
            with open(batch_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        total_answers += 1
        
        print(f"Celkový počet odpovědí: {total_answers}")
        
        # Vytvoření skutečného promptu pro výpočet ceny
        sample_answers = []
        for batch_file in valid_files[:1]:  # Použijeme první batch pro ukázku
            with open(batch_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        sample_answers.append(data['text'])
                        if len(sample_answers) >= 10:  # Použijeme prvních 10 odpovědí (velikost dávky)
                            break
                    if len(sample_answers) >= 10:
                        break
            if len(sample_answers) >= 10:
                break
        
        if sample_answers:
            system_prompt = generator.system_prompt
            user_prompt = generator._create_user_prompt(sample_answers)
            input_text = system_prompt + "\n\n" + user_prompt
            
            # Odhad ceny
            output_size_coefficient = 2.0  # Koeficient velikosti výstupu
            calculate_generation_cost(
                cost_calculator,
                input_text=input_text,
                output_size_coefficient=output_size_coefficient,
                num_batches=len(valid_files),
                model=model
            )
        
        # Dotaz na uživatele pro potvrzení
        while True:
            user_input = input("\nChcete pokračovat v generování QA datasetu? (ano/ne): ").lower().strip()
            if user_input in ['ano', 'ne']:
                break
            print("Prosím odpovězte 'ano' nebo 'ne'")
        
        if user_input == 'ne':
            print("Generování QA datasetu zrušeno uživatelem.")
            return
        
        # Zpracování všech batch souborů
        print("\n=== Zpracovávám batch soubory ===")
        processed_files = generator.process_all_batches(
            input_dir='../data/generated_batches',
            output_dir='../data/final',
            model=model,
            batch_size=10  # Zpracováváme po 10 odpovědích najednou
        )
        
        if processed_files:
            print(f"\n=== QA dataset úspěšně vytvořen ===")
            print(f"Počet zpracovaných souborů: {len(processed_files)}")
            print(f"Výstupní adresář: ../data/final/")
            print("\nZpracované soubory:")
            for file in processed_files:
                print(f"  - {file}")
        else:
            print("Nepodařilo se zpracovat žádné soubory")
        
    except Exception as e:
        print(f"Chyba při generování QA datasetu: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 