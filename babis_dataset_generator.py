#!/usr/bin/env python3
"""
Script pro generování datasetu satirických výroků Andreje Babiše pomocí ChatGPT.
Odešle systémový prompt z LLM.CreateAnswers.systemPrompt.md a 300 náhodných šablon z babis_templates_400.json.
Každá šablona je nahrazena jednou variantou s 15% pravděpodobností jazykové chyby a rovnoměrnou distribucí 5 stylů (±2%).
"""

import json
import os
import random
from typing import List, Dict, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import datetime
import logging
from openai_cost_calculator import OpenAICostCalculator
import time

# Načtení environment proměnných
load_dotenv()

class BabisDatasetGenerator:
    def __init__(self, templates_file: str = 'TASK/babis_templates_400.json', 
                 output_dir: str = 'generated_batches',
                 models_file: str = 'availablemodels.json'):
        """
        Inicializace generátoru s OpenAI klientem.
        
        Args:
            templates_file: Cesta k souboru se šablonami (výchozí 'TASK/babis_templates_400.json')
            output_dir: Adresář pro ukládání vygenerovaných dávek (výchozí 'generated_batches')
            models_file: Cesta k souboru s konfigurací modelů (výchozí 'availablemodels.json')
        """
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.system_prompt = self._load_system_prompt()
        self.templates_file = templates_file
        self.output_dir = output_dir
        self.responses_dir = os.path.join(output_dir, 'responses')  # Directory for raw responses
        self.content_dir = os.path.join(output_dir, 'content')  # New directory for content files
        self.templates = self._load_templates()
        self.models_config = self._load_models_config(models_file)
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)
        os.makedirs(self.content_dir, exist_ok=True)
        
        # Nastavení loggeru
        self._setup_logging()
        
    def _setup_logging(self):
        """Nastaví logger pro ukládání promptů a odpovědí."""
        # Vytvoření adresáře pro logy
        self.log_dir = 'logs'
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Vytvoření unikátního názvu souboru pro tuto session
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.log_dir, f'llm_interaction_{timestamp}.log')
        
        # Nastavení loggeru
        self.logger = logging.getLogger('llm_interaction')
        self.logger.setLevel(logging.INFO)
        
        # Handler pro soubor
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Formátování
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Přidání handleru
        self.logger.addHandler(file_handler)
        
        print(f"Logy interakcí s LLM budou uloženy do: {log_file}")

    def _log_interaction(self, batch_number: int, model: str, system_prompt: str, user_prompt: str, response_content: str = None, error: str = None):
        """Zaznamená interakci s jazykovým modelem."""
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'batch_number': batch_number,
            'model': model,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'response_content': response_content,
            'error': error
        }
        
        self.logger.info(json.dumps(log_entry, ensure_ascii=False, indent=2))
    
    def _load_system_prompt(self) -> str:
        """Načte systémový prompt z LLM.CreateAnswers.systemPrompt.md."""
        try:
            with open('TASK/LLM.CreateAnswers.systemPrompt.md', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError("Soubor TASK/LLM.CreateAnswers.systemPrompt.md nebyl nalezen")
    
    def _load_templates(self) -> List[str]:
        """Načte všechny šablony ze zadaného souboru."""
        try:
            with open(self.templates_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Soubor {self.templates_file} nebyl nalezen")
    
    def _load_models_config(self, models_file: str) -> Dict:
        """Načte konfiguraci dostupných modelů."""
        try:
            with open(models_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Soubor {models_file} nebyl nalezen")
    
    def _create_user_prompt(self, templates_batch: List[str]) -> str:
        """Vytvoří uživatelský prompt se šablonami."""
        prompt = (
            "Vytvoř satirické výroky inspirované následujícími šablonami. Pro každou šablonu vytvoř JEDEN výrok.\n\n"
            "Důležité pokyny:\n"
            "1. Placeholdery v šablonách (text v {}) nejdříve vyber seznamu placeholder_values, ale ber na vědomí, že jsou pouze INSPIRACÍ. Nenahrazuj je doslovně, ale použij je jako námět "
            "pro vytvoření originálního, kontextově vhodného obsahu.\n"
            "2. Každý výrok musí být originální interpretací šablony, ne jen nahrazením placeholderů.\n"
            "3. 15% pravděpodobnost jazykové chyby v každém výroku.\n"
            "4. Rovnoměrná distribuce 5 stylů (±2% na každý styl):\n"
            "   - Emocionální výlevy\n"
            "   - Odmítavý postoj\n"
            "   - Domýšlivost/vychloubání\n"
            "   - Chaotická logika\n"
            "   - Ironie/absurdní přirovnání\n\n"
            "Šablony pro inspiraci:\n"
        )
        prompt += json.dumps(templates_batch, ensure_ascii=False, indent=2)
        return prompt
    
    def _get_random_templates_batch(self, batch_size: int = 300) -> List[str]:
        """Vrátí náhodný výběr šablon o velikosti batch_size."""
        return random.sample(self.templates, batch_size)
    
    def _validate_response(self, response_content: str, expected_count: int) -> bool:
        """Ověří, zda odpověď obsahuje očekávaný počet výroků."""
        try:
            lines = [line.strip() for line in response_content.split('\n') if line.strip()]
            valid_lines = [line for line in lines if line.startswith('{"text":') and line.endswith('}')]
            
            if len(valid_lines) != expected_count:
                print(f"Varování: Očekáváno {expected_count} výroků, ale získáno {len(valid_lines)}")
                return False
                
            # Kontrola formátu a přítomnosti "Andrej Babiš" na konci
            for line in valid_lines:
                try:
                    data = json.loads(line)
                    if not data.get('text', '').endswith('Andrej Babiš'):
                        print("Varování: Některé výroky nekončí 'Andrej Babiš'")
                        return False
                    if '{' in data['text'] or '}' in data['text']:
                        print("Varování: Některé placeholdery nebyly nahrazeny")
                        return False
                except json.JSONDecodeError:
                    print("Varování: Neplatný JSON formát")
                    return False
            
            return True
        except Exception as e:
            print(f"Chyba při validaci: {str(e)}")
            return False

    def _save_response(self, batch_number: int, model: str, response_content: str, is_valid: bool) -> str:
        """
        Uloží surovou odpověď od OpenAI do samostatného souboru.
        
        Args:
            batch_number: Číslo dávky
            model: ID použitého modelu
            response_content: Surová odpověď od OpenAI
            is_valid: Zda odpověď prošla validací
            
        Returns:
            str: Cesta k uloženému souboru
        """
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        status = "valid" if is_valid else "invalid"
        filename = f"response_batch{batch_number:03d}_{status}_{timestamp}.json"
        filepath = os.path.join(self.responses_dir, filename)
        
        response_data = {
            "timestamp": timestamp,
            "batch_number": batch_number,
            "model": model,
            "is_valid": is_valid,
            "content": response_content
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, ensure_ascii=False, indent=2)
            
        return filepath

    def _save_content_file(self, batch_number: int, content: str) -> str:
        """
        Uloží obsah odpovědi do samostatného souboru pro další zpracování.
        
        Args:
            batch_number: Číslo dávky
            content: Obsah k uložení
            
        Returns:
            str: Cesta k uloženému souboru
        """
        filename = f"batch_{batch_number:03d}_content.txt"
        filepath = os.path.join(self.content_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return filepath

    def _check_existing_response(self, batch_number: int) -> Tuple[bool, str, bool]:
        """
        Zkontroluje, zda již existuje odpověď pro danou dávku.
        
        Args:
            batch_number: Číslo dávky ke kontrole
            
        Returns:
            Tuple[bool, str, bool]: (existuje_odpověď, obsah_odpovědi, je_validní)
        """
        # Check content directory
        content_file = os.path.join(self.content_dir, f"batch_{batch_number:03d}_content.txt")
        if os.path.exists(content_file):
            with open(content_file, 'r', encoding='utf-8') as f:
                content = f.read()
            # Check if there's a valid output file
            valid_file = os.path.join(self.output_dir, f"batch_{batch_number:02d}_babis_output.jsonl")
            return True, content, os.path.exists(valid_file)
        return False, "", False

    def generate_dataset_batch(self, batch_size: int, model: str, batch_number: int = 0) -> str:
        """
        Vygeneruje jednu dávku výroků pomocí OpenAI klienta.
        
        Args:
            batch_size: Počet šablon v dávce
            model: ID modelu k použití
            batch_number: Číslo dávky pro logování (výchozí 0)
            
        Returns:
            str: Vygenerované výroky ve formátu JSON Lines
        """
        # Nejprve zkontrolujeme existující odpovědi
        has_response, existing_content, is_valid = self._check_existing_response(batch_number)
        if has_response:
            print(f"Pro dávku {batch_number} již existuje {'validní' if is_valid else 'nevalidní'} odpověď.")
            return existing_content
            
        # Získání náhodné dávky šablon
        templates_batch = self._get_random_templates_batch(batch_size)
        expected_count = len(templates_batch)
        
        # Získání konfigurace modelu
        model_config = self.models_config['models'].get(model)
        if not model_config:
            raise ValueError(f"Model {model} není v konfiguraci")
        
        # Vytvoření uživatelského promptu pro celou dávku
        user_prompt = self._create_user_prompt(templates_batch)
        
        try:
            # Volání ChatGPT API s celou dávkou
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=model_config['max_output_tokens'],
                top_p=0.7
            )
            
            content = response.choices[0].message.content
            
            # Save content file regardless of validation
            content_file = self._save_content_file(batch_number, content)
            print(f"Obsah odpovědi uložen do: {content_file}")
            
            # Logování úspěšné interakce
            self._log_interaction(
                batch_number=batch_number,
                model=model,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                response_content=content
            )
            
            # Validace odpovědi
            is_valid = self._validate_response(content, expected_count)
            
            # Save the response
            response_file = self._save_response(batch_number, model, content, is_valid)
            print(f"Odpověď uložena do: {response_file}")
            
            return content
                
        except Exception as e:
            # Logování chyby
            self._log_interaction(
                batch_number=batch_number,
                model=model,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                error=str(e)
            )
            
            print(f"Chyba při volání ChatGPT API: {str(e)}")
            raise
            
        raise Exception("Nepodařilo se získat odpověď")

    def save_batch_to_file(self, batch_content: str, filename: str):
        """Uloží vygenerovanou dávku do souboru."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(batch_content)
            print(f"Dávka uložena do souboru: {filename}")
        except Exception as e:
            raise Exception(f"Chyba při ukládání do souboru {filename}: {str(e)}")
    
    

