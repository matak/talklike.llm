#!/usr/bin/env python3
"""
Library pro generování dialogů s Andrejem Babišem.
Používá systémový prompt LLM.CreateDialogue.systemPrompt.md pro generování otázek k odpovědím.
"""

import json
import os
import glob
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv
import datetime
import logging
from openai_cost_calculator import OpenAICostCalculator
import time

# Načtení environment proměnných
load_dotenv()

class BabisDialogGenerator:
    def __init__(self, models_file: str = 'availablemodels.json'):
        """
        Inicializace generátoru dialogů.
        
        Args:
            models_file: Cesta k souboru s konfigurací modelů (výchozí 'availablemodels.json')
        """
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.system_prompt = self._load_system_prompt()
        self.models_config = self._load_models_config(models_file)
        
        # Nastavení loggeru
        self._setup_logging()
        
    def _setup_logging(self):
        """Nastaví logger pro ukládání promptů a odpovědí."""
        # Vytvoření adresáře pro logy
        self.log_dir = 'logs'
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Vytvoření unikátního názvu souboru pro tuto session
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.log_dir, f'babis_dialog_{timestamp}.log')
        
        # Nastavení loggeru
        self.logger = logging.getLogger('babis_dialog')
        self.logger.setLevel(logging.INFO)
        
        # Handler pro soubor
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Formátování
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Přidání handleru
        self.logger.addHandler(file_handler)
        
        print(f"Logy dialogů budou uloženy do: {log_file}")

    def _log_interaction(self, batch_name: str, model: str, system_prompt: str, user_prompt: str, response_content: str = None, error: str = None):
        """Zaznamená interakci s jazykovým modelem."""
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'batch_name': batch_name,
            'model': model,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'response_content': response_content,
            'error': error
        }
        
        self.logger.info(json.dumps(log_entry, ensure_ascii=False, indent=2))
    
    def _load_system_prompt(self) -> str:
        """Načte systémový prompt z LLM.CreateDialogue.systemPrompt.md."""
        try:
            with open('TASK/LLM.CreateDialogue.systemPrompt.md', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError("Soubor TASK/LLM.CreateDialogue.systemPrompt.md nebyl nalezen")
    
    def _load_models_config(self, models_file: str) -> Dict:
        """Načte konfiguraci dostupných modelů."""
        try:
            with open(models_file, 'r', encoding='utf-8') as f:
                return json.load(f)['models']
        except FileNotFoundError:
            raise FileNotFoundError(f"Soubor {models_file} nebyl nalezen")
    
    def _create_user_prompt(self, answers: List[str]) -> str:
        """Vytvoří uživatelský prompt s odpověďmi pro generování otázek."""
        prompt = (
            "Pro každou z následujících odpovědí Andreje Babiše vytvoř odpovídající otázku redaktora.\n\n"
            "Odpovědi:\n"
        )
        
        for i, answer in enumerate(answers, 1):
            prompt += f"{i}. {answer}\n"
        
        prompt += "\nVytvoř otázky ve formátu JSONL (jeden JSON objekt na řádek):"
        return prompt
    
    def _validate_response(self, response_content: str, expected_count: int) -> bool:
        """Ověří, zda odpověď obsahuje očekávaný počet otázek."""
        try:
            lines = [line.strip() for line in response_content.split('\n') if line.strip()]
            valid_lines = [line for line in lines if line.startswith('{"question":') and line.endswith('}')]
            
            if len(valid_lines) != expected_count:
                print(f"Varování: Očekáváno {expected_count} otázek, ale získáno {len(valid_lines)}")
                return False
                
            # Kontrola formátu
            for line in valid_lines:
                try:
                    data = json.loads(line)
                    if 'question' not in data:
                        print("Varování: Některé řádky neobsahují 'question' pole")
                        return False
                except json.JSONDecodeError:
                    print("Varování: Neplatný JSON formát")
                    return False
            
            return True
        except Exception as e:
            print(f"Chyba při validaci: {str(e)}")
            return False

    def generate_questions_for_answers(self, answers: List[str], model: str = None, batch_name: str = "unknown", batch_size: int = 10) -> List[str]:
        """
        Vygeneruje otázky pro seznam odpovědí pomocí batching.
        
        Args:
            answers: Seznam odpovědí Andreje Babiše
            model: ID modelu pro generování (pokud není specifikován, použije se výchozí)
            batch_name: Název dávky pro logování
            batch_size: Velikost dávky pro zpracování (výchozí 10)
            
        Returns:
            List[str]: Seznam vygenerovaných otázek
        """

        print(f"DEBUG: Počet odpovědí: {len(answers)}")
        print(f"DEBUG: Model: {model}")
        print(f"DEBUG: Batch name: {batch_name}")
        print(f"DEBUG: Batch size: {batch_size}")

        if not answers:
            return []
        
        # Výběr modelu
        if not model:
            available_models = self.models_config
            model = None
            for model_id, config in available_models.items():
                if config.get('default', 0) == 1:
                    model = model_id
                    break
            if not model:
                model = next(iter(available_models))

        print(f"Generuji otázky pro {len(answers)} odpovědí pomocí modelu {self.models_config[model]['name']} v dávkách po {batch_size}")
        
        all_questions = []
        
        # Rozdělení odpovědí na dávky
        for i in range(0, len(answers), batch_size):
            batch_answers = answers[i:i + batch_size]
            batch_start = i + 1
            batch_end = min(i + batch_size, len(answers))
            
            print(f"Zpracovávám dávku {batch_start}-{batch_end} ({len(batch_answers)} odpovědí)")
            
            # Vytvoření promptu pro tuto dávku
            user_prompt = self._create_user_prompt(batch_answers)
            
            # Záznam interakce
            batch_subname = f"{batch_name}_batch_{batch_start}-{batch_end}"
            self._log_interaction(batch_subname, model, self.system_prompt, user_prompt)
            
            try:
                # Volání OpenAI API
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
                
                response_content = response.choices[0].message.content
                
                # Validace odpovědi
                is_valid = self._validate_response(response_content, len(batch_answers))
                
                if not is_valid:
                    print(f"Varování: Odpověď pro {batch_subname} není validní")
                
                # Parsování otázek
                batch_questions = []
                for line in response_content.split('\n'):
                    line = line.strip()
                    if line and line.startswith('{"question":'):
                        try:
                            data = json.loads(line)
                            batch_questions.append(data['question'])
                        except json.JSONDecodeError:
                            continue
                
                # Kontrola, zda máme správný počet otázek pro tuto dávku
                if len(batch_questions) != len(batch_answers):
                    print(f"Varování: Počet otázek ({len(batch_questions)}) neodpovídá počtu odpovědí ({len(batch_answers)}) v dávce {batch_start}-{batch_end}")
                    # Doplnění chybějících otázek nebo ořezání přebytečných
                    if len(batch_questions) < len(batch_answers):
                        for j in range(len(batch_answers) - len(batch_questions)):
                            batch_questions.append(f"Otázka {batch_start + len(batch_questions) + j}")
                    else:
                        batch_questions = batch_questions[:len(batch_answers)]
                
                all_questions.extend(batch_questions)
                print(f"Dávka {batch_start}-{batch_end} dokončena, získáno {len(batch_questions)} otázek")
                
                # Krátká pauza mezi dávkami
                time.sleep(1)
                
            except Exception as e:
                error_msg = f"Chyba při generování otázek pro {batch_subname}: {str(e)}"
                print(error_msg)
                self._log_interaction(batch_subname, model, self.system_prompt, user_prompt, error=error_msg)
                raise
        
        print(f"Celkem vygenerováno {len(all_questions)} otázek pro {len(answers)} odpovědí")
        return all_questions

    def create_qa_pairs(self, answers: List[str], model: str = None, batch_name: str = "unknown", batch_size: int = 10) -> List[Dict[str, str]]:
        """
        Vytvoří páry otázka-odpověď.
        
        Args:
            answers: Seznam odpovědí Andreje Babiše
            model: ID modelu pro generování
            batch_name: Název dávky pro logování
            batch_size: Velikost dávky pro zpracování (výchozí 10)
            
        Returns:
            List[Dict[str, str]]: Seznam QA párů ve formátu {"question": "...", "answer": "..."}
        """
        questions = self.generate_questions_for_answers(answers, model, batch_name, batch_size)
        
        qa_pairs = []
        for answer, question in zip(answers, questions):
            qa_pair = {
                "question": question,
                "answer": answer
            }
            qa_pairs.append(qa_pair)
        
        return qa_pairs

    def process_batch_file(self, batch_file: str, output_file: str, model: str = None, batch_size: int = 10) -> bool:
        """
        Zpracuje batch soubor a vytvoří QA páry.
        
        Args:
            batch_file: Cesta k batch souboru s odpověďmi
            output_file: Cesta k výstupnímu souboru s QA páry
            model: ID modelu pro generování
            batch_size: Velikost dávky pro zpracování (výchozí 10)
            
        Returns:
            bool: True pokud zpracování bylo úspěšné
        """
        try:
            # Načtení odpovědí
            answers = []
            with open(batch_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        answers.append(data['text'])
            
            if not answers:
                print(f"Soubor {batch_file} neobsahuje žádné odpovědi")
                return False
            
            print(f"Načteno {len(answers)} odpovědí ze souboru {batch_file}")
            
            # Vytvoření QA párů
            batch_name = os.path.basename(batch_file).replace('.jsonl', '')

            print(f"DEBUG: Batch name: {batch_name}")
            qa_pairs = self.create_qa_pairs(answers, model, batch_name, batch_size)
            
            # Uložení do souboru
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                for qa_pair in qa_pairs:
                    f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
            
            print(f"QA páry uloženy do {output_file}")
            return True
            
        except Exception as e:
            print(f"Chyba při zpracování souboru {batch_file}: {str(e)}")
            return False

    def process_all_batches(self, input_dir: str = 'generated_batches', output_dir: str = 'final', model: str = None, batch_size: int = 10) -> List[str]:
        """
        Zpracuje všechny batch soubory v adresáři.
        
        Args:
            input_dir: Adresář s batch soubory
            output_dir: Adresář pro výstupní soubory
            model: ID modelu pro generování
            batch_size: Velikost dávky pro zpracování (výchozí 10)
            
        Returns:
            List[str]: Seznam úspěšně zpracovaných souborů
        """
        # Získání batch souborů
        pattern = os.path.join(input_dir, 'batch_*_babis_output.jsonl')
        batch_files = glob.glob(pattern)
        
        # Filtrujeme pouze validní soubory (ne invalid)
        valid_files = [f for f in batch_files if 'invalid' not in f]
        valid_files = sorted(valid_files)
        
        if not valid_files:
            print(f"Nebyly nalezeny žádné batch soubory v adresáři {input_dir}/")
            return []
        
        print(f"Nalezeno {len(valid_files)} batch souborů")
        
        # Vytvoření výstupního adresáře
        os.makedirs(output_dir, exist_ok=True)
        
        processed_files = []
        
        for i, batch_file in enumerate(valid_files, 1):
            batch_name = os.path.basename(batch_file).replace('.jsonl', '')
            output_file = os.path.join(output_dir, f"{batch_name}_qa.jsonl")
            
            print(f"\nZpracovávám {i}/{len(valid_files)}: {batch_name}")
            
            if self.process_batch_file(batch_file, output_file, model, batch_size):
                processed_files.append(output_file)
            
            # Krátká pauza mezi soubory
            time.sleep(1)
        
        return processed_files

    def estimate_cost(self, answers: List[str], model: str = None) -> Dict:
        """
        Odhadne cenu generování otázek pro dané odpovědi.
        
        Args:
            answers: Seznam odpovědí
            model: ID modelu
            
        Returns:
            Dict: Informace o odhadované ceně
        """
        if not model:
            available_models = self.models_config
            for model_id, config in available_models.items():
                if config.get('default', 0) == 1:
                    model = model_id
                    break
            if not model:
                model = next(iter(available_models))
        
        cost_calculator = OpenAICostCalculator()
        
        # Vytvoření ukázkového promptu
        sample_answers = answers[:5] if len(answers) > 5 else answers
        sample_user_prompt = self._create_user_prompt(sample_answers)
        
        # Výpočet ceny
        input_tokens = cost_calculator.estimate_tokens(self.system_prompt + "\n\n" + sample_user_prompt, model)
        output_tokens = int(input_tokens * 2.0)  # Odpověď bude přibližně 2x větší než vstup
        simulated_output = "x" * output_tokens
        
        batch_cost, token_counts = cost_calculator.estimate_batch_cost(
            self.system_prompt + "\n\n" + sample_user_prompt, 
            simulated_output, 
            model
        )
        
        # Výpočet pro všechny odpovědi
        total_batches = (len(answers) + 4) // 5  # Zaokrouhleno nahoru
        total_cost = batch_cost * total_batches
        
        return {
            "model": model,
            "model_name": self.models_config[model]['name'],
            "total_answers": len(answers),
            "estimated_batches": total_batches,
            "cost_per_batch": batch_cost,
            "total_cost": total_cost,
            "formatted_total_cost": cost_calculator.format_price(total_cost)
        } 