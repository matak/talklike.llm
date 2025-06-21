import json
from typing import Dict, Tuple
import tiktoken
from llm_cost_calculator import LLMCostCalculator

class OpenAICostCalculator(LLMCostCalculator):
    """Implementace kalkulátoru nákladů pro OpenAI modely."""
    
    def __init__(self, models_file: str = 'availablemodels.json'):
        """
        Inicializace kalkulátoru.
        
        Args:
            models_file: Cesta k souboru s konfigurací modelů
        """
        self.models_file = models_file
        self.available_models = self._load_models()
        self.encoders = {}  # Cache pro encodery
        
    def _load_models(self) -> Dict:
        """Načte konfiguraci modelů ze souboru."""
        try:
            with open(self.models_file, 'r', encoding='utf-8') as f:
                return json.load(f)['models']
        except FileNotFoundError:
            print(f"VAROVÁNÍ: Soubor {self.models_file} nebyl nalezen, používám výchozí konfiguraci")
            return {
                "gpt-3.5-turbo": {
                    "name": "GPT-3.5 Turbo",
                    "prices": {"input": 0.0005, "output": 0.0015},
                    "batch_processing": True,
                    "max_batch_size": 4096,
                    "default": 1,
                    "description": "Výchozí model"
                }
            }
            
    def _get_encoder(self, model: str) -> tiktoken.Encoding:
        """
        Získá nebo vytvoří encoder pro daný model.
        
        Args:
            model: ID modelu (např. 'gpt-3.5-turbo')
            
        Returns:
            tiktoken.Encoding: Encoder pro daný model
        """
        if model not in self.encoders:
            try:
                self.encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback na cl100k_base pro nové modely
                self.encoders[model] = tiktoken.get_encoding("cl100k_base")
        return self.encoders[model]
        
    def estimate_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Spočítá přesný počet tokenů v textu pomocí tiktoken.
        
        Args:
            text: Text k tokenizaci
            model: ID modelu pro který počítáme tokeny
            
        Returns:
            int: Počet tokenů
        """
        encoder = self._get_encoder(model)
        tokens = encoder.encode(text)
        print(f"\n=== Debug tokenizace ===")
        print(f"Model: {model}")
        print(f"Délka textu: {len(text)} znaků")
        print(f"Počet tokenů: {len(tokens)}")
        print(f"Text na vstupu: {text}")
        print("\nVšechny tokeny a jejich dekódované hodnoty:")
        for i, token in enumerate(tokens):
            decoded = encoder.decode([token])
            print(f"Token {i}: ID={token}, Decoded='{decoded}'")
        return len(tokens)
        
    def check_token_limits(self, system_tokens: int, user_tokens: int, 
                          expected_output_tokens: int, model: str) -> Tuple[bool, str]:
        """
        Zkontroluje, zda počet tokenů nepřekračuje limity modelu.
        
        Returns:
            Tuple[bool, str]: (je_v_limitu, důvod_překročení)
        """
        if model not in self.available_models:
            raise ValueError(f"Nepodporovaný model: {model}")
            
        model_config = self.available_models[model]
        total_input_tokens = system_tokens + user_tokens
        
        # Kontrola vstupního kontextu
        if total_input_tokens > model_config['context_window']:
            return False, f"Celkový počet vstupních tokenů ({total_input_tokens:,}) překračuje limit modelu ({model_config['context_window']:,})"
            
        # Kontrola výstupních tokenů
        if expected_output_tokens > model_config['max_output_tokens']:
            return False, f"Očekávaný počet výstupních tokenů ({expected_output_tokens:,}) překračuje limit modelu ({model_config['max_output_tokens']:,})"
            
        # Kontrola celkového kontextu
        total_tokens = total_input_tokens + expected_output_tokens
        if total_tokens > model_config['context_window']:
            return False, f"Celkový počet tokenů ({total_tokens:,}) překračuje limit kontextu modelu ({model_config['context_window']:,})"
            
        return True, ""
        
    def estimate_batch_cost(self, input_text: str, output_text: str, model: str) -> Tuple[float, Dict[str, int]]:
        """
        Vypočítá cenu za zpracování na základě skutečného počtu tokenů.
        
        Args:
            input_text: Vstupní text pro zpracování
            output_text: Výstupní text (odpověď)
            model: ID modelu pro který počítáme cenu
            
        Returns:
            Tuple[float, Dict[str, int]]: (cena v USD, slovník s počty tokenů)
        """
        if model not in self.available_models:
            raise ValueError(f"Nepodporovaný model: {model}")
            
        model_config = self.available_models[model]
        
        # Spočítání skutečného počtu tokenů
        print("\n=== Debug vstupního textu ===")
        input_tokens = self.estimate_tokens(input_text, model)
        print("\n=== Debug výstupního textu ===")
        output_tokens = self.estimate_tokens(output_text, model)
        
        # Výběr správného ceníku podle velikosti dávky
        price_type = "batch" if input_tokens > model_config['max_batch_size'] else "non_batch"
        prices = model_config['prices'][price_type]
        
        print(f"\n=== Debug výpočtu ceny ===")
        print(f"Typ ceny: {price_type}")
        print(f"Cena za vstup: ${prices['input']} za 1M tokenů")
        print(f"Cena za výstup: ${prices['output']} za 1M tokenů")
        
        # Výpočet ceny (ceny jsou za 1M tokenů)
        input_cost = (input_tokens / 1_000_000) * prices['input']
        output_cost = (output_tokens / 1_000_000) * prices['output']
        total_cost = input_cost + output_cost
        
        print(f"Cena za vstup: ${input_cost:.4f}")
        print(f"Cena za výstup: ${output_cost:.4f}")
        print(f"Celková cena: ${total_cost:.4f}")
        
        token_counts = {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens
        }
        
        return total_cost, token_counts
        
    def format_price(self, price: float) -> str:
        """Formátuje cenu do čitelného formátu."""
        if price < 0.01:
            return f"{price * 100:.2f} centů"
        return f"${price:.2f}"
        
    def get_available_models(self) -> Dict:
        """Vrátí dostupné modely a jejich konfiguraci."""
        return self.available_models 