from abc import ABC, abstractmethod
from typing import Dict, Tuple

class LLMCostCalculator(ABC):
    """Abstraktní rozhraní pro kalkulaci nákladů LLM modelů."""
    
    @abstractmethod
    def estimate_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Spočítá počet tokenů v textu.
        
        Args:
            text: Text k tokenizaci
            model: ID modelu pro který počítáme tokeny
            
        Returns:
            int: Počet tokenů
        """
        pass
        
    @abstractmethod
    def check_token_limits(self, system_tokens: int, user_tokens: int, 
                          expected_output_tokens: int, model: str) -> Tuple[bool, str]:
        """Zkontroluje, zda počet tokenů nepřekračuje limity modelu."""
        pass
        
    @abstractmethod
    def estimate_batch_cost(self, input_text: str, output_text: str, model: str) -> Tuple[float, Dict[str, int]]:
        """
        Vypočítá cenu za zpracování na základě skutečného textu.
        
        Args:
            input_text: Vstupní text pro zpracování
            output_text: Výstupní text (odpověď)
            model: ID modelu pro který počítáme cenu
            
        Returns:
            Tuple[float, Dict[str, int]]: (cena v USD, slovník s počty tokenů)
        """
        pass
        
    @abstractmethod
    def format_price(self, price: float) -> str:
        """Formátuje cenu do čitelného formátu."""
        pass
        
    @abstractmethod
    def get_available_models(self) -> Dict:
        """Vrátí dostupné modely a jejich konfiguraci."""
        pass 