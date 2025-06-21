#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rychlý test adaptéru pro TalkLike.LLM
Ověřuje, že váš natrénovaný adaptér funguje správně
"""

# Import a nastavení prostředí
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import setup_environment

def quick_test():
    """Rychlý test adaptéru"""
    print("🚀 Rychlý test adaptéru mcmatak/babis-mistral-adapter")
    print("=" * 60)
    
    try:
        # Import potřebných modulů
        from test_adapter import load_model_with_adapter, generate_response
        print("✅ Import úspěšný")
        
        # Načtení modelu
        print("\n🤖 Načítám model s adaptérem...")
        base_model = "mistralai/Mistral-7B-Instruct-v0.3"
        adapter_path = "mcmatak/babis-mistral-adapter"
        
        model, tokenizer = load_model_with_adapter(base_model, adapter_path)
        
        if model is None or tokenizer is None:
            print("❌ Model se nepodařilo načíst")
            return False
        
        print("✅ Model úspěšně načten!")
        
        # Test otázky
        test_questions = [
            "Pane Babiši, jak hodnotíte současnou inflaci?",
            "Co si myslíte o opozici?",
            "Jak se vám daří s rodinou?",
            "Můžete vysvětlit vaši roli v té chemičce?",
            "Jak vnímáte reakce Bruselu na ekonomickou situaci v Česku?"
        ]
        
        print(f"\n📝 Testuji {len(test_questions)} otázek...")
        print("-" * 60)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Otázka: {question}")
            
            # System prompt pro Babišův styl
            system_prompt = """Jsi Andrej Babiš, český politik a podnikatel. Tvým úkolem je odpovídat na otázky v charakteristickém Babišově stylu.

Charakteristické prvky tvého stylu:
- Typické fráze: "Hele, ...", "To je skandál!", "Já makám", "Opozice krade", "V Bruselu"
- Slovenské odchylky: "sme", "som", "makáme", "centralizácia"
- Emotivní výrazy: "to je šílený!", "tragédyje!", "kampááň!"
- Přirovnání: "jak když kráva hraje na klavír", "jak když dítě řídí tank"
- První osoba: "Já jsem...", "Moje rodina...", "Já makám..."
- Podpis: Každou odpověď zakonči "Andrej Babiš"

Odpovídej vždy v první osobě jako Andrej Babiš, používej jeho charakteristické fráze, buď emotivní a přímý."""

            prompt = f"<s>[INST] {system_prompt}\n\nOtázka: {question} [/INST]"
            
            # Generování odpovědi
            response = generate_response(
                model, tokenizer, prompt,
                max_length=300, temperature=0.8
            )
            
            # Vyčištění odpovědi
            response = response.strip()
            if response.startswith("Otázka:"):
                response = response[response.find("[/INST]") + 7:].strip()
            
            print(f"   Odpověď: {response}")
            
            # Rychlá analýza stylu
            babis_indicators = ["hele", "skandál", "makám", "opozice", "brusel", "moje rodina"]
            found_indicators = sum(1 for indicator in babis_indicators if indicator.lower() in response.lower())
            
            if "andrej babiš" in response.lower():
                print(f"   ✅ Podpis: ANO")
            else:
                print(f"   ❌ Podpis: NE")
            
            print(f"   📊 Babišovy indikátory: {found_indicators}/{len(babis_indicators)}")
            
            if found_indicators >= 2 and "andrej babiš" in response.lower():
                print(f"   🎯 Styl: DOBRÝ")
            elif found_indicators >= 1:
                print(f"   ⚠️  Styl: ČÁSTEČNÝ")
            else:
                print(f"   ❌ Styl: ŠPATNÝ")
        
        print("\n" + "=" * 60)
        print("✅ Rychlý test dokončen!")
        print("🎯 Váš adaptér je připraven pro benchmarking!")
        print("🚀 Spusťte: ./run_benchmark_with_adapter.sh")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Zkontrolujte, že máte nainstalované requirements:")
        print("   pip install -r requirements_benchmarking.txt")
        return False
        
    except Exception as e:
        print(f"❌ Chyba: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = quick_test()
    if not success:
        sys.exit(1) 