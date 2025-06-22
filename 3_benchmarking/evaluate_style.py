#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluace Babišova stylu pro TalkLike.LLM
Automatické hodnocení charakteristických prvků stylu
"""

import os
import re
import json
from datetime import datetime
from typing import Dict, List, Tuple

class BabisStyleEvaluator:
    """Evaluátor Babišova stylu"""
    
    def __init__(self):
        # Charakteristické Babišovy fráze
        self.babis_phrases = [
            "hele", "skandál", "makám", "opozice krade", "brusel",
            "to je", "já jsem", "moje rodina", "úřady", "parlament",
            "inflace", "důchody", "daně", "korupce", "efektivizace"
        ]
        
        # Slovenské odchylky
        self.slovak_words = [
            "sme", "som", "makáme", "centralizácia", "efektivizácia",
            "obrana", "hotová", "brzdí", "kritizujú", "sabotujú"
        ]
        
        # Emotivní výrazy
        self.emotional_words = [
            "šílený", "tragédyje", "kampááň", "hrozné", "neuvěřitelný",
            "skandál", "kritizují", "sabotují", "kradou", "brzdí"
        ]
        
        # Indikátory první osoby
        self.first_person_indicators = [
            "já", "moje", "můj", "jsem", "mám", "budoval", "makám",
            "pracoval", "pomáhal", "držím", "vidím"
        ]
        
        # Přirovnání (charakteristická pro Babiše)
        self.comparisons = [
            "jak když kráva hraje na klavír",
            "jak když dítě řídí tank", 
            "jak když slepice hraje šachy",
            "jak když ryba jezdí na kole",
            "jak když pes maluje obraz",
            "jak když kráva tančí balet"
        ]

    def evaluate_babis_style(self, response_text: str) -> Dict:
        """Hodnotí Babišův styl v textu"""
        
        text_lower = response_text.lower()
        score = 0
        breakdown = {}
        
        # 1. Babišovy fráze (35%)
        phrase_score = self._evaluate_phrases(text_lower)
        breakdown["babis_phrases"] = phrase_score
        score += phrase_score * 0.35
        
        # 2. Slovenské odchylky (25%)
        slovak_score = self._evaluate_slovak_influence(text_lower)
        breakdown["slovak_influence"] = slovak_score
        score += slovak_score * 0.25
        
        # 3. Emotivní tón (25%)
        emotional_score = self._evaluate_emotional_tone(text_lower)
        breakdown["emotional_tone"] = emotional_score
        score += emotional_score * 0.25
        
        # 4. První osoba (15%)
        first_person_score = self._evaluate_first_person(text_lower)
        breakdown["first_person"] = first_person_score
        score += first_person_score * 0.15
        
        # Dodatečné body za přirovnání
        comparison_bonus = self._evaluate_comparisons(text_lower)
        if comparison_bonus > 0:
            breakdown["comparison_bonus"] = comparison_bonus
            score = min(score + comparison_bonus, 10)  # Maximálně 10 bodů
        
        return {
            "total_score": round(score, 2),
            "breakdown": breakdown,
            "grade": self._get_grade(score),
            "text": response_text,
            "length": len(response_text)
        }
    
    def _evaluate_phrases(self, text: str) -> float:
        """Hodnotí použití Babišových frází"""
        found_phrases = sum(1 for phrase in self.babis_phrases if phrase in text)
        max_expected = 3  # Očekáváme 3+ fráze v dobré odpovědi
        return min(found_phrases / max_expected * 10, 10)
    
    def _evaluate_slovak_influence(self, text: str) -> float:
        """Hodnotí slovenské odchylky"""
        found_slovak = sum(1 for word in self.slovak_words if word in text)
        max_expected = 2  # Očekáváme 1-2 slovenské odchylky
        return min(found_slovak / max_expected * 10, 10)
    
    def _evaluate_emotional_tone(self, text: str) -> float:
        """Hodnotí emotivní tón"""
        found_emotional = sum(1 for word in self.emotional_words if word in text)
        max_expected = 2  # Očekáváme 1-2 emotivní výrazy
        return min(found_emotional / max_expected * 10, 10)
    
    def _evaluate_first_person(self, text: str) -> float:
        """Hodnotí použití první osoby"""
        found_first_person = sum(1 for indicator in self.first_person_indicators if indicator in text)
        max_expected = 3  # Očekáváme 3+ indikátory první osoby
        return min(found_first_person / max_expected * 10, 10)
    
    def _evaluate_comparisons(self, text: str) -> float:
        """Hodnotí charakteristická přirovnání"""
        found_comparisons = sum(1 for comp in self.comparisons if comp in text)
        return found_comparisons * 1.0  # 1 bod za každé přirovnání
    
    def _get_grade(self, score: float) -> str:
        """Převede skóre na známku"""
        if score >= 9.0: return "A"
        elif score >= 8.0: return "B"
        elif score >= 7.0: return "C"
        elif score >= 6.0: return "D"
        else: return "F"

def evaluate_babis_style(response_text: str) -> Dict:
    """Wrapper funkce pro evaluaci stylu"""
    evaluator = BabisStyleEvaluator()
    return evaluator.evaluate_babis_style(response_text)

def evaluate_all_responses():
    """Evaluuje všechny odpovědi v results/"""
    evaluator = BabisStyleEvaluator()
    
    print("🔍 Načítám odpovědi pro evaluaci...")
    
    # Evaluace před fine-tuningem
    before_results = []
    if os.path.exists("results/before_finetune/responses.json"):
        print("📊 Evaluuji odpovědi před fine-tuningem...")
        with open("results/before_finetune/responses.json", "r", encoding="utf-8") as f:
            before_data = json.load(f)
        
        for item in before_data:
            evaluation = evaluator.evaluate_babis_style(item["response"])
            before_results.append({
                "question": item["question"],
                "response": item["response"],
                "evaluation": evaluation
            })
        print(f"✅ Evaluováno {len(before_results)} odpovědí před fine-tuningem")
    
    # Evaluace po fine-tuningem
    after_results = []
    if os.path.exists("results/after_finetune/responses.json"):
        print("📊 Evaluuji odpovědi po fine-tuningem...")
        with open("results/after_finetune/responses.json", "r", encoding="utf-8") as f:
            after_data = json.load(f)
        
        for item in after_data:
            evaluation = evaluator.evaluate_babis_style(item["response"])
            after_results.append({
                "question": item["question"],
                "response": item["response"],
                "evaluation": evaluation
            })
        print(f"✅ Evaluováno {len(after_results)} odpovědí po fine-tuningem")
    
    # Výpočet průměrných skóre
    before_avg = sum(r["evaluation"]["total_score"] for r in before_results) / len(before_results) if before_results else 0
    after_avg = sum(r["evaluation"]["total_score"] for r in after_results) / len(after_results) if after_results else 0
    
    # Uložení výsledků
    results = {
        "before_finetune": {
            "responses": before_results,
            "average_score": round(before_avg, 2),
            "count": len(before_results)
        },
        "after_finetune": {
            "responses": after_results,
            "average_score": round(after_avg, 2),
            "count": len(after_results)
        },
        "improvement": round(after_avg - before_avg, 2),
        "timestamp": datetime.now().isoformat()
    }
    
    with open("results/comparison/style_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📈 Výsledky evaluace:")
    print(f"   Před fine-tuningem: {before_avg:.2f}/10")
    print(f"   Po fine-tuningem: {after_avg:.2f}/10")
    print(f"   Zlepšení: {results['improvement']:.2f} bodů")
    
    return results

if __name__ == "__main__":
    # Test evaluace
    test_text = "Hele, inflace je jak když kráva hraje na klavír! Já makám, ale opozice krade. To je skandál!"
    result = evaluate_babis_style(test_text)
    print(f"Test evaluace: {result}") 