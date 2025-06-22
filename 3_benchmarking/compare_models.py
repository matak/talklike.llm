#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Srovnání modelů pro TalkLike.LLM
Porovnává výkon před a po fine-tuningu
"""

import json
import os
from datetime import datetime
from typing import Dict, List

def compare_models():
    """Porovnává modely před a po fine-tuningu"""
    
    print("📊 Porovnávám modely před a po fine-tuningu...")
    
    comparison_results = {
        "before_finetune": {},
        "after_finetune": {},
        "improvement": {},
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_questions": 0
        }
    }
    
    # Načtení odpovědí před fine-tuningem
    with open("results/before_finetune/responses.json", "r", encoding="utf-8") as f:
        before_data = json.load(f)
    
    comparison_results["before_finetune"] = {
        "responses": before_data,
        "count": len(before_data)
    }
    print(f"✅ Načteno {len(before_data)} odpovědí před fine-tuningem")
    
    # Načtení odpovědí po fine-tuningem
    with open("results/after_finetune/responses.json", "r", encoding="utf-8") as f:
        after_data = json.load(f)
    
    comparison_results["after_finetune"] = {
        "responses": after_data,
        "count": len(after_data)
    }
    print(f"✅ Načteno {len(after_data)} odpovědí po fine-tuningem")
    
    # Výpočet metrik
    metrics = calculate_comparison_metrics(before_data, after_data)
    comparison_results["improvement"] = metrics
    
    print(f"\n📈 Metriky srovnání:")
    print(f"   Průměrná délka odpovědi:")
    print(f"     Před: {metrics['avg_length_before']:.1f} znaků")
    print(f"     Po: {metrics['avg_length_after']:.1f} znaků")
    print(f"     Změna: {metrics['length_change']:+.1f} znaků")
    
    print(f"   Použití Babišových frází:")
    print(f"     Před: {metrics['babis_phrases_before']:.1f} frází/odpověď")
    print(f"     Po: {metrics['babis_phrases_after']:.1f} frází/odpověď")
    print(f"     Zlepšení: {metrics['babis_phrases_improvement']:+.1f} frází")
    
    print(f"   Slovenské odchylky:")
    print(f"     Před: {metrics['slovak_words_before']:.1f} slov/odpověď")
    print(f"     Po: {metrics['slovak_words_after']:.1f} slov/odpověď")
    print(f"     Zlepšení: {metrics['slovak_words_improvement']:+.1f} slov")
    
    # Uložení výsledků srovnání
    with open("results/comparison/model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Srovnání uloženo: results/comparison/model_comparison.json")
    
    return comparison_results

def calculate_comparison_metrics(before_data: List, after_data: List) -> Dict:
    """Vypočítá metriky pro srovnání modelů"""
    
    # Babišovy fráze pro analýzu
    babis_phrases = [
        "hele", "skandál", "makám", "opozice krade", "brusel",
        "to je", "já jsem", "moje rodina", "úřady", "parlament"
    ]
    
    # Slovenské odchylky
    slovak_words = [
        "sme", "som", "makáme", "centralizácia", "efektivizácia"
    ]
    
    def analyze_responses(responses: List) -> Dict:
        """Analyzuje sadu odpovědí"""
        total_length = 0
        total_babis_phrases = 0
        total_slovak_words = 0
        
        for resp in responses:
            text_lower = resp["response"].lower()
            total_length += len(resp["response"])
            
            # Počítání Babišových frází
            for phrase in babis_phrases:
                if phrase in text_lower:
                    total_babis_phrases += 1
            
            # Počítání slovenských slov
            for word in slovak_words:
                if word in text_lower:
                    total_slovak_words += 1
        
        return {
            "avg_length": total_length / len(responses) if responses else 0,
            "avg_babis_phrases": total_babis_phrases / len(responses) if responses else 0,
            "avg_slovak_words": total_slovak_words / len(responses) if responses else 0
        }
    
    # Analýza před fine-tuningem
    before_metrics = analyze_responses(before_data)
    
    # Analýza po fine-tuningem
    after_metrics = analyze_responses(after_data)
    
    # Výpočet zlepšení
    improvement = {
        "avg_length_before": before_metrics["avg_length"],
        "avg_length_after": after_metrics["avg_length"],
        "length_change": after_metrics["avg_length"] - before_metrics["avg_length"],
        
        "babis_phrases_before": before_metrics["avg_babis_phrases"],
        "babis_phrases_after": after_metrics["avg_babis_phrases"],
        "babis_phrases_improvement": after_metrics["avg_babis_phrases"] - before_metrics["avg_babis_phrases"],
        
        "slovak_words_before": before_metrics["avg_slovak_words"],
        "slovak_words_after": after_metrics["avg_slovak_words"],
        "slovak_words_improvement": after_metrics["avg_slovak_words"] - before_metrics["avg_slovak_words"],
        
        "overall_improvement_score": (
            (after_metrics["avg_babis_phrases"] - before_metrics["avg_babis_phrases"]) * 2 +
            (after_metrics["avg_slovak_words"] - before_metrics["avg_slovak_words"]) * 1.5
        )
    }
    
    return improvement

def create_comparison_table():
    """Vytvoří tabulku pro srovnání"""
    
    with open("results/comparison/model_comparison.json", "r", encoding="utf-8") as f:
        comparison_data = json.load(f)
    
    metrics = comparison_data["improvement"]
    
    # Vytvoření tabulky
    table_data = {
        "Metrika": [
            "Průměrná délka odpovědi (znaky)",
            "Babišovy fráze (počet/odpověď)",
            "Slovenské odchylky (počet/odpověď)",
            "Celkové skóre zlepšení"
        ],
        "Před fine-tuningem": [
            f"{metrics['avg_length_before']:.1f}",
            f"{metrics['babis_phrases_before']:.1f}",
            f"{metrics['slovak_words_before']:.1f}",
            "0.0"
        ],
        "Po fine-tuningem": [
            f"{metrics['avg_length_after']:.1f}",
            f"{metrics['babis_phrases_after']:.1f}",
            f"{metrics['slovak_words_after']:.1f}",
            f"{metrics['overall_improvement_score']:.1f}"
        ],
        "Zlepšení": [
            f"{metrics['length_change']:+.1f}",
            f"{metrics['babis_phrases_improvement']:+.1f}",
            f"{metrics['slovak_words_improvement']:+.1f}",
            f"{metrics['overall_improvement_score']:+.1f}"
        ]
    }
    
    # Vytvoření tabulky bez pandas
    headers = ["Metrika", "Před fine-tuningem", "Po fine-tuningem", "Zlepšení"]
    rows = []
    for i in range(len(table_data["Metrika"])):
        row = [
            table_data["Metrika"][i],
            table_data["Před fine-tuningem"][i],
            table_data["Po fine-tuningem"][i],
            table_data["Zlepšení"][i]
        ]
        rows.append(row)
    
    # Výpočet šířky sloupců
    col_widths = []
    for header in headers:
        col_widths.append(len(header))
    
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    
    # Vytvoření tabulky
    table_lines = []
    
    # Hlavička
    header_line = "|"
    separator_line = "|"
    for i, header in enumerate(headers):
        header_line += f" {header:<{col_widths[i]}} |"
        separator_line += f" {'-' * col_widths[i]} |"
    table_lines.append(header_line)
    table_lines.append(separator_line)
    
    # Řádky dat
    for row in rows:
        data_line = "|"
        for i, cell in enumerate(row):
            data_line += f" {cell:<{col_widths[i]}} |"
        table_lines.append(data_line)
    
    table_text = "\n".join(table_lines)
    
    print(f"📊 Tabulka srovnání:")
    print(table_text)
    
    return table_data

if __name__ == "__main__":
    # Test srovnání modelů
    print("🧪 Test srovnání modelů...")
    
    # Nejdříve vygenerovat testovací data
    from generate_responses import generate_responses
    
    generate_responses("base", "results/before_finetune/")
    generate_responses("finetuned", "results/after_finetune/")
    
    # Spustit srovnání
    results = compare_models()
    
    # Vytvořit tabulku
    table_data = create_comparison_table()
    
    print("\n📋 Tabulka srovnání vytvořena") 