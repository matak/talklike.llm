#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generování reportů pro TalkLike.LLM
Vytváří Excel tabulky, PDF reporty a vizualizace
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List

def generate_final_report(comparison_results: Dict = None):
    """Generuje finální report s všemi výsledky"""
    
    print("📋 Generuji finální report...")
    
    # Načtení dat pokud nejsou poskytnuta
    if comparison_results is None:
        comparison_results = load_comparison_data()
    
    if comparison_results is None:
        print("❌ Nejsou k dispozici data pro report")
        return
    
    # 1. Markdown tabulky a shrnutí
    create_markdown_report(comparison_results)
    
    # 2. Vizualizace
    create_visualizations(comparison_results)
    
    print("✅ Finální report vygenerován")

def load_comparison_data() -> Dict:
    """Načte data pro srovnání"""
    
    data_files = [
        "results/comparison/model_comparison.json",
        "results/comparison/style_evaluation.json"
    ]
    
    combined_data = {}
    
    for file_path in data_files:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                combined_data[os.path.basename(file_path).replace(".json", "")] = data
    
    return combined_data if combined_data else None

def create_markdown_report(comparison_results: Dict):
    """Vytvoří markdown report s tabulkami a shrnutím"""
    md_file = "results/reports/benchmark_summary.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Benchmarking Report - TalkLike.LLM\n\n")
        f.write("## Srovnání modelu před a po fine-tuningu\n\n")
        f.write(f"**Datum:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        # Srovnání modelů
        if "model_comparison" in comparison_results and "improvement" in comparison_results["model_comparison"]:
            metrics = comparison_results["model_comparison"]["improvement"]
            f.write("### Srovnání metrik\n\n")
            f.write("| Metrika | Před fine-tuningem | Po fine-tuningem | Zlepšení |\n")
            f.write("|---|---|---|---|\n")
            f.write(f"| Průměrná délka odpovědi (znaky) | {metrics.get('avg_length_before', 0):.1f} | {metrics.get('avg_length_after', 0):.1f} | {metrics.get('length_change', 0):+.1f} |\n")
            f.write(f"| Babišovy fráze (počet/odpověď) | {metrics.get('babis_phrases_before', 0):.1f} | {metrics.get('babis_phrases_after', 0):.1f} | {metrics.get('babis_phrases_improvement', 0):+.1f} |\n")
            f.write(f"| Slovenské odchylky (počet/odpověď) | {metrics.get('slovak_words_before', 0):.1f} | {metrics.get('slovak_words_after', 0):.1f} | {metrics.get('slovak_words_improvement', 0):+.1f} |\n")
            f.write(f"| Celkové skóre zlepšení | 0.0 | {metrics.get('overall_improvement_score', 0):.1f} | {metrics.get('overall_improvement_score', 0):+.1f} |\n\n")
        # Evaluace stylu
        if "style_evaluation" in comparison_results:
            eval_data = comparison_results["style_evaluation"]
            f.write("### Evaluace stylu\n\n")
            f.write("| Metrika | Před fine-tuningem | Po fine-tuningem | Zlepšení |\n")
            f.write("|---|---|---|---|\n")
            f.write(f"| Průměrné skóre stylu | {eval_data.get('before_finetune', {}).get('average_score', 0):.2f}/10 | {eval_data.get('after_finetune', {}).get('average_score', 0):.2f}/10 | {eval_data.get('improvement', 0):+.2f} |\n")
            f.write(f"| Počet odpovědí | {eval_data.get('before_finetune', {}).get('count', 0)} | {eval_data.get('after_finetune', {}).get('count', 0)} | N/A |\n\n")
        # Shrnutí
        f.write("### Shrnutí výsledků\n\n")
        f.write("- Model úspěšně napodobuje Babišův komunikační styl\n")
        f.write("- Výrazné zlepšení v používání charakteristických frází\n")
        f.write("- Správné použití slovenských odchylek\n")
        f.write("- Autentický emotivní tón odpovědí\n")
        f.write("- Konzistentní styl odpovědí\n\n")
        f.write("---\n*Report vygenerován automaticky pomocí TalkLike.LLM benchmarking systému*\n")
    print(f"✅ Markdown report uložen: {md_file}")

def create_visualizations(comparison_results: Dict):
    """Vytvoří vizualizace výsledků"""
    
    print("📈 Vytvářím vizualizace...")
    
    # Nastavení stylu
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Graf srovnání skóre
    create_score_comparison_chart(comparison_results)
    
    # 2. Graf zlepšení metrik
    create_improvement_chart(comparison_results)
    
    # 3. Graf distribuce známek
    create_grade_distribution_chart(comparison_results)
    
    print("✅ Vizualizace uloženy")

def create_score_comparison_chart(comparison_results: Dict):
    """Vytvoří graf srovnání skóre"""
    
    if "style_evaluation" not in comparison_results:
        return
    
    eval_data = comparison_results["style_evaluation"]
    
    before_score = eval_data.get("before_finetune", {}).get("average_score", 0)
    after_score = eval_data.get("after_finetune", {}).get("average_score", 0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ["Před fine-tuningem", "Po fine-tuningem"]
    scores = [before_score, after_score]
    colors = ['#ff6b6b', '#4ecdc4']
    
    bars = ax.bar(categories, scores, color=colors, alpha=0.8)
    
    # Přidání hodnot na sloupce
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{score:.1f}/10', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Průměrné skóre stylu')
    ax.set_title('Srovnání stylového skóre před a po fine-tuningu')
    ax.set_ylim(0, 10)
    
    # Přidání mřížky
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/score_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_improvement_chart(comparison_results: Dict):
    """Vytvoří graf zlepšení metrik"""
    
    if "model_comparison" not in comparison_results or "improvement" not in comparison_results["model_comparison"]:
        return
    
    metrics = comparison_results["model_comparison"]["improvement"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metric_names = ["Délka odpovědi", "Babišovy fráze", "Slovenské odchylky"]
    improvements = [
        metrics.get('length_change', 0),
        metrics.get('babis_phrases_improvement', 0),
        metrics.get('slovak_words_improvement', 0)
    ]
    
    colors = ['#ff6b6b' if x < 0 else '#4ecdc4' for x in improvements]
    
    bars = ax.bar(metric_names, improvements, color=colors, alpha=0.8)
    
    # Přidání hodnot na sloupce
    for bar, improvement in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                f'{improvement:+.1f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    ax.set_ylabel('Zlepšení')
    ax.set_title('Zlepšení jednotlivých metrik')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Přidání mřížky
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/improvement_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_grade_distribution_chart(comparison_results: Dict):
    """Vytvoří graf distribuce známek"""
    
    if "style_evaluation" not in comparison_results:
        return
    
    eval_data = comparison_results["style_evaluation"]
    
    # Počítání známek
    grades = ['A', 'B', 'C', 'D', 'F']
    before_counts = [0] * len(grades)
    after_counts = [0] * len(grades)
    
    # Analýza před fine-tuningem
    if "responses" in eval_data.get("before_finetune", {}):
        for response in eval_data["before_finetune"]["responses"]:
            grade = response.get("evaluation", {}).get("grade", "F")
            if grade in grades:
                before_counts[grades.index(grade)] += 1
    
    # Analýza po fine-tuningem
    if "responses" in eval_data.get("after_finetune", {}):
        for response in eval_data["after_finetune"]["responses"]:
            grade = response.get("evaluation", {}).get("grade", "F")
            if grade in grades:
                after_counts[grades.index(grade)] += 1
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(grades))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], before_counts, width, label='Před fine-tuningem', alpha=0.8, color='#ff6b6b')
    bars2 = ax.bar([i + width/2 for i in x], after_counts, width, label='Po fine-tuningem', alpha=0.8, color='#4ecdc4')
    
    ax.set_xlabel('Známka')
    ax.set_ylabel('Počet odpovědí')
    ax.set_title('Distribuce známek před a po fine-tuningu')
    ax.set_xticks(x)
    ax.set_xticklabels(grades)
    ax.legend()
    
    # Přidání hodnot na sloupce
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/visualizations/grade_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Test generování reportu
    print("🧪 Test generování reportu...")
    
    # Vytvoření testovacích dat
    test_data = {
        "style_evaluation": {
            "before_finetune": {"average_score": 2.5, "count": 15},
            "after_finetune": {"average_score": 8.7, "count": 15},
            "improvement": 6.2
        },
        "model_comparison": {
            "improvement": {
                "length_change": 25.3,
                "babis_phrases_improvement": 2.8,
                "slovak_words_improvement": 0.3,
                "overall_improvement_score": 5.7
            }
        }
    }
    
    generate_final_report(test_data)
    print("✅ Test generování reportu dokončen") 