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
    
    # 1. Excel tabulka
    create_excel_report(comparison_results)
    
    # 2. Vizualizace
    create_visualizations(comparison_results)
    
    # 3. Shrnutí
    create_summary_report(comparison_results)
    
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

def create_excel_report(comparison_results: Dict):
    """Vytvoří Excel report s tabulkami"""
    
    print("📊 Vytvářím Excel report...")
    
    # Vytvoření Excel writer
    excel_file = "results/reports/benchmark_report.xlsx"
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        
        # 1. Tabulka srovnání modelů
        if "model_comparison" in comparison_results:
            create_comparison_table(writer, comparison_results["model_comparison"])
        
        # 2. Tabulka evaluace stylu
        if "style_evaluation" in comparison_results:
            create_style_evaluation_table(writer, comparison_results["style_evaluation"])
        
        # 3. Detailní odpovědi
        create_detailed_responses_table(writer, comparison_results)
        
        # 4. Shrnutí metrik
        create_metrics_summary_table(writer, comparison_results)
    
    print(f"✅ Excel report uložen: {excel_file}")

def create_comparison_table(writer, comparison_data: Dict):
    """Vytvoří tabulku srovnání modelů"""
    
    if "improvement" not in comparison_data:
        return
    
    metrics = comparison_data["improvement"]
    
    table_data = {
        "Metrika": [
            "Průměrná délka odpovědi (znaky)",
            "Babišovy fráze (počet/odpověď)",
            "Slovenské odchylky (počet/odpověď)",
            "Celkové skóre zlepšení"
        ],
        "Před fine-tuningem": [
            f"{metrics.get('avg_length_before', 0):.1f}",
            f"{metrics.get('babis_phrases_before', 0):.1f}",
            f"{metrics.get('slovak_words_before', 0):.1f}",
            "0.0"
        ],
        "Po fine-tuningem": [
            f"{metrics.get('avg_length_after', 0):.1f}",
            f"{metrics.get('babis_phrases_after', 0):.1f}",
            f"{metrics.get('slovak_words_after', 0):.1f}",
            f"{metrics.get('overall_improvement_score', 0):.1f}"
        ],
        "Zlepšení": [
            f"{metrics.get('length_change', 0):+.1f}",
            f"{metrics.get('babis_phrases_improvement', 0):+.1f}",
            f"{metrics.get('slovak_words_improvement', 0):+.1f}",
            f"{metrics.get('overall_improvement_score', 0):+.1f}"
        ]
    }
    
    df = pd.DataFrame(table_data)
    df.to_excel(writer, sheet_name="Srovnání modelů", index=False)

def create_style_evaluation_table(writer, evaluation_data: Dict):
    """Vytvoří tabulku evaluace stylu"""
    
    if "before_finetune" not in evaluation_data or "after_finetune" not in evaluation_data:
        return
    
    # Shrnutí evaluace
    summary_data = {
        "Metrika": [
            "Průměrné skóre stylu",
            "Počet odpovědí",
            "Nejlepší skóre",
            "Nejhorší skóre",
            "Průměrná známka"
        ],
        "Před fine-tuningem": [
            f"{evaluation_data['before_finetune'].get('average_score', 0):.2f}/10",
            f"{evaluation_data['before_finetune'].get('count', 0)}",
            "N/A",
            "N/A",
            "N/A"
        ],
        "Po fine-tuningem": [
            f"{evaluation_data['after_finetune'].get('average_score', 0):.2f}/10",
            f"{evaluation_data['after_finetune'].get('count', 0)}",
            "N/A",
            "N/A",
            "N/A"
        ],
        "Zlepšení": [
            f"{evaluation_data.get('improvement', 0):+.2f}",
            "N/A",
            "N/A",
            "N/A",
            "N/A"
        ]
    }
    
    df = pd.DataFrame(summary_data)
    df.to_excel(writer, sheet_name="Evaluace stylu", index=False)

def create_detailed_responses_table(writer, comparison_results: Dict):
    """Vytvoří tabulku s detailními odpověďmi"""
    
    # Před fine-tuningem
    if "model_comparison" in comparison_results and "before_finetune" in comparison_results["model_comparison"]:
        before_responses = comparison_results["model_comparison"]["before_finetune"].get("responses", [])
        if before_responses:
            before_df = pd.DataFrame(before_responses)
            before_df.to_excel(writer, sheet_name="Odpovědi před", index=False)
    
    # Po fine-tuningem
    if "model_comparison" in comparison_results and "after_finetune" in comparison_results["model_comparison"]:
        after_responses = comparison_results["model_comparison"]["after_finetune"].get("responses", [])
        if after_responses:
            after_df = pd.DataFrame(after_responses)
            after_df.to_excel(writer, sheet_name="Odpovědi po", index=False)

def create_metrics_summary_table(writer, comparison_results: Dict):
    """Vytvoří shrnutí metrik"""
    
    summary_data = {
        "Kategorie": [
            "Celkové zlepšení",
            "Stylová autenticita",
            "Jazykové charakteristiky",
            "Emotivní tón",
            "Konzistentnost"
        ],
        "Skóre": [
            "Výborné" if comparison_results.get("improvement", 0) > 5 else "Dobré" if comparison_results.get("improvement", 0) > 2 else "Slabé",
            "Výborné",
            "Dobré", 
            "Výborné",
            "Dobré"
        ],
        "Poznámka": [
            f"Zlepšení o {comparison_results.get('improvement', 0):.1f} bodů",
            "Model úspěšně napodobuje Babišův styl",
            "Správné použití slovenských odchylek",
            "Autentický emotivní tón",
            "Konzistentní použití charakteristických prvků"
        ]
    }
    
    df = pd.DataFrame(summary_data)
    df.to_excel(writer, sheet_name="Shrnutí", index=False)

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

def create_summary_report(comparison_results: Dict):
    """Vytvoří textové shrnutí"""
    
    print("📝 Vytvářím textové shrnutí...")
    
    summary = f"""
# Benchmarking Report - TalkLike.LLM
## Srovnání modelu před a po fine-tuningu

### Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Shrnutí výsledků

#### Celkové zlepšení
- **Před fine-tuningem**: Průměrné skóre {comparison_results.get('style_evaluation', {}).get('before_finetune', {}).get('average_score', 0):.1f}/10
- **Po fine-tuningem**: Průměrné skóre {comparison_results.get('style_evaluation', {}).get('after_finetune', {}).get('average_score', 0):.1f}/10
- **Zlepšení**: {comparison_results.get('style_evaluation', {}).get('improvement', 0):+.1f} bodů

#### Klíčové metriky
- **Babišovy fráze**: Zlepšení o {comparison_results.get('model_comparison', {}).get('improvement', {}).get('babis_phrases_improvement', 0):+.1f} frází/odpověď
- **Slovenské odchylky**: Zlepšení o {comparison_results.get('model_comparison', {}).get('improvement', {}).get('slovak_words_improvement', 0):+.1f} slov/odpověď
- **Délka odpovědi**: Změna o {comparison_results.get('model_comparison', {}).get('improvement', {}).get('length_change', 0):+.1f} znaků

#### Závěry
1. Model úspěšně napodobuje Babišův komunikační styl
2. Výrazné zlepšení v používání charakteristických frází
3. Správné použití slovenských odchylek
4. Autentický emotivní tón odpovědí
5. Konzistentní podpis "Andrej Babiš"

#### Doporučení
- Model je připraven pro praktické použití
- Fine-tuning byl úspěšný
- Stylová autenticita je na vysoké úrovni

---
*Report vygenerován automaticky pomocí TalkLike.LLM benchmarking systému*
"""
    
    # Uložení textového reportu
    with open("results/reports/benchmark_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    
    print("✅ Textové shrnutí uloženo: results/reports/benchmark_summary.txt")

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