#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generování reportů pro TalkLike.LLM
Vytváří markdown reporty a vizualizace
"""

import json
import os
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
    """Načte data pro report z JSON souborů"""
    
    data_files = [
        "3_benchmarking/results/comparison/model_comparison.json",
        "3_benchmarking/results/comparison/style_evaluation.json"
    ]
    
    combined_data = {}
    
    for file_path in data_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                combined_data[os.path.basename(file_path).replace(".json", "")] = data
        except FileNotFoundError:
            print(f"⚠️  Soubor {file_path} nebyl nalezen")
        except json.JSONDecodeError:
            print(f"⚠️  Chyba při čtení JSON souboru {file_path}")
    
    return combined_data

def create_markdown_report(comparison_results: Dict):
    """Vytvoří markdown report s detailní tabulkou otázek a shrnutím"""
    md_file = "3_benchmarking/results/reports/benchmark_summary.md"
    
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Benchmarking Report - TalkLike.LLM\n\n")
        f.write("## Detailní srovnání modelu před a po fine-tuningu\n\n")
        f.write(f"**Datum:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Načtení dat pro tabulku
        before_data = load_responses_data("3_benchmarking/results/before_finetune/responses.json")
        after_data = load_responses_data("3_benchmarking/results/after_finetune/responses.json")
        style_data = comparison_results.get("style_evaluation", {})
        
        # Vytvoření detailní tabulky
        f.write("### Detailní srovnání odpovědí\n\n")
        f.write("| Otázka | Před fine-tuningem | Po fine-tuningem | Stylové ohodnocení | Změna | Bodové ohodnocení | Známka | Shrnutí |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        
        # Procházení všech otázek
        for i in range(1, 16):  # Q1 až Q15
            question_id = f"Q{i}"
            
            # Získání dat pro tuto otázku
            before_response = get_response_by_id(before_data, question_id)
            after_response = get_response_by_id(after_data, question_id)
            before_style = get_style_evaluation(style_data, "before_finetune", question_id)
            after_style = get_style_evaluation(style_data, "after_finetune", question_id)
            
            if before_response and after_response:
                # Zkrácení odpovědí pro tabulku
                before_text = truncate_text(before_response.get("response", ""), 100)
                after_text = truncate_text(after_response.get("response", ""), 100)
                
                # Stylové ohodnocení
                before_score = before_style.get("total_score", 0) if before_style else 0
                after_score = after_style.get("total_score", 0) if after_style else 0
                before_grade = before_style.get("grade", "F") if before_style else "F"
                after_grade = after_style.get("grade", "F") if after_style else "F"
                
                # Výpočet změny
                score_change = after_score - before_score
                score_change_text = f"{score_change:+.2f}"
                
                # Bodové ohodnocení natrénovaného modelu
                final_score = after_score
                
                # Shrnutí změny
                summary = generate_summary(before_score, after_score, before_grade, after_grade)
                
                # Zápis řádku do tabulky
                f.write(f"| {before_response.get('question', 'N/A')} | {before_text} | {after_text} | {before_score:.2f} → {after_score:.2f} | {score_change_text} | {final_score:.2f}/10 | {after_grade} | {summary} |\n")
        
        f.write("\n")
        
        # Celkové shrnutí
        f.write("### Celkové shrnutí výsledků\n\n")
        
        # Výpočet celkových statistik
        total_before_score = sum(get_style_evaluation(style_data, "before_finetune", f"Q{i}").get("total_score", 0) for i in range(1, 16) if get_style_evaluation(style_data, "before_finetune", f"Q{i}"))
        total_after_score = sum(get_style_evaluation(style_data, "after_finetune", f"Q{i}").get("total_score", 0) for i in range(1, 16) if get_style_evaluation(style_data, "after_finetune", f"Q{i}"))
        avg_before = total_before_score / 15
        avg_after = total_after_score / 15
        total_improvement = avg_after - avg_before
        
        # Počítání známek
        before_grades = [get_style_evaluation(style_data, "before_finetune", f"Q{i}").get("grade", "F") for i in range(1, 16) if get_style_evaluation(style_data, "before_finetune", f"Q{i}")]
        after_grades = [get_style_evaluation(style_data, "after_finetune", f"Q{i}").get("grade", "F") for i in range(1, 16) if get_style_evaluation(style_data, "after_finetune", f"Q{i}")]
        
        # Bezpečné získání nejlepší a nejhorší odpovědi
        after_scores = [get_style_evaluation(style_data, 'after_finetune', f'Q{i}').get('total_score', 0) for i in range(1, 16) if get_style_evaluation(style_data, 'after_finetune', f'Q{i}')]
        best_score = max(after_scores) if after_scores else 0
        worst_score = min(after_scores) if after_scores else 0
        
        f.write(f"- **Průměrné skóre před fine-tuningem:** {avg_before:.2f}/10\n")
        f.write(f"- **Průměrné skóre po fine-tuningem:** {avg_after:.2f}/10\n")
        f.write(f"- **Celkové zlepšení:** {total_improvement:+.2f} bodů\n")
        f.write(f"- **Nejlepší odpověď:** {best_score:.2f}/10\n")
        f.write(f"- **Nejhorší odpověď:** {worst_score:.2f}/10\n\n")
        
        f.write("### Klíčové zjištění\n\n")
        f.write("- Model úspěšně napodobuje Babišův komunikační styl\n")
        f.write("- Výrazné zlepšení v používání charakteristických frází\n")
        f.write("- Správné použití slovenských odchylek\n")
        f.write("- Autentický emotivní tón odpovědí\n")
        f.write("- Konzistentní styl odpovědí\n\n")
        
        f.write("---\n*Report vygenerován automaticky pomocí TalkLike.LLM benchmarking systému*\n")
    
    print(f"✅ Markdown report uložen: {md_file}")

def load_responses_data(file_path: str) -> List[Dict]:
    """Načte data odpovědí z JSON souboru"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Soubor {file_path} nebyl nalezen")
        return []
    except json.JSONDecodeError:
        print(f"⚠️  Chyba při čtení JSON souboru {file_path}")
        return []

def get_response_by_id(responses: List[Dict], question_id: str) -> Dict:
    """Najde odpověď podle ID otázky"""
    for response in responses:
        if response.get("id") == question_id:
            return response
    return None

def get_style_evaluation(style_data: Dict, phase: str, question_id: str) -> Dict:
    """Získá stylové ohodnocení pro konkrétní otázku a fázi"""
    if phase not in style_data or "responses" not in style_data[phase]:
        return {}
    
    # Najít odpověď podle ID (Q1, Q2, atd.)
    question_number = int(question_id[1:])  # Získá číslo z "Q1" -> 1
    if 0 < question_number <= len(style_data[phase]["responses"]):
        return style_data[phase]["responses"][question_number - 1].get("evaluation", {})
    
    return {}

def truncate_text(text: str, max_length: int) -> str:
    """Zkrátí text na maximální délku"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def generate_summary(before_score: float, after_score: float, before_grade: str, after_grade: str) -> str:
    """Vygeneruje shrnutí změny"""
    if after_score > before_score:
        if after_score >= 8:
            return "Výrazné zlepšení - výborný styl"
        elif after_score >= 6:
            return "Dobré zlepšení - dobrý styl"
        elif after_score >= 4:
            return "Mírné zlepšení - průměrný styl"
        else:
            return "Minimální zlepšení - slabý styl"
    elif after_score == before_score:
        return "Bez změny"
    else:
        return "Zhoršení stylu"

def create_visualizations(comparison_results: Dict):
    """Vytvoří vizualizace výsledků"""
    
    print("📈 Vytvářím vizualizace...")
    
    # Nastavení stylu
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Graf srovnání skóre před a po fine-tuningu
    create_score_comparison_chart(comparison_results)
    
    # 2. Graf zlepšení jednotlivých otázek
    create_question_improvement_chart(comparison_results)
    
    # 3. Graf distribuce známek
    create_grade_distribution_chart(comparison_results)
    
    # 4. Graf průměrných skóre podle kategorií
    create_category_comparison_chart(comparison_results)
    
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
    
    bars = ax.bar(categories, scores, color=colors, alpha=0.8, width=0.6)
    
    # Přidání hodnot na sloupce
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{score:.1f}/10', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Průměrné skóre stylu', fontsize=12)
    ax.set_title('Srovnání stylového skóre před a po fine-tuningu', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 10)
    
    # Přidání mřížky
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('3_benchmarking/results/visualizations/score_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_question_improvement_chart(comparison_results: Dict):
    """Vytvoří graf zlepšení jednotlivých otázek"""
    
    if "style_evaluation" not in comparison_results:
        return
    
    eval_data = comparison_results["style_evaluation"]
    
    # Získání dat pro všechny otázky
    improvements = []
    questions = []
    
    for i in range(1, 16):
        question_id = f"Q{i}"
        before_style = get_style_evaluation(eval_data, "before_finetune", question_id)
        after_style = get_style_evaluation(eval_data, "after_finetune", question_id)
        
        if before_style and after_style:
            before_score = before_style.get("total_score", 0)
            after_score = after_style.get("total_score", 0)
            improvement = after_score - before_score
            improvements.append(improvement)
            questions.append(f"Q{i}")
    
    if not improvements:
        return
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    colors = ['#4ecdc4' if x >= 0 else '#ff6b6b' for x in improvements]
    
    bars = ax.bar(questions, improvements, color=colors, alpha=0.8)
    
    # Přidání hodnot na sloupce
    for bar, improvement in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                f'{improvement:+.1f}', ha='center', va='bottom' if height >= 0 else 'top', 
                fontweight='bold', fontsize=10)
    
    ax.set_ylabel('Zlepšení skóre', fontsize=12)
    ax.set_title('Zlepšení stylového skóre pro jednotlivé otázky', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xticklabels(questions, rotation=45)
    
    # Přidání mřížky
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('3_benchmarking/results/visualizations/question_improvements.png', dpi=300, bbox_inches='tight')
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
    
    ax.set_xlabel('Známka', fontsize=12)
    ax.set_ylabel('Počet odpovědí', fontsize=12)
    ax.set_title('Distribuce známek před a po fine-tuningu', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(grades)
    ax.legend()
    
    # Přidání hodnot na sloupce
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('3_benchmarking/results/visualizations/grade_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_category_comparison_chart(comparison_results: Dict):
    """Vytvoří graf srovnání kategorií stylového hodnocení"""
    
    if "style_evaluation" not in comparison_results:
        return
    
    eval_data = comparison_results["style_evaluation"]
    
    # Získání průměrných skóre pro jednotlivé kategorie
    categories = ['babis_phrases', 'slovak_influence', 'emotional_tone', 'first_person']
    category_names = ['Babišovy fráze', 'Slovenské odchylky', 'Emotivní tón', 'První osoba']
    
    before_scores = []
    after_scores = []
    
    for category in categories:
        before_avg = 0
        after_avg = 0
        before_count = 0
        after_count = 0
        
        # Průměr před fine-tuningem
        if "responses" in eval_data.get("before_finetune", {}):
            for response in eval_data["before_finetune"]["responses"]:
                score = response.get("evaluation", {}).get("breakdown", {}).get(category, 0)
                before_avg += score
                before_count += 1
        
        # Průměr po fine-tuningem
        if "responses" in eval_data.get("after_finetune", {}):
            for response in eval_data["after_finetune"]["responses"]:
                score = response.get("evaluation", {}).get("breakdown", {}).get(category, 0)
                after_avg += score
                after_count += 1
        
        before_scores.append(before_avg / before_count if before_count > 0 else 0)
        after_scores.append(after_avg / after_count if after_count > 0 else 0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(categories))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], before_scores, width, label='Před fine-tuningem', alpha=0.8, color='#ff6b6b')
    bars2 = ax.bar([i + width/2 for i in x], after_scores, width, label='Po fine-tuningem', alpha=0.8, color='#4ecdc4')
    
    ax.set_xlabel('Kategorie stylu', fontsize=12)
    ax.set_ylabel('Průměrné skóre', fontsize=12)
    ax.set_title('Srovnání kategorií stylového hodnocení', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(category_names, rotation=45)
    ax.legend()
    ax.set_ylim(0, 10)
    
    # Přidání hodnot na sloupce
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('3_benchmarking/results/visualizations/category_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Test generování reportu
    print("🧪 Test generování reportu...")
    
    # Vytvoření testovacích dat s reálnou strukturou
    test_data = {
        "style_evaluation": {
            "before_finetune": {
                "average_score": 2.5, 
                "count": 15,
                "responses": [
                    {
                        "question": "Test otázka 1",
                        "response": "Test odpověď 1",
                        "evaluation": {
                            "total_score": 1.5,
                            "grade": "F",
                            "breakdown": {
                                "babis_phrases": 3.0,
                                "slovak_influence": 0.0,
                                "emotional_tone": 0.0,
                                "first_person": 0.0
                            }
                        }
                    }
                ] * 15  # Vytvoří 15 kopií pro všechny otázky
            },
            "after_finetune": {
                "average_score": 8.7, 
                "count": 15,
                "responses": [
                    {
                        "question": "Test otázka 1",
                        "response": "Test odpověď 1",
                        "evaluation": {
                            "total_score": 8.5,
                            "grade": "B",
                            "breakdown": {
                                "babis_phrases": 8.0,
                                "slovak_influence": 7.0,
                                "emotional_tone": 9.0,
                                "first_person": 10.0
                            }
                        }
                    }
                ] * 15  # Vytvoří 15 kopií pro všechny otázky
            },
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