#!/usr/bin/env python3
"""
Script pro kontrolu kvality dat a generování reportu o datasetu.
Analyzuje strukturu, obsah a kvalitu vygenerovaných dat.
"""

import os
import json
import re
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class DataQualityChecker:
    def __init__(self, dataset_file: str = "data/all.jsonl"):
        """
        Inicializace kontroloru kvality dat.
        
        Args:
            dataset_file: Cesta k dataset souboru
        """
        self.dataset_file = dataset_file
        self.data = None
        self.messages = []
        self.qa_pairs = []
        
    def load_dataset(self) -> bool:
        """Načte dataset ze souboru."""
        try:
            with open(self.dataset_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            self.messages = self.data.get("messages", [])
            
            # Extrakce QA párů
            self.qa_pairs = []
            for i in range(1, len(self.messages), 2):
                if i + 1 < len(self.messages):
                    user_msg = self.messages[i]
                    assistant_msg = self.messages[i + 1]
                    
                    if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
                        self.qa_pairs.append({
                            "question": user_msg["content"],
                            "answer": assistant_msg["content"]
                        })
            
            print(f"Dataset načten: {len(self.messages)} zpráv, {len(self.qa_pairs)} QA párů")
            return True
            
        except Exception as e:
            print(f"Chyba při načítání datasetu: {e}")
            return False
    
    def check_basic_structure(self) -> Dict[str, Any]:
        """Kontroluje základní strukturu datasetu."""
        print("\n=== Kontrola základní struktury ===")
        
        if not self.messages:
            return {"error": "Žádné zprávy k analýze"}
        
        # Analýza rolí
        roles = [msg["role"] for msg in self.messages]
        role_counts = Counter(roles)
        
        # Kontrola střídání rolí
        alternating_issues = 0
        for i in range(1, len(self.messages)):
            if self.messages[i]["role"] == self.messages[i-1]["role"]:
                alternating_issues += 1
        
        result = {
            "total_messages": len(self.messages),
            "role_distribution": dict(role_counts),
            "qa_pairs_count": len(self.qa_pairs),
            "alternating_issues": alternating_issues,
            "structure_valid": alternating_issues == 0
        }
        
        print(f"✅ Struktura kontrolována")
        print(f"  - Celkem zpráv: {result['total_messages']}")
        print(f"  - QA párů: {result['qa_pairs_count']}")
        print(f"  - Role: {result['role_distribution']}")
        print(f"  - Problémy se střídáním: {result['alternating_issues']}")
        
        return result
    
    def analyze_babis_style(self) -> Dict[str, Any]:
        """Analyzuje charakteristické prvky Babišova stylu."""
        print("\n=== Analýza Babišova stylu ===")
        
        # Charakteristické výrazy
        babis_phrases = {
            "hele": r"\bhele\b",
            "skandál": r"\bskandál\b",
            "makám": r"\bmakám\b",
            "krade": r"\bkrade\b",
            "brusel": r"\bbrusel\b",
            "inflace": r"\binflace\b",
            "opozice": r"\bopozice\b",
            "úřady": r"\búřady\b",
            "rodina": r"\brodina\b",
            "továrna": r"\btovárna\b",
            "holding": r"\bholding\b",
            "centralizace": r"\bcentralizace\b",
            "efektivizace": r"\befektivizace\b",
            "parlament": r"\bparlament\b",
            "ministerstvo": r"\bministerstvo\b"
        }
        
        # Absurdní přirovnání
        absurd_comparisons = [
            r"jak když kráva",
            r"jak když slepice",
            r"jak když pes",
            r"jak když dítě",
            r"jak když ryba"
        ]
        
        # Slovensko-české odchylky
        slovak_patterns = [
            r"\bsme\b",
            r"\bsom\b",
            r"\bviděl som\b",
            r"\bmakáme\b"
        ]
        
        # Analýza všech odpovědí
        phrase_counts = defaultdict(int)
        comparison_counts = defaultdict(int)
        slovak_count = 0
        total_answers = len(self.qa_pairs)
        
        for qa_pair in self.qa_pairs:
            answer = qa_pair["answer"].lower()
            
            # Počítání charakteristických frází
            for phrase_name, pattern in babis_phrases.items():
                if re.search(pattern, answer, re.IGNORECASE):
                    phrase_counts[phrase_name] += 1
            
            # Počítání absurdních přirovnání
            for comparison in absurd_comparisons:
                if re.search(comparison, answer, re.IGNORECASE):
                    comparison_counts[comparison] += 1
            
            # Počítání slovenských odchylek
            for slovak_pattern in slovak_patterns:
                if re.search(slovak_pattern, answer, re.IGNORECASE):
                    slovak_count += 1
                    break
        
        # Výpočet procent
        phrase_percentages = {
            phrase: (count / total_answers) * 100 
            for phrase, count in phrase_counts.items()
        }
        
        comparison_percentages = {
            comp: (count / total_answers) * 100 
            for comp, count in comparison_counts.items()
        }
        
        slovak_percentage = (slovak_count / total_answers) * 100 if total_answers > 0 else 0
        
        result = {
            "total_answers": total_answers,
            "phrase_counts": dict(phrase_counts),
            "phrase_percentages": phrase_percentages,
            "comparison_counts": dict(comparison_counts),
            "comparison_percentages": comparison_percentages,
            "slovak_count": slovak_count,
            "slovak_percentage": slovak_percentage
        }
        
        print(f"✅ Styl analyzován")
        print(f"  - Celkem odpovědí: {result['total_answers']}")
        print(f"  - Slovenské odchylky: {result['slovak_count']} ({result['slovak_percentage']:.1f}%)")
        
        print(f"  - Nejčastější fráze:")
        sorted_phrases = sorted(phrase_percentages.items(), key=lambda x: x[1], reverse=True)
        for phrase, percentage in sorted_phrases[:5]:
            print(f"    {phrase}: {percentage:.1f}%")
        
        print(f"  - Absurdní přirovnání:")
        for comp, percentage in comparison_percentages.items():
            if percentage > 0:
                print(f"    {comp}: {percentage:.1f}%")
        
        return result
    
    def analyze_content_length(self) -> Dict[str, Any]:
        """Analyzuje délku otázek a odpovědí."""
        print("\n=== Analýza délky obsahu ===")
        
        question_lengths = [len(qa["question"]) for qa in self.qa_pairs]
        answer_lengths = [len(qa["answer"]) for qa in self.qa_pairs]
        
        result = {
            "questions": {
                "count": len(question_lengths),
                "min": min(question_lengths) if question_lengths else 0,
                "max": max(question_lengths) if question_lengths else 0,
                "avg": sum(question_lengths) / len(question_lengths) if question_lengths else 0,
                "lengths": question_lengths
            },
            "answers": {
                "count": len(answer_lengths),
                "min": min(answer_lengths) if answer_lengths else 0,
                "max": max(answer_lengths) if answer_lengths else 0,
                "avg": sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
                "lengths": answer_lengths
            }
        }
        
        print(f"✅ Délka obsahu analyzována")
        print(f"  - Otázky:")
        print(f"    Průměr: {result['questions']['avg']:.1f} znaků")
        print(f"    Min: {result['questions']['min']} znaků")
        print(f"    Max: {result['questions']['max']} znaků")
        print(f"  - Odpovědi:")
        print(f"    Průměr: {result['answers']['avg']:.1f} znaků")
        print(f"    Min: {result['answers']['min']} znaků")
        print(f"    Max: {result['answers']['max']} znaků")
        
        return result
    
    def generate_report(self) -> Dict[str, Any]:
        """Generuje kompletní report o kvalitě dat."""
        print("=== Generování reportu o kvalitě dat ===")
        
        if not self.load_dataset():
            return {"error": "Nepodařilo se načíst dataset"}
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "dataset_file": self.dataset_file,
            "basic_structure": self.check_basic_structure(),
            "babis_style": self.analyze_babis_style(),
            "content_length": self.analyze_content_length()
        }
        
        # Uložení reportu
        report_file = "data_quality_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Report uložen do: {report_file}")
        
        return report
    
    def create_visualizations(self, report: Dict[str, Any]):
        """Vytváří vizualizace pro report."""
        print("\n=== Vytváření vizualizací ===")
        
        try:
            # Nastavení stylu
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Analýza kvality Babišova datasetu', fontsize=16)
            
            # 1. Délka odpovědí
            answer_lengths = report['content_length']['answers']['lengths']
            axes[0, 0].hist(answer_lengths, bins=30, alpha=0.7, color='skyblue')
            axes[0, 0].set_title('Distribuce délky odpovědí')
            axes[0, 0].set_xlabel('Počet znaků')
            axes[0, 0].set_ylabel('Počet odpovědí')
            
            # 2. Nejčastější fráze
            phrase_data = report['babis_style']['phrase_percentages']
            top_phrases = sorted(phrase_data.items(), key=lambda x: x[1], reverse=True)[:10]
            phrases, percentages = zip(*top_phrases)
            
            axes[0, 1].barh(phrases, percentages, color='lightcoral')
            axes[0, 1].set_title('Nejčastější Babišovy fráze')
            axes[0, 1].set_xlabel('Procento odpovědí (%)')
            
            # 3. Délka otázek vs odpovědí
            question_lengths = report['content_length']['questions']['lengths']
            axes[1, 0].scatter(question_lengths, answer_lengths, alpha=0.6, color='green')
            axes[1, 0].set_title('Délka otázek vs odpovědí')
            axes[1, 0].set_xlabel('Délka otázky (znaky)')
            axes[1, 0].set_ylabel('Délka odpovědi (znaky)')
            
            # 4. Přehled statistik
            stats_text = f"""
            Celkem QA párů: {report['basic_structure']['qa_pairs_count']}
            Průměrná délka odpovědi: {report['content_length']['answers']['avg']:.0f} znaků
            """
            axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                           fontsize=12, verticalalignment='center')
            axes[1, 1].set_title('Přehled statistik')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Uložení grafu
            plot_file = "data_quality_analysis.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"✅ Vizualizace uložena do: {plot_file}")
            
        except Exception as e:
            print(f"Chyba při vytváření vizualizací: {e}")

def main():
    """Hlavní funkce pro kontrolu kvality dat."""
    checker = DataQualityChecker()
    report = checker.generate_report()
    
    if "error" not in report:
        checker.create_visualizations(report)
        print("\n✅ Kontrola kvality dat dokončena!")

if __name__ == "__main__":
    main() 