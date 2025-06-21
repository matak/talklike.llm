#!/usr/bin/env python3
"""
Hlavní skript pro spuštění celého procesu přípravy dat.
Automaticky spustí všechny kroky přípravy datasetu pro fine-tuning.
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from typing import List, Dict, Any

class DataPreparationRunner:
    def __init__(self):
        """Inicializace runneru pro přípravu dat."""
        self.steps = [
            {
                "name": "Kontrola prostředí",
                "script": None,
                "function": self.check_environment
            },
            {
                "name": "Generování odpovědí",
                "script": "generate_answers.py",
                "function": None
            },
            {
                "name": "Generování QA datasetu",
                "script": "generate_qa_dataset.py",
                "function": None
            },
            {
                "name": "Sloučení dat",
                "script": "dataset_merger.py",
                "function": None
            },
            {
                "name": "Kontrola kvality dat",
                "script": "data_quality_check.py",
                "function": None
            }
        ]
        
        self.results = []
        self.start_time = None
        
    def check_environment(self) -> Dict[str, Any]:
        """Kontroluje prostředí před spuštěním."""
        print("=== Kontrola prostředí ===")
        
        checks = {
            "openai_api_key": False,
            "python_version": False,
            "required_files": False,
            "output_directories": False
        }
        
        # Kontrola OpenAI API klíče
        if os.getenv('OPENAI_API_KEY'):
            checks["openai_api_key"] = True
            print("✅ OPENAI_API_KEY je nastaven")
        else:
            print("❌ OPENAI_API_KEY není nastaven")
        
        # Kontrola verze Pythonu
        if sys.version_info >= (3, 8):
            checks["python_version"] = True
            print(f"✅ Python verze: {sys.version}")
        else:
            print(f"❌ Python verze {sys.version} není podporována (vyžaduje se 3.8+)")
        
        # Kontrola požadovaných souborů
        required_files = [
            "babis_templates_400.json",
            "LLM.CreateAnswers.systemPrompt.md",
            "LLM.CreateDialogue.systemPrompt.md",
            "availablemodels.json"
        ]
        
        missing_files = []
        for file in required_files:
            if os.path.exists(file):
                print(f"✅ {file}")
            else:
                missing_files.append(file)
                print(f"❌ {file} - chybí")
        
        if not missing_files:
            checks["required_files"] = True
        
        # Vytvoření výstupních adresářů
        output_dirs = [
            "data",
            "data/generated_batches",
            "data/final",
            "logs"
        ]
        
        for dir_path in output_dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ Adresář {dir_path} připraven")
        
        checks["output_directories"] = True
        
        # Celkový výsledek
        all_checks_passed = all(checks.values())
        
        if all_checks_passed:
            print("✅ Všechny kontroly prostředí prošly")
        else:
            print("❌ Některé kontroly prostředí selhaly")
        
        return {
            "success": all_checks_passed,
            "checks": checks,
            "missing_files": missing_files
        }
    
    def run_script(self, script_name: str) -> Dict[str, Any]:
        """Spustí Python skript a vrátí výsledek."""
        print(f"\n=== Spouštím {script_name} ===")
        
        start_time = time.time()
        
        try:
            # Spuštění skriptu
            result = subprocess.run(
                [sys.executable, script_name],
                capture_output=True,
                text=True,
                encoding='utf-8',
                cwd=os.getcwd()
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Výpis výstupu
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
            
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            
            success = result.returncode == 0
            
            if success:
                print(f"✅ {script_name} dokončen úspěšně ({duration:.1f}s)")
            else:
                print(f"❌ {script_name} selhal (kód: {result.returncode}, {duration:.1f}s)")
            
            return {
                "success": success,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration
            }
            
        except Exception as e:
            print(f"❌ Chyba při spouštění {script_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    def run_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Spustí jeden krok přípravy dat."""
        print(f"\n{'='*50}")
        print(f"KROK: {step['name']}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        try:
            if step["function"]:
                # Spuštění funkce
                result = step["function"]()
            elif step["script"]:
                # Spuštění skriptu
                result = self.run_script(step["script"])
            else:
                result = {"success": False, "error": "Není definována funkce ani skript"}
            
            end_time = time.time()
            duration = end_time - start_time
            
            step_result = {
                "name": step["name"],
                "success": result.get("success", False),
                "duration": duration,
                "details": result
            }
            
            if step_result["success"]:
                print(f"✅ Krok '{step['name']}' dokončen úspěšně")
            else:
                print(f"❌ Krok '{step['name']}' selhal")
            
            return step_result
            
        except Exception as e:
            print(f"❌ Neočekávaná chyba v kroku '{step['name']}': {e}")
            return {
                "name": step["name"],
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e)
            }
    
    def run_all_steps(self) -> Dict[str, Any]:
        """Spustí všechny kroky přípravy dat."""
        print("🚀 Spouštím kompletní přípravu dat pro fine-tuning")
        print(f"Čas spuštění: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.start_time = time.time()
        self.results = []
        
        # Spuštění všech kroků
        for i, step in enumerate(self.steps, 1):
            print(f"\n📋 Krok {i}/{len(self.steps)}")
            
            result = self.run_step(step)
            self.results.append(result)
            
            # Kontrola, zda pokračovat
            if not result["success"]:
                print(f"\n❌ Krok '{step['name']}' selhal. Chcete pokračovat? (ano/ne): ", end="")
                try:
                    user_input = input().lower().strip()
                    if user_input not in ['ano', 'a', 'yes', 'y']:
                        print("Příprava dat přerušena uživatelem.")
                        break
                except KeyboardInterrupt:
                    print("\nPříprava dat přerušena uživatelem.")
                    break
        
        # Výpočet celkového času
        total_duration = time.time() - self.start_time
        
        # Shrnutí výsledků
        successful_steps = sum(1 for r in self.results if r["success"])
        total_steps = len(self.results)
        
        summary = {
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "failed_steps": total_steps - successful_steps,
            "total_duration": total_duration,
            "success_rate": successful_steps / total_steps if total_steps > 0 else 0,
            "results": self.results
        }
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Vypíše shrnutí výsledků."""
        print(f"\n{'='*60}")
        print("📊 SHRNUTÍ PŘÍPRAVY DAT")
        print(f"{'='*60}")
        
        print(f"Celkový čas: {summary['total_duration']:.1f} sekund")
        print(f"Úspěšných kroků: {summary['successful_steps']}/{summary['total_steps']}")
        print(f"Úspěšnost: {summary['success_rate']:.1%}")
        
        print(f"\nDetailní výsledky:")
        for i, result in enumerate(summary["results"], 1):
            status = "✅" if result["success"] else "❌"
            duration = f"{result['duration']:.1f}s"
            print(f"  {i}. {status} {result['name']} ({duration})")
            
            if not result["success"] and "error" in result:
                print(f"     Chyba: {result['error']}")
        
        if summary["success_rate"] == 1.0:
            print(f"\n🎉 Všechny kroky byly úspěšné! Dataset je připraven pro fine-tuning.")
        elif summary["success_rate"] >= 0.8:
            print(f"\n⚠️  Většina kroků byla úspěšná, ale některé selhaly.")
        else:
            print(f"\n❌ Mnoho kroků selhalo. Zkontrolujte chyby a zkuste znovu.")
    
    def save_log(self, summary: Dict[str, Any]):
        """Uloží log z přípravy dat."""
        log_file = f"logs/data_preparation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "environment": {
                "python_version": sys.version,
                "working_directory": os.getcwd(),
                "openai_api_key_set": bool(os.getenv('OPENAI_API_KEY'))
            }
        }
        
        try:
            import json
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            print(f"📝 Log uložen do: {log_file}")
        except Exception as e:
            print(f"⚠️  Nepodařilo se uložit log: {e}")

def main():
    """Hlavní funkce pro spuštění přípravy dat."""
    runner = DataPreparationRunner()
    
    try:
        # Spuštění všech kroků
        summary = runner.run_all_steps()
        
        # Výpis shrnutí
        runner.print_summary(summary)
        
        # Uložení logu
        runner.save_log(summary)
        
        # Návratový kód
        if summary["success_rate"] == 1.0:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Příprava dat přerušena uživatelem.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Neočekávaná chyba: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 