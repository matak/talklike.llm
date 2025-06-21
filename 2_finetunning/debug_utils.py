import os
import json
from datetime import datetime

class DatasetDebugger:
    """Třída pro debugování a ukládání mezikroků zpracování datasetu"""
    
    def __init__(self, debug_dir="debug_dataset"):
        self.debug_dir = debug_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_dir = f"{debug_dir}_{self.timestamp}"
        
        # Vytvoření debug adresáře
        os.makedirs(self.debug_dir, exist_ok=True)
        print(f"🔍 Debug adresář vytvořen: {self.debug_dir}")
    
    def save_step(self, step_name, data, description=""):
        """Uloží krok zpracování datasetu"""
        step_file = os.path.join(self.debug_dir, f"step_{step_name}.json")
        
        # Přidání metadat
        debug_info = {
            "step_name": step_name,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "data_type": type(data).__name__,
            "data_count": len(data) if hasattr(data, '__len__') else "N/A"
        }
        
        # Uložení dat podle typu
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                # Seznam slovníků - uložíme jako JSON
                debug_info["data"] = data
            else:
                # Jiný typ dat - uložíme jako text
                debug_info["data"] = [str(item) for item in data]
        elif isinstance(data, dict):
            debug_info["data"] = data
        else:
            debug_info["data"] = str(data)
        
        with open(step_file, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Uložen debug krok: {step_name} -> {step_file}")
        
        # Vytvoření také čitelné verze pro první 3 položky
        if isinstance(data, list) and len(data) > 0:
            readable_file = os.path.join(self.debug_dir, f"step_{step_name}_readable.txt")
            with open(readable_file, 'w', encoding='utf-8') as f:
                f.write(f"Debug krok: {step_name}\n")
                f.write(f"Čas: {debug_info['timestamp']}\n")
                f.write(f"Popis: {description}\n")
                f.write(f"Počet položek: {len(data)}\n")
                f.write("-" * 80 + "\n\n")
                
                for i, item in enumerate(data[:3]):  # První 3 položky
                    f.write(f"Položka {i+1}:\n")
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if key == 'content' and isinstance(value, str) and len(value) > 200:
                                f.write(f"  {key}: {value[:200]}...\n")
                            else:
                                f.write(f"  {key}: {value}\n")
                    else:
                        f.write(f"  {item}\n")
                    f.write("\n")
                
                if len(data) > 3:
                    f.write(f"... a dalších {len(data) - 3} položek\n")
    
    def save_sample(self, step_name, sample_data, sample_index=0):
        """Uloží ukázkovou položku z datasetu"""
        sample_file = os.path.join(self.debug_dir, f"sample_{step_name}_{sample_index}.json")
        
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        print(f"📝 Uložena ukázka: {step_name} -> {sample_file}")
    
    def create_summary(self):
        """Vytvoří shrnutí všech debug kroků"""
        summary_file = os.path.join(self.debug_dir, "debug_summary.txt")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("DEBUG SHRNUTÍ ZPRACOVÁNÍ DATASETU\n")
            f.write("=" * 50 + "\n")
            f.write(f"Čas vytvoření: {self.timestamp}\n")
            f.write(f"Debug adresář: {self.debug_dir}\n\n")
            
            # Najdeme všechny debug soubory
            debug_files = [f for f in os.listdir(self.debug_dir) if f.startswith("step_") and f.endswith(".json")]
            debug_files.sort()
            
            for debug_file in debug_files:
                try:
                    with open(os.path.join(self.debug_dir, debug_file), 'r', encoding='utf-8') as df:
                        debug_info = json.load(df)
                    
                    f.write(f"Krok: {debug_info['step_name']}\n")
                    f.write(f"  Čas: {debug_info['timestamp']}\n")
                    f.write(f"  Popis: {debug_info['description']}\n")
                    f.write(f"  Typ dat: {debug_info['data_type']}\n")
                    f.write(f"  Počet: {debug_info['data_count']}\n")
                    f.write("\n")
                except Exception as e:
                    f.write(f"Chyba při čtení {debug_file}: {e}\n\n")
        
        print(f"📋 Vytvořeno shrnutí: {summary_file}") 