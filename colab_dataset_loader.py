"""
Modul pro načítání a zpracování datasetu
"""
import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

def create_babis_dataset():
    """Načte skutečný dataset s Babišovými výroky z JSONL souborů"""
    
    # Cesta k souborům s daty
    data_dir = Path("final")
    jsonl_files = list(data_dir.glob("batch_*_babis_output_qa.jsonl"))
    
    if not jsonl_files:
        raise FileNotFoundError(f"Nenalezeny žádné JSONL soubory v adresáři {data_dir}")
    
    conversations = []
    
    # Načtení všech souborů
    for file_path in jsonl_files:
        print(f"Načítám {file_path.name}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    question = data.get('question', '')
                    answer = data.get('answer', '')
                    
                    # Vytvoření konverzace ve formátu pro fine-tuning
                    conversation = {
                        "prompt": question,
                        "completion": answer,
                        "full_conversation": f"Uživatel: {question}\nAndrej Babiš: {answer}"
                    }
                    conversations.append(conversation)
                    
                except json.JSONDecodeError as e:
                    print(f"Chyba při parsování JSON na řádku {line_num} v souboru {file_path.name}: {e}")
                    continue
    
    print(f"Celkem načteno {len(conversations)} konverzací z {len(jsonl_files)} souborů")
    
    if len(conversations) == 0:
        raise ValueError("Nebyla načtena žádná konverzace z JSONL souborů")
    
    # Rozdělení na train/validation
    train_data, eval_data = train_test_split(conversations, train_size=0.9, random_state=42)
    
    # Vytvoření Dataset objektů
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    # Vytvoření DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': eval_dataset
    })
    
    return dataset_dict

# Vytvoření datasetu
dataset = create_babis_dataset()
print(f"Dataset vytvořen:")
print(f"Train samples: {len(dataset['train'])}")
print(f"Validation samples: {len(dataset['validation'])}")
print(f"\nPříklad dat:")
print(dataset['train'][0])