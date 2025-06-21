# Disk Manager Library

Knihovna pro správu diskového prostoru, čištění cache a optimalizaci úložiště pro ML projekty.

## 🚀 Instalace

### Lokální instalace
```bash
cd disk_manager
pip install -e .
```

### Z PyPI (až bude publikováno)
```bash
pip install disk-manager
```

## 📖 Použití

### Základní použití

```python
from disk_manager import DiskManager

# Inicializace
dm = DiskManager()

# Kontrola místa na disku
if dm.check_disk_space():
    print("✅ Dost místa na disku")
else:
    print("⚠️ Málo místa na disku")
    dm.cleanup_cache()
```

### Rychlé funkce

```python
from disk_manager import quick_cleanup, check_and_cleanup, setup_for_ml_project

# Rychlé vyčištění cache
quick_cleanup()

# Kontrola a vyčištění pokud je potřeba
check_and_cleanup(threshold=95)

# Nastavení pro ML projekt
dm = setup_for_ml_project("/workspace")
```

### Optimalizace pro velké modely

```python
from disk_manager import DiskManager

dm = DiskManager()

# Optimalizace pro Mistral/Llama
if dm.optimize_for_large_models("mistralai/Mistral-7B-Instruct-v0.3"):
    print("✅ Prostředí optimalizováno pro velký model")
else:
    print("❌ Nedost místa pro velký model")
```

### Nastavení network storage

```python
dm = DiskManager()

# Nastavení cache na network storage
dm.setup_network_storage("/workspace")

# Kontrola nastavení
dm.print_storage_summary()
```

## 🔧 API Reference

### DiskManager

Hlavní třída pro správu diskového prostoru.

#### Metody

##### `__init__(log_level=logging.INFO)`
Inicializuje DiskManager s volitelnou úrovní logování.

##### `check_disk_space(path="/", threshold=95) -> bool`
Zkontroluje dostupné místo na disku.

**Parametry:**
- `path`: Cesta k filesystem (výchozí: "/")
- `threshold`: Kritická hranice v procentech (výchozí: 95)

**Vrací:** `True` pokud je dost místa, `False` jinak

##### `cleanup_cache(cache_dirs=None) -> Dict[str, int]`
Vyčistí cache adresáře.

**Parametry:**
- `cache_dirs`: Seznam adresářů k vyčištění (výchozí: přednastavené adresáře)

**Vrací:** Slovník s informacemi o vyčištěných adresářích

##### `aggressive_cleanup() -> Dict[str, int]`
Agresivní vyčištění pro uvolnění maximálního místa.

**Vrací:** Slovník s informacemi o vyčištěných adresářích

##### `setup_network_storage(network_path) -> bool`
Nastaví network storage pro cache.

**Parametry:**
- `network_path`: Cesta k network storage

**Vrací:** `True` pokud se nastavení povedlo

##### `optimize_for_large_models(model_name) -> bool`
Optimalizuje prostředí pro velké modely.

**Parametry:**
- `model_name`: Název modelu

**Vrací:** `True` pokud je optimalizace úspěšná

##### `get_storage_summary() -> Dict[str, any]`
Získá shrnutí úložiště.

**Vrací:** Slovník s informacemi o úložišti

##### `print_storage_summary()`
Vypíše shrnutí úložiště do konzole.

### Utility funkce

#### `quick_cleanup() -> Dict[str, int]`
Rychlé vyčištění cache.

#### `check_and_cleanup(threshold=95) -> bool`
Zkontroluje místo a vyčistí cache pokud je potřeba.

#### `setup_for_ml_project(network_path="/workspace") -> DiskManager`
Nastaví prostředí pro ML projekt.

## 📊 Příklad výstupu

```
💾 Shrnutí úložiště:
==================================================
Root filesystem: overlay
  Velikost: 20G, Použito: 19G, Volné: 1G
  Použití: 95%
Network storage: mfs#eu-cz-1.runpod.net:9421
  Velikost: 245T, Použito: 162T, Volné: 84T
  Použití: 66%
Cache adresáře:
  /root/.cache: 2.1 GB
  /tmp: 0.5 GB
```

## 🛠️ Integrace s fine-tuning skripty

```python
from disk_manager import DiskManager

def main():
    # Inicializace
    dm = DiskManager()
    
    # Kontrola místa
    if not dm.check_disk_space():
        print("⚠️ Málo místa, čistím cache...")
        dm.cleanup_cache()
        
        if not dm.check_disk_space():
            print("⚠️ Stále málo místa, agresivní vyčištění...")
            dm.aggressive_cleanup()
    
    # Nastavení network storage
    dm.setup_network_storage("/workspace")
    
    # Optimalizace pro velký model
    if not dm.optimize_for_large_models("mistralai/Mistral-7B-Instruct-v0.3"):
        print("❌ Nedost místa pro velký model")
        return
    
    # Pokračování s fine-tuningem...
```

## 🔍 Debug a logování

```python
import logging
from disk_manager import DiskManager

# Nastavení detailního logování
dm = DiskManager(log_level=logging.DEBUG)

# Všechny operace se logují do souboru disk_manager.log
dm.cleanup_cache()
```

## 📦 Balíčkování

### Vývoj
```bash
# Instalace v development módu
pip install -e .

# Spuštění testů
python -m pytest

# Kontrola kódu
black disk_manager/
flake8 disk_manager/
mypy disk_manager/
```

### Publikování
```bash
# Build balíčku
python setup.py sdist bdist_wheel

# Upload na PyPI
twine upload dist/*
```

## 🤝 Příspěvky

1. Fork repozitáře
2. Vytvořte feature branch (`git checkout -b feature/amazing-feature`)
3. Commit změn (`git commit -m 'Add amazing feature'`)
4. Push do branch (`git push origin feature/amazing-feature`)
5. Otevřete Pull Request

## 📄 Licence

Tento projekt je licencován pod MIT licencí - viz soubor [LICENSE](LICENSE) pro detaily.

## 🆘 Podpora

- **Issues**: [GitHub Issues](https://github.com/yourusername/disk-manager/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/disk-manager/discussions)
- **Email**: roman@matena.cz

## 🗺️ Roadmap

- [ ] Podpora pro více filesystemů
- [ ] Automatické zálohování před vyčištěním
- [ ] Integrace s Docker
- [ ] Webové rozhraní pro monitoring
- [ ] Podpora pro cloud storage (S3, GCS, Azure)
- [ ] Machine learning pro predikci místa
- [ ] Grafické rozhraní (GUI)

## 📈 Statistiky

- **Velikost knihovny**: ~15 KB
- **Závislosti**: 0 (pouze standardní knihovny)
- **Podporované Python verze**: 3.8+
- **Podporované OS**: Linux, macOS, Windows 