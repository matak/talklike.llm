#!/usr/bin/env python3
"""
Core functionality for Disk Manager Library
"""

import os
import shutil
import subprocess
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class DiskInfo:
    """Informace o diskovém prostoru"""
    filesystem: str
    size: str
    used: str
    available: str
    use_percent: int
    mounted_on: str

class DiskManager:
    """
    Správce diskového prostoru pro ML projekty
    
    Poskytuje funkce pro:
    - Kontrolu dostupného místa
    - Čištění cache
    - Nastavení network storage
    - Optimalizaci pro velké modely
    """
    
    def __init__(self, log_level: int = logging.INFO):
        """
        Inicializace DiskManager
        
        Args:
            log_level: Úroveň logování
        """
        self.setup_logging(log_level)
        self.network_storage_path = None
        self.cache_dirs = [
            os.path.expanduser("~/.cache/huggingface"),
            "/tmp",
            "/root/.cache",
            "/root/.local/share/huggingface",
            "/var/cache",
            "/usr/local/lib/python3.10/dist-packages/transformers/.cache",
            "/usr/local/lib/python3.10/dist-packages/huggingface_hub/.cache",
            "/usr/local/lib/python3.10/dist-packages/datasets/.cache"
        ]
        
        self.aggressive_cleanup_dirs = [
            "/tmp",
            "/var/tmp", 
            "/root/.cache",
            "/root/.local",
            "/root/.config",
            "/usr/local/lib/python3.10/dist-packages/transformers/.cache",
            "/usr/local/lib/python3.10/dist-packages/huggingface_hub/.cache",
            "/usr/local/lib/python3.10/dist-packages/datasets/.cache"
        ]
    
    def setup_logging(self, log_level: int):
        """Nastaví logování"""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('disk_manager.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_disk_info(self, path: str = "/") -> DiskInfo:
        """
        Získá informace o diskovém prostoru
        
        Args:
            path: Cesta k filesystem
            
        Returns:
            DiskInfo objekt s informacemi o disku
        """
        try:
            result = subprocess.run(['df', '-h', path], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 6:
                    use_percent = int(parts[4].rstrip('%'))
                    return DiskInfo(
                        filesystem=parts[0],
                        size=parts[1],
                        used=parts[2],
                        available=parts[3],
                        use_percent=use_percent,
                        mounted_on=parts[5]
                    )
        except Exception as e:
            self.logger.error(f"Chyba při získávání informací o disku: {e}")
        
        return None
    
    def check_disk_space(self, path: str = "/", threshold: int = 95) -> bool:
        """
        Zkontroluje dostupné místo na disku
        
        Args:
            path: Cesta k filesystem
            threshold: Kritická hranice v procentech
            
        Returns:
            True pokud je dost místa, False jinak
        """
        disk_info = self.get_disk_info(path)
        if disk_info:
            self.logger.info(f"💾 Použito místa na {path}: {disk_info.use_percent}%")
            
            if disk_info.use_percent > threshold:
                self.logger.warning(f"⚠️ VAROVÁNÍ: {path} je téměř plný ({disk_info.use_percent}%)")
                return False
            return True
        
        self.logger.warning(f"⚠️ Nelze zkontrolovat místo na disku pro {path}")
        return True
    
    def get_cache_size(self, path: str) -> int:
        """
        Vypočítá velikost cache adresáře
        
        Args:
            path: Cesta k cache adresáři
            
        Returns:
            Velikost v bytech
        """
        total_size = 0
        if os.path.exists(path):
            try:
                for dirpath, dirnames, filenames in os.walk(path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        try:
                            total_size += os.path.getsize(filepath)
                        except:
                            pass
            except Exception as e:
                self.logger.error(f"Chyba při výpočtu velikosti {path}: {e}")
        
        return total_size
    
    def cleanup_cache(self, cache_dirs: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Vyčistí cache adresáře
        
        Args:
            cache_dirs: Seznam adresářů k vyčištění (výchozí: self.cache_dirs)
            
        Returns:
            Slovník s informacemi o vyčištěných adresářích
        """
        if cache_dirs is None:
            cache_dirs = self.cache_dirs
        
        self.logger.info("🧹 Čistím cache...")
        cleaned_info = {}
        
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                try:
                    # Vypočítáme velikost před vyčištěním
                    total_size = self.get_cache_size(cache_dir)
                    
                    if total_size > 0:
                        size_gb = total_size / (1024**3)
                        self.logger.info(f"🗑️ Mažu cache v {cache_dir} (velikost: {size_gb:.1f} GB)")
                        
                        shutil.rmtree(cache_dir, ignore_errors=True)
                        os.makedirs(cache_dir, exist_ok=True)
                        
                        cleaned_info[cache_dir] = total_size
                except Exception as e:
                    self.logger.error(f"⚠️ Nelze vyčistit {cache_dir}: {e}")
        
        # Další vyčištění
        self._cleanup_pip_cache()
        self._cleanup_conda_cache()
        
        return cleaned_info
    
    def _cleanup_pip_cache(self):
        """Vyčistí pip cache"""
        try:
            subprocess.run(["pip", "cache", "purge"], capture_output=True, check=False)
            self.logger.info("🗑️ Vyčištěn pip cache")
        except Exception as e:
            self.logger.debug(f"Nelze vyčistit pip cache: {e}")
    
    def _cleanup_conda_cache(self):
        """Vyčistí conda cache"""
        try:
            subprocess.run(["conda", "clean", "-a", "-y"], capture_output=True, check=False)
            self.logger.info("🗑️ Vyčištěn conda cache")
        except Exception as e:
            self.logger.debug(f"Nelze vyčistit conda cache: {e}")
    
    def aggressive_cleanup(self) -> Dict[str, int]:
        """
        Agresivní vyčištění pro uvolnění maximálního místa
        
        Returns:
            Slovník s informacemi o vyčištěných adresářích
        """
        self.logger.info("🧹 Agresivní vyčištění...")
        
        # Vyčištění všech možných cache adresářů
        cleaned_info = {}
        
        for dir_path in self.aggressive_cleanup_dirs:
            if os.path.exists(dir_path):
                try:
                    total_size = self.get_cache_size(dir_path)
                    self.logger.info(f"🗑️ Mažu {dir_path}")
                    
                    shutil.rmtree(dir_path, ignore_errors=True)
                    os.makedirs(dir_path, exist_ok=True)
                    
                    if total_size > 0:
                        cleaned_info[dir_path] = total_size
                except Exception as e:
                    self.logger.error(f"⚠️ Nelze vyčistit {dir_path}: {e}")
        
        # Vyčištění log souborů
        self._cleanup_logs()
        
        # Synchronizace filesystem
        self._sync_filesystem()
        
        return cleaned_info
    
    def _cleanup_logs(self):
        """Vyčistí log soubory"""
        try:
            subprocess.run(["find", "/var/log", "-name", "*.log", "-delete"], 
                         capture_output=True, check=False)
            subprocess.run(["find", "/var/log", "-name", "*.gz", "-delete"], 
                         capture_output=True, check=False)
            self.logger.info("🗑️ Vyčištěny log soubory")
        except Exception as e:
            self.logger.debug(f"Nelze vyčistit logy: {e}")
    
    def _sync_filesystem(self):
        """Synchronizuje filesystem"""
        try:
            subprocess.run(["sync"], check=False)
            self.logger.info("💾 Synchronizováno filesystem")
        except Exception as e:
            self.logger.debug(f"Nelze synchronizovat filesystem: {e}")
    
    def setup_network_storage(self, network_path: str) -> bool:
        """
        Nastaví network storage pro cache
        
        Args:
            network_path: Cesta k network storage
            
        Returns:
            True pokud se nastavení povedlo
        """
        try:
            self.network_storage_path = network_path
            
            # Nastavení environment proměnných
            os.environ['HF_HOME'] = f'{network_path}/.cache/huggingface'
            os.environ['TRANSFORMERS_CACHE'] = f'{network_path}/.cache/huggingface/transformers'
            os.environ['HF_DATASETS_CACHE'] = f'{network_path}/.cache/huggingface/datasets'
            os.environ['HF_HUB_CACHE'] = f'{network_path}/.cache/huggingface/hub'
            
            # Vytvoření cache adresářů
            cache_dirs = [
                f'{network_path}/.cache/huggingface',
                f'{network_path}/.cache/huggingface/transformers',
                f'{network_path}/.cache/huggingface/datasets',
                f'{network_path}/.cache/huggingface/hub'
            ]
            
            for cache_dir in cache_dirs:
                os.makedirs(cache_dir, exist_ok=True)
            
            self.logger.info(f"💾 Cache nastaven na: {os.environ['HF_HOME']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Chyba při nastavení network storage: {e}")
            return False
    
    def optimize_for_large_models(self, model_name: str) -> bool:
        """
        Optimalizuje prostředí pro velké modely
        
        Args:
            model_name: Název modelu
            
        Returns:
            True pokud je optimalizace úspěšná
        """
        large_models = ["mistral", "llama", "gpt", "falcon", "t5"]
        is_large_model = any(model in model_name.lower() for model in large_models)
        
        if is_large_model:
            self.logger.info("🧹 Optimalizace pro velký model...")
            
            # Kontrola místa
            if not self.check_disk_space():
                self.logger.warning("⚠️ Málo místa, zkouším vyčištění...")
                self.cleanup_cache()
                
                if not self.check_disk_space():
                    self.logger.warning("⚠️ Stále málo místa, agresivní vyčištění...")
                    self.aggressive_cleanup()
                    
                    if not self.check_disk_space():
                        self.logger.error("❌ Stále není dost místa pro velký model")
                        return False
            
            # Nastavení network storage pokud není nastaveno
            if not self.network_storage_path:
                self.setup_network_storage("/workspace")
        
        return True
    
    def get_storage_summary(self) -> Dict[str, any]:
        """
        Získá shrnutí úložiště
        
        Returns:
            Slovník s informacemi o úložišti
        """
        summary = {
            "root_disk": self.get_disk_info("/"),
            "network_storage": None,
            "cache_dirs": {}
        }
        
        # Network storage info
        if self.network_storage_path:
            summary["network_storage"] = self.get_disk_info(self.network_storage_path)
        
        # Cache info
        for cache_dir in self.cache_dirs:
            if os.path.exists(cache_dir):
                size = self.get_cache_size(cache_dir)
                summary["cache_dirs"][cache_dir] = {
                    "size_bytes": size,
                    "size_gb": size / (1024**3)
                }
        
        return summary
    
    def print_storage_summary(self):
        """Vypíše shrnutí úložiště"""
        summary = self.get_storage_summary()
        
        print("💾 Shrnutí úložiště:")
        print("=" * 50)
        
        if summary["root_disk"]:
            disk = summary["root_disk"]
            print(f"Root filesystem: {disk.filesystem}")
            print(f"  Velikost: {disk.size}, Použito: {disk.used}, Volné: {disk.available}")
            print(f"  Použití: {disk.use_percent}%")
        
        if summary["network_storage"]:
            disk = summary["network_storage"]
            print(f"Network storage: {disk.filesystem}")
            print(f"  Velikost: {disk.size}, Použito: {disk.used}, Volné: {disk.available}")
            print(f"  Použití: {disk.use_percent}%")
        
        if summary["cache_dirs"]:
            print(f"Cache adresáře:")
            for cache_dir, info in summary["cache_dirs"].items():
                print(f"  {cache_dir}: {info['size_gb']:.1f} GB")


# Utility funkce pro snadné použití
def quick_cleanup():
    """Rychlé vyčištění cache"""
    dm = DiskManager()
    return dm.cleanup_cache()

def check_and_cleanup(threshold: int = 95):
    """Zkontroluje místo a vyčistí cache pokud je potřeba"""
    dm = DiskManager()
    if not dm.check_disk_space(threshold=threshold):
        dm.cleanup_cache()
        return dm.check_disk_space(threshold=threshold)
    return True

def setup_for_ml_project(network_path: str = "/workspace"):
    """Nastaví prostředí pro ML projekt"""
    dm = DiskManager()
    dm.setup_network_storage(network_path)
    return dm 