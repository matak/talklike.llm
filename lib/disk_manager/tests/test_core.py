#!/usr/bin/env python3
"""
Tests pro Disk Manager Library
"""

import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
import shutil

# Import testovaných modulů
from ..core import DiskManager, DiskInfo, quick_cleanup, check_and_cleanup


class TestDiskInfo(unittest.TestCase):
    """Testy pro DiskInfo dataclass"""
    
    def test_disk_info_creation(self):
        """Test vytvoření DiskInfo objektu"""
        disk_info = DiskInfo(
            filesystem="/dev/sda1",
            size="100G",
            used="50G",
            available="50G",
            use_percent=50,
            mounted_on="/"
        )
        
        self.assertEqual(disk_info.filesystem, "/dev/sda1")
        self.assertEqual(disk_info.size, "100G")
        self.assertEqual(disk_info.used, "50G")
        self.assertEqual(disk_info.available, "50G")
        self.assertEqual(disk_info.use_percent, 50)
        self.assertEqual(disk_info.mounted_on, "/")


class TestDiskManager(unittest.TestCase):
    """Testy pro DiskManager třídu"""
    
    def setUp(self):
        """Setup před každým testem"""
        self.dm = DiskManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup po každém testu"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test inicializace DiskManager"""
        self.assertIsNotNone(self.dm.logger)
        self.assertIsNone(self.dm.network_storage_path)
        self.assertIsInstance(self.dm.cache_dirs, list)
        self.assertIsInstance(self.dm.aggressive_cleanup_dirs, list)
    
    @patch('subprocess.run')
    def test_get_disk_info_success(self, mock_run):
        """Test úspěšného získání informací o disku"""
        # Mock výstup df příkazu
        mock_output = "Filesystem      Size  Used Avail Use% Mounted on\n/dev/sda1       100G   50G   50G  50% /"
        mock_run.return_value.stdout = mock_output
        
        disk_info = self.dm.get_disk_info("/")
        
        self.assertIsNotNone(disk_info)
        self.assertEqual(disk_info.filesystem, "/dev/sda1")
        self.assertEqual(disk_info.size, "100G")
        self.assertEqual(disk_info.used, "50G")
        self.assertEqual(disk_info.available, "50G")
        self.assertEqual(disk_info.use_percent, 50)
        self.assertEqual(disk_info.mounted_on, "/")
    
    @patch('subprocess.run')
    def test_get_disk_info_failure(self, mock_run):
        """Test neúspěšného získání informací o disku"""
        mock_run.side_effect = Exception("Command failed")
        
        disk_info = self.dm.get_disk_info("/")
        
        self.assertIsNone(disk_info)
    
    @patch.object(DiskManager, 'get_disk_info')
    def test_check_disk_space_sufficient(self, mock_get_disk_info):
        """Test kontroly dostatku místa na disku"""
        # Mock dostatku místa
        mock_disk_info = DiskInfo(
            filesystem="/dev/sda1",
            size="100G",
            used="50G",
            available="50G",
            use_percent=50,
            mounted_on="/"
        )
        mock_get_disk_info.return_value = mock_disk_info
        
        result = self.dm.check_disk_space("/", threshold=95)
        
        self.assertTrue(result)
    
    @patch.object(DiskManager, 'get_disk_info')
    def test_check_disk_space_insufficient(self, mock_get_disk_info):
        """Test kontroly nedostatku místa na disku"""
        # Mock nedostatku místa
        mock_disk_info = DiskInfo(
            filesystem="/dev/sda1",
            size="100G",
            used="95G",
            available="5G",
            use_percent=95,
            mounted_on="/"
        )
        mock_get_disk_info.return_value = mock_disk_info
        
        result = self.dm.check_disk_space("/", threshold=90)
        
        self.assertFalse(result)
    
    def test_get_cache_size(self):
        """Test výpočtu velikosti cache"""
        # Vytvoření testovacích souborů
        test_file1 = os.path.join(self.temp_dir, "test1.txt")
        test_file2 = os.path.join(self.temp_dir, "test2.txt")
        
        with open(test_file1, 'w') as f:
            f.write("test content 1")
        
        with open(test_file2, 'w') as f:
            f.write("test content 2")
        
        size = self.dm.get_cache_size(self.temp_dir)
        
        # Velikost by měla být větší než 0
        self.assertGreater(size, 0)
    
    def test_get_cache_size_nonexistent(self):
        """Test výpočtu velikosti neexistujícího cache"""
        size = self.dm.get_cache_size("/nonexistent/path")
        
        self.assertEqual(size, 0)
    
    @patch.object(DiskManager, 'get_cache_size')
    @patch('shutil.rmtree')
    @patch('os.makedirs')
    def test_cleanup_cache(self, mock_makedirs, mock_rmtree, mock_get_cache_size):
        """Test čištění cache"""
        # Mock velikosti cache
        mock_get_cache_size.return_value = 1024 * 1024  # 1MB
        
        result = self.dm.cleanup_cache([self.temp_dir])
        
        self.assertIsInstance(result, dict)
        mock_rmtree.assert_called()
        mock_makedirs.assert_called()
    
    def test_setup_network_storage(self):
        """Test nastavení network storage"""
        result = self.dm.setup_network_storage(self.temp_dir)
        
        self.assertTrue(result)
        self.assertEqual(self.dm.network_storage_path, self.temp_dir)
        
        # Kontrola environment proměnných
        self.assertIn('HF_HOME', os.environ)
        self.assertIn('TRANSFORMERS_CACHE', os.environ)
        self.assertIn('HF_DATASETS_CACHE', os.environ)
        self.assertIn('HF_HUB_CACHE', os.environ)
    
    def test_optimize_for_large_models(self):
        """Test optimalizace pro velké modely"""
        # Test s velkým modelem
        result = self.dm.optimize_for_large_models("mistralai/Mistral-7B-Instruct-v0.3")
        
        # Mělo by vrátit True (pokud není problém s místem)
        self.assertIsInstance(result, bool)
    
    def test_optimize_for_small_models(self):
        """Test optimalizace pro malé modely"""
        # Test s malým modelem
        result = self.dm.optimize_for_large_models("gpt2")
        
        # Mělo by vrátit True (malé modely nevyžadují optimalizaci)
        self.assertTrue(result)


class TestUtilityFunctions(unittest.TestCase):
    """Testy pro utility funkce"""
    
    def test_quick_cleanup(self):
        """Test rychlého vyčištění"""
        result = quick_cleanup()
        
        self.assertIsInstance(result, dict)
    
    @patch('disk_manager.core.DiskManager')
    def test_check_and_cleanup(self, mock_disk_manager_class):
        """Test kontroly a vyčištění"""
        # Mock DiskManager instance
        mock_dm = MagicMock()
        mock_disk_manager_class.return_value = mock_dm
        
        # Test s dostatkem místa
        mock_dm.check_disk_space.return_value = True
        
        result = check_and_cleanup(threshold=95)
        
        self.assertTrue(result)
        mock_dm.check_disk_space.assert_called_once_with(threshold=95)
        mock_dm.cleanup_cache.assert_not_called()
        
        # Test s nedostatkem místa
        mock_dm.check_disk_space.return_value = False
        
        result = check_and_cleanup(threshold=95)
        
        mock_dm.cleanup_cache.assert_called_once()
        mock_dm.check_disk_space.assert_called()


if __name__ == '__main__':
    unittest.main() 