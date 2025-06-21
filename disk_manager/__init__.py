#!/usr/bin/env python3
"""
Disk Manager Library
===================

Knihovna pro správu diskového prostoru, čištění cache a optimalizaci úložiště pro ML projekty.

Použití:
    from disk_manager import DiskManager
    
    dm = DiskManager()
    if dm.check_disk_space():
        dm.cleanup_cache()
        dm.setup_network_storage("/workspace")
"""

from .core import (
    DiskManager,
    DiskInfo,
    quick_cleanup,
    check_and_cleanup,
    setup_for_ml_project
)

__version__ = "1.0.0"
__author__ = "Roman Matena"
__email__ = "roman@matena.cz"

__all__ = [
    "DiskManager",
    "DiskInfo", 
    "quick_cleanup",
    "check_and_cleanup",
    "setup_for_ml_project"
] 