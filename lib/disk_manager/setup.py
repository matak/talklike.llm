#!/usr/bin/env python3
"""
Setup script pro Disk Manager Library
"""

from setuptools import setup, find_packages
import os

# Načtení README
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Disk Manager Library pro správu diskového prostoru v ML projektech"

setup(
    name="disk-manager",
    version="1.0.0",
    author="Roman Matena",
    author_email="roman@matena.cz",
    description="Knihovna pro správu diskového prostoru, čištění cache a optimalizaci úložiště pro ML projekty",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/disk-manager",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Systems Administration",
        "Topic :: Utilities",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Základní závislosti - žádné externí závislosti
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "ml": [
            "torch>=1.9.0",
            "transformers>=4.20.0",
            "datasets>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "disk-manager=disk_manager.core:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="disk management, cache cleanup, storage optimization, machine learning, ml, ai",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/disk-manager/issues",
        "Source": "https://github.com/yourusername/disk-manager",
        "Documentation": "https://github.com/yourusername/disk-manager/blob/main/README.md",
    },
) 