# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ctmap",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="CTMAP: Cell Type Mapping with Adversarial Profile alignment for spatial transcriptomics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/CTMAP",
    packages=find_packages(),  # 自动找到 ctmap 包
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.26.4",
        "pandas>=2.2.3",
        "scanpy>=1.10.3",
        "torch>=2.4.1",
        "tqdm>=4.64.0",
        "scikit-learn>=1.5.2",
        "scipy>=1.13.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)