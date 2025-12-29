from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ctmap",
    version="0.1.0",
    author="Wang Ying",
    author_email="your_email@xxx",
    description="CTMAP: Adversarial profile alignment for cell type mapping in spatial transcriptomics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SDU-Math-SunLab/CTMAP-main",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.26.4",
        "pandas>=2.2.3",
        "scanpy>=1.9.0",
        "tqdm>=4.64.0",
        "scikit-learn>=1.5.2",
        "scipy>=1.13.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
