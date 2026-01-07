# CTMAP: an adversarial cross-modal learning framework for accurate and robust cell-type annotation in single-cell spatial transcriptomics

## Introduction
`CTMAP` is a deep learning framework for accurate cell-type annotation in single-cell spatial transcriptomics. It aligns scRNA-seq reference data with spatial transcriptomic profiles in a shared latent space through adversarial learning, enabling precise and robust cell-type annotation.

<div align="center">
  <img src="./overview.jpg" alt="CTMAP Framework Overview" width="80%">
</div>

## Installation
CTMAP is designed to run on GPUs and is tested with Python 3.9 in a Conda environment.
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/SDU-Math-SunLab/CTMAP.git
    cd CTMAP
    ```
2.  **Create the Conda environment**:
    ```bash
    conda env create -f environment.yml
    ```
3.  **Activate the environment**:
    ```bash
    conda activate ctmap
    ```
## Data Preparation
To run the tutorial, place the required datasets (`adata_merfish.h5ad` and `adata_rna.h5ad`) in the following directory:

```bash
CTMAP/dataset/MERFISH/
```

Update the file paths in `tutorial.ipynb` if your data is located elsewhere.
## Quick Start

After setting up the environment and preparing the data, you can run CTMAP with:

```bash
python CTMAP/run.py
```
## Tutorials
* **[CTMAP pipeline on the MERFISH dataset](./tutorial.ipynb)**: Workflow covering data loading, model training, evaluation (Accuracy, NMI, ARI), and visualization for the mouse hypothalamic preoptic MERFISH dataset.

## Citation
Citation details will be added upon publication.

