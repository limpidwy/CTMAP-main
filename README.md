# CTMAP: Adversarially Aligned Attention Networks for Robust Cell-Type Annotation in Single-Cell Spatial Transcriptomics
## Introduction
`CTMAP` is a deep learning framework for accurate cell type annotation of single-cell resolution spatial transcriptomics data by integrating labeled scRNA-seq reference with domain adversarial alignment in a shared latent space.
<div align="center">
  <img src="./overview.jpg" alt="Overview" width="80%">
</div>


## Prerequisites
It is recommended to use Python version `3.9`.
* Set up conda environment for `CTMAP`:
  ```
  conda create -n ctmap python=3.9
  
  conda activate ctmap
  ```
* To build the required environment:
  ```
  conda env create -f environment.yml
  ```
`CTMAP` is tested on GPU. The versions of torch, torchvision, torchaudio should be compatible with your CUDA version.

## Installation
* You can install `CTMAP` via:
  ```
  git clone https://github.com/yourusername/CTMAP-main.git

  cd CTMAP-main
  ```
## Tutorials
The following are detailed tutorials. All tutotials were ran on GPU.
1. [Full CTMAP pipeline on MERFISH dataset](/tutorial.ipynb).
