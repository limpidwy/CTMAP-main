# CTMAP: Inferring gene regulatory networks from single-cell RNA sequencing data by dual-role graph contrastive learning
---
## Introduction
---
`CTMAP` is a deep learning framework for accurate cell type annotation of single-cell resolution spatial transcriptomics data by integrating labeled scRNA-seq reference with domain adversarial alignment in a shared latent space.
<div align="center">
  <img src="./overview.jpg" alt="Overview" width="80%">
</div>


## Installation
---

### Requirements
---
`CTMAP` depends on the following Python packages:
```
numpy==1.26.4
pandas==2.2.3
scanpy==1.10.3
torch==2.4.1
tqdm==4.64.0
scikit-learn==1.5.2
scipy==1.13.1
```
For better performance, we recommend running CTMAP on an NVIDIA GPU with CUDA support.
### Create a new conda environment
```
conda create -n ctmap python=3.9
conda activate ctmap
```
### Install using pip
```
pip install git+http://github.com/SDU-Math-SunLab/CTMAP.git
```

## Usage example
---
### CTMAP Inputs
- scRNA-seq reference: An `.h5ad` file containing an AnnData object with cell type annotations in `.obs['cell_type']`.
- Spatial transcriptomics data: An `.h5ad` file containing an AnnData object with spatial coordinates in `.obs['X']` and `.obs['Y']`. Ground-truth cell types in `.obs['cell_type']` are optional (used only for evaluation).


### Package usage
Clone the repository and navigate to the project root:
```
git clone https://github.com/yourusername/CTMAP-main.git
cd CTMAP-main
```
Run CTMAP on the provided example dataset:
```
python run.py
```
## Reference
Qiyuan Guan, Jiating Yu, Jieyi Pan, Fan Yuan, Jiadong Ji, Rusong Zhao, Zhi-Ping Liu, Bingqiang Liu, Ling-Yun Wu, and Duanchen Sun. "Inferring gene regulatory networks from single-cell RNA sequencing data by dual-role graph contrastive learning". *Advanced Science*, 2025.<br>
https://doi.org/10.1002/advs.202518277
