# CTMAP
---
## Introduction
---
<div align="center">
  <img src="./figures/overview.jpg" alt="CTMAP Overview" width="80%">
  <p><i>Overview of the CTMAP framework (please replace with your actual figure)</i></p>
</div>

CTMAP integrates scRNA-seq reference and spatial transcriptomics data (e.g., MERFISH, seqFISH+) through a shared latent space learned by a multi-head self-attention encoder. It employs modality-specific decoders for reconstruction, cell type classification on scRNA-seq, and domain adversarial training to align distributions between modalities. Final cell type labels are transferred to spatial spots via robust nearest-centroid matching with automatic outlier removal.

---
## Installation
---
### Requirements

CTMAP depends on the following Python packages:

