import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Some cells have zero counts")
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from dataprocess import cell_type_encoder, anndata_preprocess, generate_dataloaders
from model import CTMAP

seed = 0
print(f"\n{'='*70}")
print(f"CTMAP - MERFISH | Single Run with Seed {seed}")
print(f"{'='*70}")

torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

adata_r = sc.read('/opt/data/private/ywang/CTMAP-main/dataset/MERFISH/adata_rna.h5ad')
adata_s = sc.read('/opt/data/private/ywang/CTMAP-main/dataset/MERFISH/adata_merfish.h5ad')

common_genes = adata_r.var_names.intersection(adata_s.var_names)
adata_rna = adata_r[:, common_genes].copy()
adata_spa = adata_s[:, common_genes].copy()

adata_rna.obs['source'] = 'RNA'
adata_spa.obs['source'] = ' MERFISH'
adata_rna.X = adata_rna.X.astype(np.float32)
adata_spa.X = adata_spa.X.astype(np.float32)

# 修改：接收第4个返回值 label_to_name（整数 → 细胞类型字符串字典）
_, _, cell_types, label_to_name = cell_type_encoder(adata_rna, adata_spa)

adata_spa, adata_rna = anndata_preprocess(adata_spa, adata_rna)

rna_train_loader, st_train_loader, rna_test_loader, st_test_loader = generate_dataloaders(adata_spa, adata_rna)

r_dim = len(adata_rna)
s_dim = len(adata_spa)
rna_dim = adata_rna.shape[1]
st_dim = adata_spa.shape[1]

model = CTMAP(rna_dim,
              st_dim,
              latent_dim=64,
              hidden_dim=256,
              mha_heads_1=4,
              mha_dim_1=256,
              mha_dim_2=128,
              mha_heads_2=4,
              class_num=len(np.unique(adata_rna.obs['cell_type'])),
              device=device)

truth_label, pred_label, truth_rna, rna_embeddings, st_embeddings = (
    model.train(r_dim, s_dim, rna_train_loader, st_train_loader, adata_spa.obs[["X", "Y"]],
                rna_test_loader, st_test_loader,
                lr=5e-4, maxiter=4000, miditer1=3000, log_interval=100,
                stage1_recon_weight=3.0, stage1_cls_weight=0.01,
                stage2_recon_weight=4.0, stage2_domain_weight=0.1, stage2_cls_weight=0.01)
)

# ===== 正确评估：使用 label_to_name 转字符串 =====
truth_ct = [label_to_name[i] for i in truth_label]
pred_ct = [label_to_name[i] for i in pred_label]

accuracy = accuracy_score(truth_ct, pred_ct)
nmi = normalized_mutual_info_score(truth_ct, pred_ct)
ari = adjusted_rand_score(truth_ct, pred_ct)

print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"Accuracy: {accuracy:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}")
print(f"{'='*70}")

