from collections import Counter
from scipy import sparse
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import MaxAbsScaler, maxabs_scale
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder


def batch_scale(adata, use_rep="X", chunk_size=20000):
    for b in adata.obs["source"].unique():
        idx = np.where(adata.obs["source"] == b)[0]
        if use_rep == "X":
            scaler = MaxAbsScaler(copy=False).fit(adata.X[idx])
            for i in range(len(idx) // chunk_size + 1):
                adata.X[idx[i * chunk_size : (i + 1) * chunk_size]] = scaler.transform(
                    adata.X[idx[i * chunk_size : (i + 1) * chunk_size]]
                )
        else:
            scaler = MaxAbsScaler(copy=False).fit(adata.obsm[use_rep][idx])
            for i in range(len(idx) // chunk_size + 1):
                adata.obsm[use_rep][idx[i * chunk_size : (i + 1) * chunk_size]] = scaler.transform(
                    adata.obsm[use_rep][idx[i * chunk_size : (i + 1) * chunk_size]] )


def cell_type_encoder(adata_rna, adata_spa):
    # 提取细胞类型
    rna_celltype = adata_rna.obs['cell_type'].values
    st_celltype = adata_spa.obs['cell_type'].values
    label_encoder_rna = LabelEncoder()
    rna_labels = label_encoder_rna.fit_transform(rna_celltype)
    rna_label_mapping = {label: i for i, label in enumerate(label_encoder_rna.classes_)}

    seqfish_labels = []
    new_label = len(rna_label_mapping)
    seqfish_label_mapping = {}
    for label in st_celltype:
        if label in rna_label_mapping:
            seqfish_labels.append(rna_label_mapping[label])
        else:
            if label not in seqfish_label_mapping:
                seqfish_label_mapping[label] = new_label
                new_label += 1
            seqfish_labels.append(seqfish_label_mapping[label])


    full_label_mapping = {**rna_label_mapping, **seqfish_label_mapping}
    inverse_full_label_mapping = {v: k for k, v in full_label_mapping.items()}
    seqfish_labels = np.array(seqfish_labels)
    cell_types = np.unique(np.concatenate((rna_labels, seqfish_labels)))
    adata_rna.obs['labels'] = rna_labels
    adata_spa.obs['labels'] = seqfish_labels
    print("Unique RNA Labels:", np.unique(rna_labels))
    print("Unique SeqFISH Labels:", np.unique(seqfish_labels))
    common_labels = np.intersect1d(rna_labels, seqfish_labels)
    print("common Labels:", common_labels)
    print("Number of classes:", len(cell_types))

    return rna_labels, seqfish_labels, cell_types

def anndata_preprocess(adata_spa, adata_rna):
    """
    Preprocess the rna and spatial AnnData

    :adata_spa: AnnData file of spatial dataset, .obs contains 'X','Y', 'source'
    :adata_rna: AnnData file of rna dataset, .obs contains 'cell_type', 'source'
    :spatial_labels: if there are ground truth spatial data labels, if True, adata_spa.obs should contains 'cell_type'
    :return: preprocessed AnnData file, adata_cm, adata_rna, adata_spa
    """

    # Step 1: Standardize spatial and RNA data independently
    sc.pp.normalize_total(adata_spa)
    sc.pp.log1p(adata_spa)
    batch_scale(adata_spa)

    sc.pp.normalize_total(adata_rna)
    sc.pp.log1p(adata_rna)
    batch_scale(adata_rna)
    adata_combined = adata_spa.concatenate(adata_rna, join='inner', batch_key="source")
    sc.pp.normalize_total(adata_combined)
    sc.pp.log1p(adata_combined)
    batch_scale(adata_combined)

    return adata_spa, adata_rna

class SingleCellDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.shape = data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if isinstance(self.data[idx], sparse.spmatrix):
            x = self.data[idx].toarray().squeeze()
        else:
            x = self.data[idx].squeeze()
        labels = self.labels[idx].squeeze()

        return x, labels, idx


def generate_dataloaders( adata_spa, adata_rna, batch_size=256):

    rna_labels = np.array(adata_rna.obs["labels"])
    spa_labels = np.array(adata_spa.obs["labels"])
    rna_dataset = SingleCellDataset(adata_rna.X, rna_labels)
    st_dataset = SingleCellDataset(adata_spa.X, spa_labels)

    classes = rna_labels
    freq = Counter(classes)
    class_weight = {x: 1.0 / freq[x] for x in freq}
    source_weights = [class_weight[x] for x in classes]
    sampler = WeightedRandomSampler(source_weights, len(rna_dataset.labels))

    rna_train_loader = DataLoader(
        dataset= rna_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, drop_last=True
    )
    st_train_loader = DataLoader(dataset=st_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    rna_test_loader = DataLoader(
        dataset=rna_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False
    )
    st_test_loader = DataLoader(
        dataset=st_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False
    )
    return  rna_train_loader, st_train_loader, rna_test_loader, st_test_loader




