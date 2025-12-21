# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import cycle
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_grl, None

class SharedEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=512, heads=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(0)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(0)
        x = F.leaky_relu(self.fc2(x))
        return x


class RNADecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = torch.sigmoid(self.bn2(self.fc2(x)))
        return x

class STDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, heads=1):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, z):
        z = F.elu(self.bn1(self.fc1(z)))
        z = z.unsqueeze(0)
        z, _ = self.attention(z, z, z)
        z = z.squeeze(0)
        x_recon = torch.sigmoid(self.bn2(self.fc2(z)))
        return x_recon


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DomainClassifier(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc_new = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x, lambda_grl=1.0):
        x = GradientReversal.apply(x, lambda_grl)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc_new(x))
        x = self.fc2(x)
        return x


def calculate_centroid_auto_threshold(z_st, labels, n_neighbors=25, quantile=0.9):
    unique_labels = np.unique(labels)
    centroids = {}
    for label in unique_labels:
        embeddings = z_st[labels == label]
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)
        avg_distances = np.mean(distances, axis=1)
        distance_threshold = np.quantile(avg_distances, quantile)
        clustered_points = embeddings[avg_distances < distance_threshold]
        if len(clustered_points) > 0:
            centroids[label] = np.mean(clustered_points, axis=0)
        else:
            centroids[label] = np.nan
    return centroids


class CTMAP:
    """
        CTMAP model for integrating single-cell RNA-seq and spatial transcriptomics data.

        The model uses a shared encoder for both modalities, separate decoders for reconstruction,
        a classifier for cell type prediction on RNA data, and a domain classifier for adversarial
        alignment between RNA and spatial domains.

        :param rna_dim: Input dimension (number of genes) for RNA data
        :param st_dim: Input dimension (number of genes) for spatial transcriptomics data
        :param latent_dim: Dimension of the shared latent embedding space
        :param hidden_dim: Hidden dimension used in classifier and shared encoder
        :param mha_heads_1: Number of attention heads in the shared encoder
        :param mha_dim_1: Hidden dimension in the shared encoder's feed-forward layer
        :param mha_dim_2: Hidden dimension in the spatial decoder's attention layer
        :param mha_heads_2: Number of attention heads in the spatial decoder
        :param class_num: Number of cell types (classes) for classification
        :param device: Torch device to run the model on ('cuda' or 'cpu')
        """
    def __init__(self, rna_dim, st_dim, latent_dim, hidden_dim, mha_heads_1, mha_dim_1, mha_dim_2, mha_heads_2, class_num, device):
        self.class_num = class_num
        self.device = device
        self.encoder_1 = SharedEncoder(rna_dim, latent_dim, hidden_dim=mha_dim_1, heads=mha_heads_1).to(device)
        self.classifier = Classifier(latent_dim, hidden_dim, class_num).to(device)
        self.decoder_1 = RNADecoder(latent_dim, rna_dim).to(device)
        self.decoder_2 = STDecoder(latent_dim, mha_dim_2, st_dim, heads=mha_heads_2).to(device)
        self.domain_classifier = DomainClassifier(latent_dim).to(device)

    def train(self, rna_dim, spa_dim, rna_train_loader, st_train_loader, spatial_coor, rna_test_loader, st_test_loader,
              lr=5e-4, maxiter=4000, miditer1=3000, log_interval=100,
              stage1_recon_weight=3.0, stage1_cls_weight=0.01,
              stage2_recon_weight=4.0, stage2_domain_weight=0.1, stage2_cls_weight=0.01):

        """
                Train the CTMAP model in two stages:
                Stage 1: Pretraining with reconstruction and cell type classification losses.
                Stage 2: Domain adversarial training to align RNA and spatial embeddings.

                :param rna_dim: Number of cells in the RNA dataset
                :param spa_dim: Number of spots/cells in the spatial dataset
                :param rna_train_loader: DataLoader for RNA training data
                :param st_train_loader: DataLoader for spatial training data
                :param spatial_coor: Spatial coordinates (DataFrame with 'X' and 'Y' columns)
                :param rna_test_loader: DataLoader for RNA test/inference data
                :param st_test_loader: DataLoader for spatial test/inference data
                :param lr: Learning rate for Adam optimizer (default: 5e-4)
                :param maxiter: Total number of training iterations (default: 4000)
                :param miditer1: Iteration to switch from Stage 1 to Stage 2 (default: 3000)
                :param log_interval: Interval for logging loss values (default: 100)
                :param stage1_recon_weight: Weight for reconstruction loss in Stage 1 (default: 3.0)
                :param stage1_cls_weight: Weight for classification loss in Stage 1 (default: 0.01)
                :param stage2_recon_weight: Weight for reconstruction loss in Stage 2 (default: 4.0)
                :param stage2_domain_weight: Weight for domain adversarial loss in Stage 2 (default: 0.1)
                :param stage2_cls_weight: Weight for classification loss in Stage 2 (default: 0.01)

                :return:
                    - truth_label: Ground truth cell type labels for spatial data
                    - pred_label: Predicted cell type labels for spatial data
                    - truth_rna: Ground truth cell type labels for RNA data
                    - rna_embeddings: Latent embeddings of RNA cells
                    - st_embeddings: Latent embeddings of spatial spots/cells
                """

        optim_enc_1 = torch.optim.Adam(self.encoder_1.parameters(), lr=lr, weight_decay=5e-4)
        optim_dec_1 = torch.optim.Adam(self.decoder_1.parameters(), lr=lr, weight_decay=5e-4)
        optim_dec_2 = torch.optim.Adam(self.decoder_2.parameters(), lr=lr, weight_decay=5e-4)
        optim_cls = torch.optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=5e-4)
        optim_domain = torch.optim.Adam(self.domain_classifier.parameters(), lr=lr, weight_decay=5e-4)

        recon_crit = nn.MSELoss()
        criterion = nn.CrossEntropyLoss().to(self.device)

        print("=== Starting Stage 1: Pretraining (Reconstruction + Classification) ===")
        pbar = tqdm(total=maxiter, desc="Training", ncols=110)

        iteration = 0
        domain_loss = torch.tensor(0.0).to(self.device)

        while iteration < maxiter:
            if rna_dim > spa_dim:
                iters = zip(rna_train_loader, cycle(st_train_loader))
            else:
                iters = zip(cycle(rna_train_loader), st_train_loader)

            for minibatch_id, ((rna_data, rna_label, id_rna), (st_data, st_label, id_st)) in enumerate(iters):
                rna_label = rna_label.to(self.device)
                st_label = st_label.to(self.device)
                rna_data = rna_data.to(self.device)
                st_data = st_data.to(self.device)
                y_source = torch.ones(len(rna_data)).long().to(self.device)
                y_target = torch.zeros(len(st_data)).long().to(self.device)

                z_rna = self.encoder_1(rna_data)
                z_st = self.encoder_1(st_data)
                logits = self.classifier(z_rna)
                recon_source_c = self.decoder_1(z_rna)
                recon_target_c = self.decoder_2(z_st)

                recon_loss = recon_crit(recon_source_c, rna_data) + recon_crit(recon_target_c, st_data)
                cls_loss = criterion(logits, rna_label)

                if iteration <= miditer1:
                    loss = stage1_recon_weight * recon_loss + stage1_cls_weight * cls_loss

                    optim_enc_1.zero_grad()
                    optim_dec_1.zero_grad()
                    optim_dec_2.zero_grad()
                    optim_cls.zero_grad()
                    loss.backward()
                    optim_enc_1.step()
                    optim_dec_1.step()
                    optim_dec_2.step()
                    optim_cls.step()

                    domain_loss = torch.tensor(0.0).to(self.device)
                else:
                    lambda_grl = 0.6
                    domain_logits_source = self.domain_classifier(z_rna, lambda_grl)
                    domain_logits_target = self.domain_classifier(z_st, lambda_grl)
                    domain_loss = criterion(domain_logits_source, y_source) + criterion(domain_logits_target, y_target)

                    total_loss = stage2_recon_weight * recon_loss + stage2_domain_weight * domain_loss + stage2_cls_weight * cls_loss

                    optim_enc_1.zero_grad()
                    optim_dec_1.zero_grad()
                    optim_dec_2.zero_grad()
                    optim_cls.zero_grad()
                    optim_domain.zero_grad()
                    total_loss.backward()
                    optim_enc_1.step()
                    optim_dec_1.step()
                    optim_dec_2.step()
                    optim_cls.step()
                    optim_domain.step()

                iteration += 1
                pbar.update(1)

                if iteration % log_interval == 0:
                    print(f"#Iter {iteration}: recon_loss: {recon_loss.item():.6f}, cls loss: {cls_loss.item():.6f}, domain loss: {domain_loss.item():.6f}")

                if iteration == miditer1 + 1:
                    print("\n=== Stage 1 Completed ===\n=== Starting Stage 2: Domain Adversarial Training ===\n")

                if iteration == maxiter:
                    with torch.no_grad():
                        truth_rna = []
                        truth_label = []
                        rna_embeddings = []
                        st_embeddings = []
                        pred_label = []

                        for _, ((rna_data, rna_label, id_rna)) in enumerate(rna_test_loader):
                            rna_data = rna_data.to(self.device)
                            rna_label = rna_label.to(self.device)
                            z_rna = self.encoder_1(rna_data)
                            truth_rna.append(rna_label.cpu().numpy())
                            rna_embeddings.append(z_rna.cpu().numpy())

                        truth_rna = np.concatenate(truth_rna)
                        rna_embeddings = np.concatenate(rna_embeddings, axis=0)

                        rna_centroids_dict = calculate_centroid_auto_threshold(rna_embeddings, truth_rna)

                        unique_cell_types = []
                        cell_type_means = []
                        for cell_type, centroid in rna_centroids_dict.items():
                            if not np.isnan(centroid).any():
                                unique_cell_types.append(cell_type)
                                cell_type_means.append(centroid)

                        unique_cell_types = np.array(unique_cell_types)
                        cell_type_means = np.array(cell_type_means)

                        for _, ((st_data, st_label, id_st)) in enumerate(st_test_loader):
                            st_data = st_data.to(self.device)
                            id_st = id_st.to(self.device)
                            z_st = self.encoder_1(st_data)
                            truth_label.append(st_label.cpu().numpy())
                            st_embeddings.append(z_st.cpu().numpy())

                            distances = cdist(z_st.cpu().numpy(), cell_type_means, metric='euclidean')
                            closest_cell_type_indices = np.argmin(distances, axis=1)
                            pred_l = unique_cell_types[closest_cell_type_indices]
                            pred_label.append(pred_l)

                        truth_label = np.concatenate(truth_label)
                        st_embeddings = np.concatenate(st_embeddings, axis=0)
                        pred_label = np.concatenate(pred_label)

                        pbar.close()
                        return truth_label, pred_label, truth_rna, rna_embeddings, st_embeddings