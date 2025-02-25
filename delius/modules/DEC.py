import torch
from torch import nn


import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, input_dim, alpha=1.0, weights=None):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.empty(n_clusters, input_dim))
        nn.init.xavier_uniform_(self.clusters)  # Equivalent to 'glorot_uniform'
        
        if weights is not None:
            self.clusters.data = torch.tensor(weights, dtype=torch.float32)

    def forward(self, inputs):
        # Compute pairwise squared Euclidean distances ||z_i - c_j||^2
        distances = torch.sum((inputs.unsqueeze(1) - self.clusters) ** 2, dim=2)

        # Compute soft assignment (q_ij)
        q = 1.0 / (1.0 + distances / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)  # Normalize over clusters

        return q


class DEC(nn.Module):
    def __init__(self, encoder, n_clusters, embeddings_dim, alpha=1.0):
        super(DEC, self).__init__()
        self.encoder = encoder
        self.clustering_layer = ClusteringLayer(n_clusters, embeddings_dim, alpha)

    def forward(self, x):
        z = self.encoder(x)  # Get latent representation
        q = self.clustering_layer(z)  # Apply clustering
        return z, q

    
def target_distribution(q):
    weight = q ** 2 / torch.sum(q, dim=0)
    p = (weight.t() / torch.sum(weight, dim=1)).t()
    return p