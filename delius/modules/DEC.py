import torch
from torch import nn


import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters=5, input_dim=10, alpha=1.0, weights=None):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.empty(n_clusters, input_dim))
        nn.init.xavier_uniform_(self.clusters)
        
        if weights is not None:
            self.clusters.data = torch.tensor(weights, dtype=torch.float32)

    def forward(self, inputs):
        distances = torch.sum((inputs.unsqueeze(1) - self.clusters) ** 2, dim=2)

        q = 1.0 / (1.0 + distances / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)

        return q


class DEC(nn.Module):
    def __init__(self, encoder, n_clusters=5, embeddings_dim=10, alpha=1.0):
        super(DEC, self).__init__()
        self.encoder = encoder
        self.clustering_layer = ClusteringLayer(n_clusters, embeddings_dim, alpha)

    def forward(self, x):
        z = self.encoder(x)
        q = self.clustering_layer(z)
        return z, q

    
def target_distribution(q):
    weight = q ** 2 / torch.sum(q, dim=0)
    p = (weight.t() / torch.sum(weight, dim=1)).t()
    return p
