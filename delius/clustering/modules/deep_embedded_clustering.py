import torch
from torch import nn

from delius.clustering.modules.features_encoder import FeaturesEncoder

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters=10, input_dim=10, alpha=1.0, weights=None):
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
    def __init__(self, encoder, n_clusters=10, embeddings_dim=10, alpha=1.0):
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


def infer_dec_dimensions(state_dict):
    input_dim = None
    encoder_dims = []
    clustering_shape = None

    for key, value in state_dict.items():
        if "encoder" in key and "weight" in key:
            out_dim, in_dim = value.shape
            if input_dim is None:
                input_dim = in_dim
            encoder_dims.append(out_dim)

        if "clustering_layer.clusters" in key:
            clustering_shape = value.shape

    return input_dim, encoder_dims, clustering_shape


def load_dec(input_dec_file):
    weights = torch.load(input_dec_file, weights_only=True)

    input_dims, encoder_hidden_dims, [n_clusters, embeddings_dims] = infer_dec_dimensions(weights)

    encoder = FeaturesEncoder(
        input_dims,
        encoder_hidden_dims
    )

    model = DEC(
        encoder,
        n_clusters,
        embeddings_dims
    )

    model.load_state_dict(weights)

    return model