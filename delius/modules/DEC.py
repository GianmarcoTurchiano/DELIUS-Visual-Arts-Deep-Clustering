import torch
from torch import nn


class DEC(nn.Module):
    def __init__(
        self,
        encoder,
        centroids,
        alpha=1.0
    ):
        super(DEC, self).__init__()

        self.encoder = encoder
        self.centroids = nn.Parameter(centroids)
        self.alpha = alpha

    def forward(self, x):
        x = self.encoder(x)
        
        expanded_x = x.unsqueeze(1)  # Shape: (batch_size, 1, input_dim)
        distances = torch.sum((expanded_x - self.centroids) ** 2, dim=2)
        
        q = 1.0 / (1.0 + distances / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        
        return x, q
    
def target_distribution(q):
    weight = q ** 2 / torch.sum(q, dim=0)
    p = (weight.t() / torch.sum(weight, dim=1)).t()
    return p