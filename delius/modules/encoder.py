import torch.nn as nn


class EmbeddingsEncoder(nn.Module):
    def __init__(
        self,
        input_embeddings_dimension=1024,
        hidden_dims=[500, 500, 2000, 10]
    ):
        super(EmbeddingsEncoder, self).__init__()

        encoder_layers = []

        prev_dim = input_embeddings_dimension

        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers[:-1])

    def forward(self, x):
        return self.encoder(x)
