import torch.nn as nn
import torch


class FeaturesEncoder(nn.Module):
    def __init__(
        self,
        input_embeddings_dimensions=1024,
        hidden_dims: list[int]=[500, 500, 2000, 10]
    ):
        super(FeaturesEncoder, self).__init__()

        encoder_layers = []

        prev_dim = input_embeddings_dimensions

        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers[:-1])

    def forward(self, x):
        return self.encoder(x)


def load_features_encoder(
    file_path: str,
    input_embeddings_dimensions=1024,
    hidden_dims: list[int]=[500, 500, 2000, 10],
):
    model = FeaturesEncoder(
        input_embeddings_dimensions,
        hidden_dims
    )

    weights = torch.load(file_path, weights_only=True)

    model.load_state_dict(weights)

    return model