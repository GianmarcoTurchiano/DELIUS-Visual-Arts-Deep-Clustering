import torch.nn as nn


class FeaturesEncoder(nn.Module):
    def __init__(
        self,
        input_embeddings_dimensions,
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


def infer_encoder_dimensions(state_dict):
    hidden_dims = []
    input_dim = None

    for key in state_dict:
        if "weight" in key:
            out_dim, in_dim = state_dict[key].shape
            
            if input_dim is None:
                input_dim = in_dim
            
            hidden_dims.append(out_dim)

    return input_dim, hidden_dims


def load_features_encoder(weights):
    input_dims, hidden_dims = infer_encoder_dimensions(weights)

    model = FeaturesEncoder(
        input_dims,
        hidden_dims
    )

    model.load_state_dict(weights)

    return model