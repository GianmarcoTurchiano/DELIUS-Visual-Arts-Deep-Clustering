import torch.nn as nn
from delius.modules.features_encoder import FeaturesEncoder


class FeaturesAutoencoder(nn.Module):
    def __init__(
        self,
        input_embeddings_dimension=1024,
        hidden_dims=[500, 500, 2000, 10]
    ):
        super(FeaturesAutoencoder, self).__init__()

        self.encoder = FeaturesEncoder(input_embeddings_dimension, hidden_dims)
        
        decoder_layers = []

        prev_dim = hidden_dims[-1]

        for h_dim in reversed(hidden_dims[:-1]):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim

        decoder_layers.append(nn.Linear(prev_dim, input_embeddings_dimension))

        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoding = self.encoder(x)
        reconstruction = self.decoder(encoding)

        return encoding, reconstruction
