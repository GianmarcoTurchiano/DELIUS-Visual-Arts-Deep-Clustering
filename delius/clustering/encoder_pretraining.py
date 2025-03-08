from tqdm.autonotebook import tqdm
from typing import Callable

import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from delius.clustering.modules.features_dataset import FeaturesDataset
from delius.clustering.modules.features_autoencoder import FeaturesAutoencoder


def pretrain_encoder(
    dataset: FeaturesDataset,
    encoder_hidden_dimensions: list[int]=[500, 500, 2000, 10],
    batch_size=256,
    epochs=200,
    learning_rate=1e-3,
    seed=42,
    log_epoch_loss_fn: Callable[[int, float], None] = None
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FeaturesAutoencoder(
        dataset.features_dims,
        encoder_hidden_dimensions
    ).to(device)

    model.train()

    optimizer = Adam(model.parameters(), lr=learning_rate)

    mse_loss = nn.MSELoss()

    for epoch in tqdm(range(1, epochs + 1), desc='Epochs'):
        total_loss = 0.0

        for _, _, features in tqdm(loader, desc='Batches', leave=False):
            features = features.to(device)
            optimizer.zero_grad()

            _, reconstrutions = model(features)

            loss = mse_loss(reconstrutions, features)
            loss.backward()

            optimizer.step()

            total_loss += loss.item() * features.size(0)

        avg_loss = total_loss / len(loader.dataset)
        tqdm.write(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

        if log_epoch_loss_fn != None:
            log_epoch_loss_fn(epoch, avg_loss)

    return model.encoder
