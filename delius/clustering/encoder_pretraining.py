from tqdm.autonotebook import tqdm
import os
import sys
from functools import wraps

import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from delius.clustering.modules.features_dataset import FeaturesDataset
from delius.clustering.modules.features_autoencoder import FeaturesAutoencoder


def loss_logger(train_step_name):
    def decorator(step_training_fn):
        @wraps(step_training_fn)
        def wrapper(*args, **kwargs):
            step, steps, loss_fn, avg_loss = step_training_fn(*args, **kwargs)
            loss_name = loss_fn._get_name()
            
            tqdm.write(f"{train_step_name} {step}/{steps}, {loss_name}: {avg_loss:.4f}")

            mlflow = sys.modules.get("mlflow") 
            if mlflow:  # Log only if mlflow was imported
                mlflow.log_metric(loss_name, avg_loss, step=step)
                
        return wrapper
    return decorator


@loss_logger(train_step_name="Epoch")
def _train_epoch(
    model: FeaturesAutoencoder,
    loader: DataLoader,
    optimizer: Adam,
    mse_loss: nn.MSELoss,
    device: torch.device,
    epoch: int,
    epochs: int
):
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

    return epoch, epochs, mse_loss, avg_loss


def pretrain_encoder(
    dataset: FeaturesDataset,
    encoder_hidden_dimensions: list[int]=[500, 500, 2000, 10],
    batch_size=256,
    epochs=200,
    learning_rate=1e-3,
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

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
        _train_epoch(model, loader, optimizer, mse_loss, device, epoch, epochs)

    return model.encoder
