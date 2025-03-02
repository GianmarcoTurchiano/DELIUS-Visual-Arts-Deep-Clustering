import argparse
from tqdm import tqdm

import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from delius.modules.embeddings_dataset import EmbeddingsDataset, load_embeddings_dataset
from delius.modules.autoencoder import EmbeddingsAutoencoder


def pretrain_encoder(
    dataset: EmbeddingsDataset,
    input_embeddings_dimensions=1024,
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

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmbeddingsAutoencoder(
        input_embeddings_dimensions,
        encoder_hidden_dimensions
    ).to(device)

    model.train()

    optimizer = Adam(model.parameters(), lr=learning_rate)

    mse_loss = nn.MSELoss()

    for epoch in tqdm(range(epochs), desc='Epochs'):
        total_loss = 0.0
        
        for _, _, features in tqdm(loader, desc='Batches', leave=False):
            features = features.to(device)
            optimizer.zero_grad()
            
            _, reconstrutions = model(features)
            
            loss = mse_loss(reconstrutions, features)
            loss.backward()
            
            optimizer.step()
            
            total_loss += loss.item() * features.size(0)
        
        tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader.dataset):.4f}")

    return model.encoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_embeddings_file', type=str)
    parser.add_argument('--output_encoder_file', type=str)
    parser.add_argument('--input_embeddings_dimensions', type=int)
    parser.add_argument('--encoder_hidden_dimensions', type=int, nargs='+')
    parser.add_argument('--batch', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    tqdm.write(f"Loading features from '{args.input_embeddings_file}'...")
    dataset = load_embeddings_dataset(args.input_embeddings_file)

    model = pretrain_encoder(
        dataset,
        args.input_embeddings_dimensions,
        args.encoder_hidden_dimensions,
        args.batch,
        args.epochs,
        args.learning_rate,
        args.seed
    )

    tqdm.write(f"Saving encoder to '{args.output_encoder_file}'...")
    torch.save(model.state_dict(), args.output_encoder_file)
