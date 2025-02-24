import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from delius.modules.embeddings_dataset import EmbeddingsDataset
from delius.modules.autoencoder import EmbeddingsAutoencoder


def pretrain_autoencoder(
    input_embeddings_file_path,
    output_encoder_file_path,
    input_embeddings_dimensions=1024,
    encoder_hidden_dimensions=[500, 500, 2000, 10],
    batch_size=256,
    epochs=200,
    learning_rate=1e-3
):
    tqdm.write(f"Loading features from '{input_embeddings_file_path}'...")

    dataset = EmbeddingsDataset(input_embeddings_file_path)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
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
        
        for _, features in tqdm(loader, desc='Batches', leave=False):
            features = features.to(device)
            optimizer.zero_grad()
            
            _, reconstrutions = model(features)
            
            loss = mse_loss(reconstrutions, features)
            loss.backward()
            
            optimizer.step()
            
            total_loss += loss.item() * features.size(0)
        
        tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader.dataset):.4f}")

    tqdm.write(f"Saving encoder to '{output_encoder_file_path}'...")
    torch.save(model.encoder.state_dict(), output_encoder_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_embeddings_file', type=str)
    parser.add_argument('--output_encoder_file', type=str)
    parser.add_argument('--input_embeddings_dimensions', type=int)
    parser.add_argument('--encoder_hidden_dimensions', type=int, nargs='+')
    parser.add_argument('--batch', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning_rate', type=float)

    args = parser.parse_args()

    pretrain_autoencoder(
        args.input_embeddings_file,
        args.output_encoder_file,
        args.input_embeddings_dimensions,
        args.encoder_hidden_dimensions,
        args.batch,
        args.epochs,
        args.learning_rate
    )