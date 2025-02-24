import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from delius.modules.DEC import DEC, target_distribution
from delius.modules.encoder import EmbeddingsEncoder
from delius.modules.embeddings_dataset import EmbeddingsDataset


def cluster(
    input_centroids_file_path,
    input_assignments_file_path,
    input_pretrained_encoder_file_path,
    input_embeddings_file_path,
    output_dec_file_path,
    input_embeddings_dimensions=1024,
    encoder_hidden_dimensions=[500, 500, 2000, 10],
    batch_size=256,
    learning_rate=1e-3,
    steps=8000,
    update_interval=140,
    delta_tol=0.001,
    n_clusters=5
):
    tqdm.write(f"Loading features from '{input_embeddings_file_path}'...")

    dataset = EmbeddingsDataset(input_embeddings_file_path)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    tqdm.write(f"Loading initial centroids from '{input_centroids_file_path}'...")
    
    centroids = torch.load(input_centroids_file_path)

    tqdm.write(f"Loading initial cluster assignments from '{input_assignments_file_path}'...")
    
    assignments = torch.load(input_assignments_file_path)

    tqdm.write(f"Loading encoder from '{input_pretrained_encoder_file_path}'...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = EmbeddingsEncoder(
        input_embeddings_dimensions,
        encoder_hidden_dimensions
    )

    weights = torch.load(input_pretrained_encoder_file_path)

    encoder.load_state_dict(weights)

    model = DEC(
        encoder,
        n_clusters,
        encoder_hidden_dimensions[-1]
    ).to(device)

    model.centroids = centroids

    optimizer = Adam(model.parameters(), lr=learning_rate)
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    y_pred_last = torch.clone(assignments).numpy()

    step = 1

    pbar = tqdm(total=steps)

    converged = False

    while True:
        for _, train_embeddings in train_loader:
            if step % update_interval == 0:
                q_all = []
                
                model.eval()
                with torch.no_grad():
                    for _, val_embeddings in tqdm(val_loader, leave=False, desc='Computing delta'):
                        val_embeddings = val_embeddings.to(device)
                        _, q = model(val_embeddings)
                        q_all.append(q.detach().cpu())
                
                q_all = torch.cat(q_all, dim=0)
                
                y_pred = torch.argmax(q_all, dim=1).numpy()

                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred.copy()
                
                tqdm.write(f"Step {step}/{steps}, label change: {delta_label:.4f}")
                
                if step > 0 and delta_label < delta_tol:
                    tqdm.write(f'delta_label {delta_label} < tol {delta_tol}')
                    tqdm.write('Reached tolerance threshold. Stopping training.')
                    converged = True
                    break
            
            model.train()
            train_embeddings = train_embeddings.to(device)
            optimizer.zero_grad()
            _, q = model(train_embeddings)
            p_batch = target_distribution(q.detach())
            loss = kl_loss(torch.log(q), p_batch)
            loss.backward()
            optimizer.step()

            if step % update_interval == 0:
                tqdm.write(f"Step {step}/{steps}, KL Loss: {loss.item():.4f}")

            step += 1
            pbar.update(1)

            if step >= steps:
                converged = True
                break

        if converged:
            break

    pbar.close()

    tqdm.write(f"Saving DEC to '{output_dec_file_path}'...")
    torch.save(model.state_dict(), output_dec_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_centroids_file', type=str)
    parser.add_argument('--input_assignments_file_path', type=str)
    parser.add_argument('--input_pretrained_encoder_file', type=str)
    parser.add_argument('--input_embeddings_file', type=str)
    parser.add_argument('--output_dec_file', type=str)
    parser.add_argument('--input_embeddings_dimensions', type=int)
    parser.add_argument('--encoder_hidden_dimensions', type=int, nargs='+')
    parser.add_argument('--batch', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--update_interval', type=int)
    parser.add_argument('--delta_tol', type=float)
    parser.add_argument('--n_clusters', type=int)

    args = parser.parse_args()

    cluster(
        args.input_centroids_file,
        args.input_assignments_file_path,
        args.input_pretrained_encoder_file,
        args.input_embeddings_file,
        args.output_dec_file,
        args.input_embeddings_dimensions,
        args.encoder_hidden_dimensions,
        args.batch,
        args.learning_rate,
        args.steps,
        args.update_interval,
        args.delta_tol,
        args.n_clusters
    )
