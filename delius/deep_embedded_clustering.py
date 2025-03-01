import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from delius.modules.DEC import DEC, target_distribution
from delius.modules.encoder import EmbeddingsEncoder, load_embeddings_encoder
from delius.modules.embeddings_dataset import EmbeddingsDataset, load_embeddings_dataset


def deep_embedded_clustering(
    encoder: EmbeddingsEncoder,
    dataset: EmbeddingsDataset,
    centroids: torch.Tensor,
    assignments: torch.Tensor,
    bottleneck_dimensions=10,
    batch_size=256,
    learning_rate=1e-3,
    steps=8000,
    update_interval=140,
    delta_tol=0.001,
    n_clusters=5
):
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = DEC(
        encoder,
        n_clusters,
        bottleneck_dimensions
    )

    model.clustering_layer.clusters.data = centroids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    y_pred_last = torch.clone(assignments).numpy()

    step = 0
    loss_total = 0

    pbar = tqdm(total=steps)

    converged = False

    while True:
        for idx, _, train_embeddings in train_loader:
            if step % update_interval == 0:
                q_all = []
                p_all = []
                
                model.eval()
                with torch.no_grad():
                    for _, _, val_embeddings in tqdm(val_loader, leave=False, desc='Computing delta'):
                        val_embeddings = val_embeddings.to(device)
                        _, q = model(val_embeddings)
                        q_all.append(q.detach().cpu())
                
                q_all = torch.cat(q_all, dim=0)
                p_all = target_distribution(q_all)

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
            _, q_batch = model(train_embeddings)
            p_batch = p_all[idx]
            p_batch = p_batch.to(device)
            loss = kl_loss(torch.log(q_batch), p_batch)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

            if step % update_interval == 0:
                loss_avg = loss_total / update_interval
                tqdm.write(f"Step {step}/{steps}, KL Loss: {loss_avg:.4f}")
                loss_total = 0

            step += 1
            pbar.update(1)

            if step >= steps:
                converged = True
                break

        if converged:
            break

    pbar.close()

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_centroids_file', type=str)
    parser.add_argument('--input_assignments_file', type=str)
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

    tqdm.write(f"Loading features from '{args.input_embeddings_file}'...")

    dataset = load_embeddings_dataset(args.input_embeddings_file)

    tqdm.write(f"Loading initial centroids from '{args.input_centroids_file}'...")
    
    centroids = torch.load(args.input_centroids_file, weights_only=True)

    tqdm.write(f"Loading initial cluster assignments from '{args.input_assignments_file}'...")
    
    assignments = torch.load(args.input_assignments_file, weights_only=True)

    tqdm.write(f"Loading encoder from '{args.input_pretrained_encoder_file}'...")

    encoder = load_embeddings_encoder(
        args.input_pretrained_encoder_file,
        args.input_embeddings_dimensions,
        args.encoder_hidden_dimensions
    )

    model = deep_embedded_clustering(
        encoder,
        dataset,
        centroids,
        assignments,
        args.encoder_hidden_dimensions[-1],
        args.batch,
        args.learning_rate,
        args.steps,
        args.update_interval,
        args.delta_tol,
        args.n_clusters
    )

    tqdm.write(f"Saving DEC to '{args.output_dec_file}'...")
    torch.save(model.state_dict(), args.output_dec_file)
