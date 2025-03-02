import argparse
from tqdm import tqdm

import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from delius.clustering.modules.deep_embedded_clustering import DEC, target_distribution
from delius.clustering.modules.features_encoder import FeaturesEncoder
from delius.clustering.modules.features_dataset import FeaturesDataset


def fit_dec(
    encoder: FeaturesEncoder,
    dataset: FeaturesDataset,
    centroids: torch.Tensor,
    assignments: torch.Tensor,
    bottleneck_dimensions=10,
    batch_size=256,
    learning_rate=1e-3,
    steps=8000,
    update_interval=140,
    delta_tol=0.001,
    n_clusters=10,
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
