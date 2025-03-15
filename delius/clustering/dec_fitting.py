from tqdm.autonotebook import tqdm
import sys
import os
from functools import wraps

import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from delius.clustering.modules.deep_embedded_clustering import DEC, target_distribution
from delius.clustering.modules.features_encoder import FeaturesEncoder
from delius.clustering.modules.features_dataset import FeaturesDataset


def metrics_logger(train_step_name):
    def decorator(step_training_fn):
        @wraps(step_training_fn)
        def wrapper(*args, **kwargs):
            step, steps, avg_loss, delta_label, converged = step_training_fn(*args, **kwargs)
            loss_name = 'KLDivloss'
            delta_name = 'delta'

            tqdm.write(f"{train_step_name} {step}/{steps}, {loss_name}: {avg_loss:.4f}")
            tqdm.write(f"{train_step_name} {step}/{steps}, {delta_name}: {delta_label:.4f}")

            mlflow = sys.modules.get("mlflow") 
            if mlflow:  # Log only if mlflow was imported
                mlflow.log_metric(loss_name, avg_loss, step=step)
                mlflow.log_metric(delta_name, delta_label, step=step)
            
            if converged:
                tqdm.write(f'Reached {delta_name} tolerance threshold. Stopping training.')

            return converged
                
        return wrapper
    return decorator


@metrics_logger(train_step_name="Step")
def _check_clustering_metrics(
    y_pred_last: np.ndarray,
    y_pred: np.ndarray,
    loss_total: float,
    delta_tol: float,
    update_interval: int,
    step: int,
    steps: int
):
    delta = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]

    avg_loss = loss_total / update_interval

    converged = delta < delta_tol

    return step, steps, avg_loss, delta, converged


def fit_dec(
    encoder: FeaturesEncoder,
    dataset: FeaturesDataset,
    centroids: torch.Tensor,
    assignments: torch.Tensor,
    batch_size=256,
    learning_rate=1e-3,
    steps=8000,
    update_interval=140,
    delta_tol=0.001,
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

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

    n_clusters, bottleneck_dimensions = centroids.shape

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

    pbar = tqdm(total=steps, desc='Steps')

    converged = False

    while True:
        for idx, _, train_embeddings in train_loader:
            if step % update_interval == 0:
                q_all = []
                p_all = []
                
                model.eval()
                with torch.no_grad():
                    for _, _, val_embeddings in tqdm(val_loader, leave=False, desc='Computing assignments'):
                        val_embeddings = val_embeddings.to(device)
                        _, q = model(val_embeddings)
                        q_all.append(q.detach().cpu())

                q_all = torch.cat(q_all, dim=0)
                p_all = target_distribution(q_all)

                if step > 0:
                    y_pred = torch.argmax(q_all, dim=1).numpy()

                    converged = _check_clustering_metrics(
                        y_pred_last, y_pred, loss_total,
                        delta_tol, update_interval, step, steps
                    )

                    if converged:
                        break

                    y_pred_last = y_pred.copy()
                    loss_total = 0

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

            step += 1
            pbar.update(1)

            if step >= steps:
                converged = True
                break

        if converged:
            break

    pbar.close()

    return model
