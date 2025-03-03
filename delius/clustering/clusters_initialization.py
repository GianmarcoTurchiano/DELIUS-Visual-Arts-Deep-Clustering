from tqdm.autonotebook import tqdm

import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader

from delius.clustering.modules.features_encoder import FeaturesEncoder
from delius.clustering.modules.features_dataset import FeaturesDataset


def initialize_clusters(
    dataset: FeaturesDataset,
    model: FeaturesEncoder,
    n_clusters=10,
    batch_size=256,
    seed=42
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    model.eval()
    bottlenecks = []

    with torch.no_grad():
        for _, _, features in tqdm(loader, desc='Computing bottleneck features'):
            features = features.to(device)
            bottleneck = model(features)
            bottlenecks.append(bottleneck.cpu().numpy())

    bottlenecks = np.concatenate(bottlenecks, axis=0)

    tqdm.write(f"Clustering bottleneck features into {n_clusters} clusters...")

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    assignments = kmeans.fit_predict(bottlenecks)

    centroids = torch.from_numpy(kmeans.cluster_centers_).float() 
    assignments = torch.tensor(assignments, dtype=torch.long)

    return centroids, assignments
