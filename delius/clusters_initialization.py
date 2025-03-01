import argparse
from tqdm import tqdm

import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader

from delius.modules.encoder import EmbeddingsEncoder, load_embeddings_encoder
from delius.modules.embeddings_dataset import EmbeddingsDataset, load_embeddings_dataset


def init_clusters(
    dataset: EmbeddingsDataset,
    model: EmbeddingsEncoder,
    n_clusters=5,
    batch_size=256,
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

    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    assignments = kmeans.fit_predict(bottlenecks)

    centroids = torch.from_numpy(kmeans.cluster_centers_).float() 
    assignments = torch.tensor(assignments, dtype=torch.long)

    return centroids, assignments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_embeddings_file', type=str)
    parser.add_argument('--input_pretrained_encoder_file', type=str)
    parser.add_argument('--output_centroids_file', type=str)
    parser.add_argument('--output_assignments_file', type=str)
    parser.add_argument('--input_embeddings_dimensions', type=int)
    parser.add_argument('--encoder_hidden_dimensions', type=int, nargs='+')
    parser.add_argument('--batch', type=int)
    parser.add_argument('--n_clusters', type=int)

    args = parser.parse_args()

    tqdm.write(f"Loading features from '{args.input_embeddings_file}'...")

    dataset = load_embeddings_dataset(args.input_embeddings_file)

    tqdm.write(f"Loading encoder from '{args.input_pretrained_encoder_file}'...")

    model = load_embeddings_encoder(
        args.input_pretrained_encoder_file,
        args.input_embeddings_dimensions,
        args.encoder_hidden_dimensions
    )

    centroids, assignments = init_clusters(
        dataset,
        model,
        args.n_clusters,
        args.batch
    )

    tqdm.write(f"Saving centroids to '{args.output_centroids_file}'...")
    torch.save(centroids, args.output_centroids_file)

    tqdm.write(f"Saving cluster assignments to '{args.output_assignments_file}'...")
    torch.save(assignments, args.output_assignments_file)
