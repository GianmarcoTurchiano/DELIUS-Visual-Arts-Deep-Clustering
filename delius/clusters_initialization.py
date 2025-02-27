import argparse
from tqdm import tqdm
import joblib

import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader

from delius.modules.encoder import EmbeddingsEncoder
from delius.modules.embeddings_dataset import EmbeddingsDataset


def init_clusters(
    input_embeddings_file_path,
    input_pretrained_encoder_file_path,
    output_centroids_file_path,
    output_assignments_file_path,
    n_clusters=5,
    input_embeddings_dimensions=1024,
    encoder_hidden_dimensions=[500, 500, 2000, 10],    
    batch_size=256,
):
    tqdm.write(f"Loading features from '{input_embeddings_file_path}'...")

    dataset = EmbeddingsDataset(input_embeddings_file_path)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tqdm.write(f"Loading encoder from '{input_pretrained_encoder_file_path}'...")

    model = EmbeddingsEncoder(
        input_embeddings_dimensions,
        encoder_hidden_dimensions
    ).to(device)

    weights = torch.load(input_pretrained_encoder_file_path)

    model.load_state_dict(weights)

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

    tqdm.write(f"Saving centroids to '{output_centroids_file_path}'...")

    centroids = torch.from_numpy(kmeans.cluster_centers_).float() 
    torch.save(centroids, output_centroids_file_path)

    tqdm.write(f"Saving cluster assignments to '{output_assignments_file_path}'...")

    assignments = torch.tensor(assignments, dtype=torch.long)
    torch.save(assignments, output_assignments_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_embeddings_file', type=str)
    parser.add_argument('--input_pretrained_encoder_file', type=str)
    parser.add_argument('--output_centroids_file', type=str)
    parser.add_argument('--output_assignments_file_path', type=str)
    parser.add_argument('--input_embeddings_dimensions', type=int)
    parser.add_argument('--encoder_hidden_dimensions', type=int, nargs='+')
    parser.add_argument('--batch', type=int)
    parser.add_argument('--n_clusters', type=int)

    args = parser.parse_args()

    init_clusters(
        args.input_embeddings_file,
        args.input_pretrained_encoder_file,
        args.output_centroids_file,
        args.output_assignments_file_path,
        args.n_clusters,
        args.input_embeddings_dimensions,
        args.encoder_hidden_dimensions,
        args.batch
    )
