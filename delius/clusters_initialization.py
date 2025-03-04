import argparse
from tqdm.autonotebook import tqdm

import torch

from delius.clustering.modules.features_encoder import load_features_encoder
from delius.clustering.modules.features_dataset import load_features_dataset
from delius.clustering.clusters_initialization import initialize_clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_embeddings_file', type=str)
    parser.add_argument('--input_pretrained_encoder_file', type=str)
    parser.add_argument('--output_centroids_file', type=str)
    parser.add_argument('--output_assignments_file', type=str)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--n_clusters', type=int)
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    tqdm.write(f"Loading features from '{args.input_embeddings_file}'...")

    dataset = load_features_dataset(args.input_embeddings_file)

    tqdm.write(f"Loading encoder from '{args.input_pretrained_encoder_file}'...")

    model = load_features_encoder(args.input_pretrained_encoder_file)

    centroids, assignments = initialize_clusters(
        dataset,
        model,
        args.n_clusters,
        args.batch,
        args.seed
    )

    tqdm.write(f"Saving centroids to '{args.output_centroids_file}'...")
    torch.save(centroids, args.output_centroids_file)

    tqdm.write(f"Saving cluster assignments to '{args.output_assignments_file}'...")
    torch.save(assignments, args.output_assignments_file)
