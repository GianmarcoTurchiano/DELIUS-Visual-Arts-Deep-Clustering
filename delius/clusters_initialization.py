import argparse
from tqdm.autonotebook import tqdm
import os

import torch
from dotenv import load_dotenv
import mlflow

from delius.clustering.modules.features_encoder import FeaturesEncoder
from delius.clustering.modules.features_dataset import load_features_dataset
from delius.clustering.clusters_initialization import initialize_clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_embeddings_file', type=str)
    parser.add_argument('--input_encoder_mlflow_run_id_path', type=str)
    parser.add_argument('--output_centroids_file', type=str)
    parser.add_argument('--output_assignments_file', type=str)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--n_clusters', type=int)
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    tqdm.write(f"Loading environment variables...")
    load_dotenv()
    tqdm.write(f"Done.")

    tqdm.write(f"Loading features from '{args.input_embeddings_file}'...")

    dataset = load_features_dataset(args.input_embeddings_file)

    tqdm.write(f"Done.")

    with open(args.input_encoder_mlflow_run_id_path, 'r') as file:
        parent_run_id = file.read()

    model_uri = f'runs:/{parent_run_id}/{FeaturesEncoder.__name__}'

    tqdm.write(f"Loading encoder from '{model_uri}'...")
    model = mlflow.pytorch.load_model(model_uri)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tqdm.write(f"Done.")

    centroids, assignments = initialize_clusters(
        dataset,
        model,
        args.n_clusters,
        args.batch,
        args.seed
    )

    directory = os.path.dirname(args.output_centroids_file)

    if not os.path.exists(directory):
        os.makedirs(directory)
        tqdm.write(f"Created directory '{directory}'.")

    directory = os.path.dirname(args.output_assignments_file)

    if not os.path.exists(directory):
        os.makedirs(directory)
        tqdm.write(f"Created directory '{directory}'.")

    tqdm.write(f"Saving centroids to '{args.output_centroids_file}'...")
    torch.save(centroids, args.output_centroids_file)
    tqdm.write(f"Done.")

    tqdm.write(f"Saving cluster assignments to '{args.output_assignments_file}'...")
    torch.save(assignments, args.output_assignments_file)
    tqdm.write(f"Done.")
