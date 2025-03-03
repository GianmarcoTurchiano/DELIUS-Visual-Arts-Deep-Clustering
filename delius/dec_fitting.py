import argparse
from tqdm.autonotebook import tqdm

import torch

from delius.clustering.modules.features_encoder import load_features_encoder
from delius.clustering.modules.features_dataset import load_features_dataset
from delius.clustering.dec_fitting import fit_dec


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
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    tqdm.write(f"Loading features from '{args.input_embeddings_file}'...")

    dataset = load_features_dataset(args.input_embeddings_file)

    tqdm.write(f"Loading initial centroids from '{args.input_centroids_file}'...")
    
    centroids = torch.load(args.input_centroids_file, weights_only=True)

    tqdm.write(f"Loading initial cluster assignments from '{args.input_assignments_file}'...")
    
    assignments = torch.load(args.input_assignments_file, weights_only=True)

    tqdm.write(f"Loading encoder from '{args.input_pretrained_encoder_file}'...")

    encoder = load_features_encoder(
        args.input_pretrained_encoder_file,
        args.input_embeddings_dimensions,
        args.encoder_hidden_dimensions
    )

    model = fit_dec(
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
        args.n_clusters,
        args.seed
    )

    tqdm.write(f"Saving DEC to '{args.output_dec_file}'...")
    torch.save(model.state_dict(), args.output_dec_file)
