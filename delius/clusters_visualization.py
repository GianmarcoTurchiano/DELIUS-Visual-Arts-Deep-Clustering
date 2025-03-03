import argparse
from tqdm.autonotebook import tqdm

import torch
import matplotlib.pyplot as plt

from delius.clustering.modules.features_dataset import load_features_dataset
from delius.clustering.modules.deep_embedded_clustering import DEC
from delius.clustering.modules.features_encoder import FeaturesEncoder
from delius.clustering.clusters_visualization import (
    compute_embeddings_and_assignments,
    sample_clustered_embeddings,
    plot_2D_clusters,
    sample_n_files_per_cluster,
    composite_clustered_pics
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dec_file', type=str)
    parser.add_argument('--input_embeddings_file', type=str)
    parser.add_argument('--output_tsne_pic_file', type=str)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--input_embeddings_dimensions', type=int)
    parser.add_argument('--encoder_hidden_dimensions', type=int, nargs='+')
    parser.add_argument('--n_clusters', type=int)
    parser.add_argument('--input_image_dir', type=str)
    parser.add_argument('--output_clustered_samples_file', type=str)
    parser.add_argument('--n_samples_per_cluster', type=int)
    parser.add_argument('--embeddings_sample_fraction', type=float)
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    tqdm.write(f"Loading features from '{args.input_embeddings_file}'...")

    dataset = load_features_dataset(args.input_embeddings_file)

    tqdm.write(f"Loading DEC from '{args.input_dec_file}'...")

    weights = torch.load(args.input_dec_file, weights_only=True)

    encoder = FeaturesEncoder(
        args.input_embeddings_dimensions,
        args.encoder_hidden_dimensions
    )

    model = DEC(
        encoder,
        args.n_clusters,
        args.encoder_hidden_dimensions[-1]
    )

    model.load_state_dict(weights)

    names, embeddings, assignments = compute_embeddings_and_assignments(
        model,
        dataset,
        args.batch
    )

    sampled_names, sampled_embeddings, sampled_assignments = sample_clustered_embeddings(
        names,
        embeddings,
        assignments,
        args.embeddings_sample_fraction,
        args.seed
    )

    fig = plot_2D_clusters(
        sampled_embeddings,
        sampled_assignments,
        args.seed
    )

    tqdm.write(f"Saving image to '{args.output_tsne_pic_file}'...")

    fig.savefig(args.output_tsne_pic_file, bbox_inches='tight')
    plt.close(fig)

    cluster_images = sample_n_files_per_cluster(
        sampled_names,
        sampled_assignments,
        args.input_image_dir,
        args.n_clusters,
        args.n_samples_per_cluster
    )

    composite_img = composite_clustered_pics(
        cluster_images
    )

    tqdm.write(f"Saving image to '{args.output_clustered_samples_file}'...")

    composite_img.save(args.output_clustered_samples_file)
