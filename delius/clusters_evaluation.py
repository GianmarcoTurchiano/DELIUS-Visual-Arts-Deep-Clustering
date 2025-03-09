import argparse
from tqdm.autonotebook import tqdm
import os
import random

import numpy as np
import torch
from dotenv import load_dotenv
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from delius.clustering.modules.features_dataset import load_features_dataset
from delius.clustering.modules.deep_embedded_clustering import (
    load_dec,
    compute_embeddings_and_assignments
)
from delius.clustering.clusters_visualization import (
    sample_clustered_embeddings,
    plot_2D_clusters_tsne,
    plot_2D_clusters_umap,
    sample_n_files_per_cluster,
    composite_clustered_pics
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dec_file', type=str)
    parser.add_argument('--input_embeddings_file', type=str)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--input_image_dir', type=str)
    parser.add_argument('--n_samples_per_cluster', type=int)
    parser.add_argument('--embeddings_sample_fraction', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--env_file_path', type=str)
    parser.add_argument('--mlflow_parent_run_id_env_var_name', type=str)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tqdm.write(f"Loading features from '{args.input_embeddings_file}'...")
    dataset = load_features_dataset(args.input_embeddings_file)
    tqdm.write(f"Done.")

    tqdm.write(f"Loading DEC from '{args.input_dec_file}'...")
    model = load_dec(args.input_dec_file)
    tqdm.write(f"Done.")

    tqdm.write(f"Loading environment variables from '{args.env_file_path}'...")
    load_dotenv(args.env_file_path)
    tqdm.write(f"Done.")

    parent_run_id = os.environ[args.mlflow_parent_run_id_env_var_name]

    tqdm.write(f"Creating a new MLFlow child run (parent run id: {parent_run_id})...")

    with mlflow.start_run(
        parent_run_id=parent_run_id,
        nested=True,
        run_name='Clusters evaluation'
    ):
        tqdm.write(f"Done.")

        mlflow.log_param('t-SNE embeddings sample fraction', args.embeddings_sample_fraction)
        mlflow.log_param('samples per cluster', args.n_samples_per_cluster)
        mlflow.log_param('seed', args.seed)

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

        fig_tsne = plot_2D_clusters_tsne(
            sampled_embeddings,
            sampled_assignments,
            args.seed
        )

        tsne_pic_file_name = "tsne.png"

        tqdm.write(f"Logging t-SNE clustering scatter plot as '{tsne_pic_file_name}'...")
        mlflow.log_figure(fig_tsne, tsne_pic_file_name)
        plt.close(fig_tsne)
        tqdm.write("Done.")

        fig_umap = plot_2D_clusters_umap(
            sampled_embeddings,
            sampled_assignments,
            args.seed
        )

        umap_pic_file_name = "umap.png"

        tqdm.write(f"Logging UMAP clustering scatter plot as '{umap_pic_file_name}'...")
        mlflow.log_figure(fig_umap, umap_pic_file_name)
        plt.close(fig_umap)
        tqdm.write("Done.")

        cluster_images = sample_n_files_per_cluster(
            sampled_names,
            sampled_assignments,
            args.input_image_dir,
            args.n_samples_per_cluster
        )

        composite_img = composite_clustered_pics(cluster_images)

        samples_pic_file_name = "samples.png"

        tqdm.write(f"Logging sampled images as '{samples_pic_file_name}'...")
        mlflow.log_image(composite_img, samples_pic_file_name)
        tqdm.write("Done.")

        ch_score = calinski_harabasz_score(embeddings, assignments)
        tqdm.write(f'Calinski-Harabasz score: {ch_score}')
        mlflow.log_metric('Calinski-Harabasz score', ch_score)

        db_score = davies_bouldin_score(embeddings, assignments)
        tqdm.write(f'Davies-Bouldin score: {db_score}')
        mlflow.log_metric('Davies-Bouldin score', db_score)

        sil_score = silhouette_score(embeddings, assignments)
        tqdm.write(f'Silhouette score: {sil_score}')
        mlflow.log_metric('Silhouette score', sil_score)
