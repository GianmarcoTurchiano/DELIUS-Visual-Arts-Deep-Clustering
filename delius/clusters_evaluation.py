import argparse
from tqdm.autonotebook import tqdm
import random

import pandas as pd
import numpy as np
import torch
from dotenv import load_dotenv
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_completeness_v_measure,
    fowlkes_mallows_score
)

from delius.clustering.modules.features_dataset import load_features_dataset
from delius.clustering.modules.deep_embedded_clustering import (
    DEC,
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

    parser.add_argument('--input_dec_mlflow_run_id_file', type=str)
    parser.add_argument('--input_embeddings_file', type=str)
    parser.add_argument('--input_labels_file', type=str)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--input_image_dir', type=str)
    parser.add_argument('--n_samples_per_cluster', type=int)
    parser.add_argument('--embeddings_sample_fraction', type=float)
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tqdm.write(f"Loading environment variables...")
    load_dotenv()
    tqdm.write(f"Done.")

    tqdm.write(f"Loading features from '{args.input_embeddings_file}'...")
    dataset = load_features_dataset(args.input_embeddings_file)
    tqdm.write(f"Done.")

    tqdm.write(f"Loading ground truths from '{args.input_labels_file}'...")
    labels_df = pd.read_csv(args.input_labels_file)
    labels_df = labels_df.set_index('subject')
    labels_df = pd.DataFrame({col: pd.factorize(labels_df[col])[0] for col in labels_df.columns}, index=labels_df.index)
    tqdm.write(f"Done.")

    with open(args.input_dec_mlflow_run_id_file, 'r') as file:
        parent_run_id = file.read()

    model_uri = f'runs:/{parent_run_id}/{DEC.__name__}'

    tqdm.write(f"Loading DEC from '{model_uri}'...")
    model = mlflow.pytorch.load_model(model_uri)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tqdm.write(f"Done.")

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

        tqdm.write(f"Retrieving ground truths...")
        ground_truths = labels_df.loc[names].values
        tqdm.write('Done')

        for i, column_name in enumerate(labels_df.columns):
            tqdm.write(f"Computing metrics for the {column_name} labels...")
            true_labels = ground_truths[:, i]

            ari = adjusted_rand_score(true_labels, assignments)
            mlflow.log_metric(f'Adjusted Rand Index ({column_name})', ari)

            nmi = normalized_mutual_info_score(true_labels, assignments)
            mlflow.log_metric(f'Normalized Mutual Information ({column_name})', nmi)

            ami = adjusted_mutual_info_score(true_labels, assignments)
            mlflow.log_metric(f'Adjusted Mutual Information ({column_name})', ami)

            homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(true_labels, assignments)
            mlflow.log_metric(f'Homogeneity ({column_name})', homogeneity)
            mlflow.log_metric(f'Completeness ({column_name})', completeness)
            mlflow.log_metric(f'V-Measure ({column_name})', v_measure)

            fmi = fowlkes_mallows_score(true_labels, assignments)
            mlflow.log_metric(f'Fowlkes-Mallows Index ({column_name})', fmi)

            tqdm.write('Done')
