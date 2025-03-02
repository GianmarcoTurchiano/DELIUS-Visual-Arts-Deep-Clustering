import argparse
from tqdm import tqdm
import os
from PIL import ImageDraw, ImageFont, Image

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from delius.modules.embeddings_dataset import EmbeddingsDataset, load_embeddings_dataset
from delius.modules.DEC import DEC
from delius.modules.encoder import EmbeddingsEncoder


def compute_embeddings_and_assignments(
    model: DEC,
    dataset: EmbeddingsDataset,
    batch_size=256
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    model.eval()
    embeddings = []
    cluster_assignments = []
    names_all = []

    with torch.no_grad():
        for _, names, features in  tqdm(loader, desc='Computing embeddings and cluster assignments'):
            features = features.to(device)
            z, q = model(features)
            cluster_ids = torch.argmax(q, dim=1).cpu().numpy()
            embeddings.append(z.cpu().numpy())
            cluster_assignments.append(cluster_ids)
            names_all.extend(names)
 
    embeddings = np.concatenate(embeddings, axis=0)
    cluster_assignments = np.concatenate(cluster_assignments, axis=0)

    return names_all, embeddings, cluster_assignments


def plot_2D_clusters(
    embeddings: np.ndarray,
    cluster_assignments: np.ndarray,
    seed=42
):
    tqdm.write('Applying TSNE...')

    tsne = TSNE(n_components=2, random_state=seed)
    tsne_embedding = tsne.fit_transform(embeddings)

    # Create figure and plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=cluster_assignments, cmap='tab10', alpha=0.6)
    ax.set_title("t-SNE Visualization of Clusters")

    return fig


def sample_clustered_embeddings(
    names: list[str],
    embeddings: np.ndarray,
    assignments: np.ndarray,
    embeddings_sample_fraction=0.1,
    seed=42
):
    tqdm.write(f'Sampling {embeddings_sample_fraction * 100}% of the embeddings...')

    num_samples = int(len(embeddings) * embeddings_sample_fraction)

    rng = np.random.default_rng(seed=seed)
    indices = rng.choice(len(embeddings), num_samples, replace=False)

    sampled_embeddings = embeddings[indices]
    sampled_assignments = assignments[indices]
    sampled_names = [names[i] for i in indices]

    return sampled_names, sampled_embeddings, sampled_assignments


def sample_n_files_per_cluster(
    names: list[str],
    assignments: np.ndarray,
    image_dir_path: np.ndarray,
    n_clusters=10,
    n_samples_per_cluster=5,
):
    tqdm.write(f"Sampling {n_samples_per_cluster} pictures for each of the {n_clusters} clusters...")
    
    cluster_images = {i: [] for i in range(n_clusters)}
    cluster_counts = {i: 0 for i in range(n_clusters)}
    
    collected = 0
    total_needed = n_clusters * n_samples_per_cluster

    for img_name, cluster in zip(names, assignments):
        if cluster_counts[cluster] < n_samples_per_cluster:
            cluster_images[cluster].append(os.path.join(image_dir_path, img_name))
            cluster_counts[cluster] += 1
            collected += 1
            
            if collected >= total_needed:
                break

    return cluster_images


def composite_clustered_pics(
    cluster_images: dict[int, list[str]]
):
    img_size = (100, 100)
    rows = len(cluster_images.keys())
    cols = len(list(cluster_images.values())[0])

    composite_img = Image.new("RGB", (cols * img_size[0] + 100, rows * img_size[1]), (255, 255, 255))
    draw = ImageDraw.Draw(composite_img)

    # Font for label (choose the font path if needed)
    font = ImageFont.load_default()

    for row, (cluster, img_paths) in enumerate(cluster_images.items()):
        # Add the label on the left
        label = f"Cluster {cluster + 1}"
        draw.text((10, row * img_size[1] + img_size[1] // 2), label, font=font, fill=(0, 0, 0))

        for col, img_path in enumerate(img_paths):
            try:
                img = Image.open(img_path).resize(img_size)
                composite_img.paste(img, (col * img_size[0] + 100, row * img_size[1]))  # Shift images to the right
            except Exception as e:
                tqdm.write(f"Error loading image {img_path}: {e}")

    return composite_img


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

    dataset = load_embeddings_dataset(args.input_embeddings_file)

    tqdm.write(f"Loading DEC from '{args.input_dec_file}'...")

    weights = torch.load(args.input_dec_file, weights_only=True)

    encoder = EmbeddingsEncoder(
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
