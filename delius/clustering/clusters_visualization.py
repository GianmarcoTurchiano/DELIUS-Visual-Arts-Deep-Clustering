import os
from PIL import ImageDraw, ImageFont, Image

from tqdm.autonotebook import tqdm
import numpy as np
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt


def _plot_2D_clusters(
    embeddings: np.ndarray,
    cluster_assignments: np.ndarray,
):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(embeddings[:, 0], embeddings[:, 1], c=cluster_assignments, cmap='tab10', alpha=0.6)
    
    return fig

def plot_2D_clusters_tsne(
    embeddings: np.ndarray,
    cluster_assignments: np.ndarray,
    seed=42
):
    tqdm.write('Applying TSNE...')
    tsne_embeddings = TSNE(n_components=2, random_state=seed).fit_transform(embeddings)
    tqdm.write("Done.")

    fig = _plot_2D_clusters(tsne_embeddings, cluster_assignments)

    return fig


def plot_2D_clusters_umap(
    embeddings: np.ndarray,
    cluster_assignments: np.ndarray,
    seed=42
):
    tqdm.write('Applying UMAP...')
    tsne_embeddings = umap.UMAP(n_components=2, random_state=seed).fit_transform(embeddings)
    tqdm.write("Done.")

    fig = _plot_2D_clusters(tsne_embeddings, cluster_assignments)

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

    tqdm.write("Done.")

    return sampled_names, sampled_embeddings, sampled_assignments


def sample_n_files_per_cluster(
    names: list[str],
    assignments: np.ndarray,
    image_dir_path: np.ndarray,
    n_samples_per_cluster=5,
):
    (n_clusters,) = np.unique(assignments).shape

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

    tqdm.write("Done.")

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
