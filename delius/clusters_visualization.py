import argparse
from tqdm import tqdm
import os
from PIL import ImageDraw, ImageFont, Image

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from delius.modules.embeddings_dataset import EmbeddingsDataset
from delius.modules.DEC import DEC
from delius.modules.encoder import EmbeddingsEncoder


def display_clusters(
    input_embeddings_file_path,
    input_dec_file_path,
    input_image_dir_path,
    output_tsne_pic_file_path,
    output_clustered_samples_file_path,
    n_samples_per_cluster = 5,
    batch_size=256,
    input_embeddings_dimensions=1024,
    encoder_hidden_dimensions=[500, 500, 2000, 10],
    n_clusters=5,
    embeddings_sample_fraction = 0.1
):
    tqdm.write(f"Loading features from '{input_embeddings_file_path}'...")

    dataset = EmbeddingsDataset(input_embeddings_file_path)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    tqdm.write(f"Loading DEC from '{input_dec_file_path}'...")

    weights = torch.load(input_dec_file_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = EmbeddingsEncoder(
        input_embeddings_dimensions,
        encoder_hidden_dimensions
    )

    model = DEC(
        encoder,
        n_clusters,
        encoder_hidden_dimensions[-1]
    ).to(device)

    model.load_state_dict(weights)

    model.eval()
    names_all = []
    embeddings_all = []
    q_all = []

    with torch.no_grad():
        for _, name, features in tqdm(loader, desc='Computing embeddings and cluster assignments'):
            names_all.extend(name)
            features = features.to(device)
            embeddings, q = model(features)
            q_all.append(q.detach().cpu())
            embeddings_all.append(embeddings.cpu())
    
    q_all = torch.cat(q_all, dim=0)
    embeddings_all = torch.cat(embeddings_all, dim=0)

    y_pred = torch.argmax(q_all, dim=1).numpy()

    tqdm.write(f'Sampling {embeddings_sample_fraction * 100}% of the embeddings...')

    num_samples = int(len(embeddings_all) * embeddings_sample_fraction)

    indices = np.random.choice(len(embeddings_all), num_samples, replace=False)
    sampled_embeddings = embeddings_all[indices]
    sampled_y_pred = y_pred[indices]
    sampled_names = [names_all[i] for i in indices]

    tqdm.write('Applying TSNE...')

    tsne = TSNE(n_components=2, random_state=42)
    tsne_embedding = tsne.fit_transform(sampled_embeddings)

    tqdm.write(f"Saving image to '{output_tsne_pic_file_path}'...")

    # Plot and save the result
    plt.figure(dpi=600)
    plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=sampled_y_pred, cmap='viridis')
    plt.savefig(output_tsne_pic_file_path, bbox_inches='tight')
    plt.close()

    tqdm.write(f"Sampling {n_samples_per_cluster} pictures for each of the {n_clusters}.")

    cluster_images = {i: [] for i in range(n_clusters)}
    for img_name, cluster in zip(sampled_names, sampled_y_pred):
        if len(cluster_images[cluster]) < 5:  # Limit to 5 per cluster
            cluster_images[cluster].append(os.path.join(input_image_dir_path, img_name))

    # Create a composite image
    img_size = (100, 100)  # Resize images for uniformity
    cols = n_samples_per_cluster  # 5 images per row
    rows = n_clusters  # One row per cluster

    composite_img = Image.new("RGB", (cols * img_size[0] + 100, rows * img_size[1]), (255, 255, 255))  # Extra space for labels
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

    tqdm.write(f"Saving image to '{output_clustered_samples_file_path}'...")

    # Save the composite image
    composite_img.save(output_clustered_samples_file_path)

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

    args = parser.parse_args()

    display_clusters(
        args.input_embeddings_file,
        args.input_dec_file,
        args.input_image_dir,
        args.output_tsne_pic_file,
        args.output_clustered_samples_file,
        args.n_samples_per_cluster,
        args.batch,
        args.input_embeddings_dimensions,
        args.encoder_hidden_dimensions,
        args.n_clusters,
        args.embeddings_sample_fraction,
    )
