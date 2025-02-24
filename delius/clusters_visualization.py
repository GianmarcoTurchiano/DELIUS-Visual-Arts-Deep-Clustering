import argparse
from tqdm import tqdm

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
    output_tsne_pic_file_path,
    output_umap_pic_file_path,
    batch_size=256,
    input_embeddings_dimensions=1024,
    encoder_hidden_dimensions=[500, 500, 2000, 10],
    n_clusters=5
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
        torch.zeros(n_clusters, encoder_hidden_dimensions[-1])
    ).to(device)

    model.load_state_dict(weights)

    model.eval()
    embeddings_all = []
    q_all = []

    with torch.no_grad():
        for _, features in tqdm(loader, desc='Computing embeddings and cluster assignments'):
            features = features.to(device)
            embeddings, q = model(features)
            q_all.append(q.detach().cpu())
            embeddings_all.append(embeddings.cpu())
    
    q_all = torch.cat(q_all, dim=0)
    embeddings_all = torch.cat(embeddings_all, dim=0)

    y_pred = torch.argmax(q_all, dim=1).numpy()

    tqdm.write('Applying TSNE...')

    tsne = TSNE(n_components=2, random_state=42)
    tsne_embedding = tsne.fit_transform(embeddings_all)

    tqdm.write(f"Saving image to '{output_tsne_pic_file_path}'...")

    plt.figure(dpi=600)
    plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=y_pred, cmap='viridis')
    plt.savefig(output_tsne_pic_file_path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dec_file', type=str)
    parser.add_argument('--input_embeddings_file', type=str)
    parser.add_argument('--output_tsne_pic_file', type=str)
    parser.add_argument('--output_umap_pic_file', type=str)
    parser.add_argument('--batch', type=int)

    args = parser.parse_args()

    display_clusters(
        args.input_embeddings_file,
        args.input_dec_file,
        args.output_tsne_pic_file,
        args.output_umap_pic_file,
        args.batch
    )
