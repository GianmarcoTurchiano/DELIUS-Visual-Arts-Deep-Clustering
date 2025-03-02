import argparse
from tqdm import tqdm

import torch

from delius.clustering.modules.features_dataset import load_features_dataset
from delius.clustering.encoder_pretraining import pretrain_encoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_embeddings_file', type=str)
    parser.add_argument('--output_encoder_file', type=str)
    parser.add_argument('--input_embeddings_dimensions', type=int)
    parser.add_argument('--encoder_hidden_dimensions', type=int, nargs='+')
    parser.add_argument('--batch', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    tqdm.write(f"Loading features from '{args.input_embeddings_file}'...")
    dataset = load_features_dataset(args.input_embeddings_file)

    model = pretrain_encoder(
        dataset,
        args.input_embeddings_dimensions,
        args.encoder_hidden_dimensions,
        args.batch,
        args.epochs,
        args.learning_rate,
        args.seed
    )

    tqdm.write(f"Saving encoder to '{args.output_encoder_file}'...")
    torch.save(model.state_dict(), args.output_encoder_file)
