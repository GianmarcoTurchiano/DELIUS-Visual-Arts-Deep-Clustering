import argparse
from tqdm.autonotebook import tqdm
import os

from dotenv import load_dotenv, set_key
import mlflow
import torch

from delius.clustering.modules.features_dataset import load_features_dataset
from delius.clustering.encoder_pretraining import pretrain_encoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_embeddings_file', type=str)
    parser.add_argument('--output_encoder_file', type=str)
    parser.add_argument('--encoder_hidden_dimensions', type=int, nargs='+')
    parser.add_argument('--batch', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()
    env_file_name = '.env'
    env_var_mlflow_run = 'MLFLOW_RUN_ID'

    tqdm.write(f"Loading features from '{args.input_embeddings_file}'...")
    dataset = load_features_dataset(args.input_embeddings_file)
    tqdm.write(f"Done.")

    tqdm.write(f"Loading environment variables from '{env_file_name}'...")
    load_dotenv(env_file_name)
    tqdm.write(f"Done.")

    os.environ.pop(env_var_mlflow_run, None)

    tqdm.write(f"Creating a new MLFlow run...")

    with mlflow.start_run(run_name='Encoder pre-training') as run:
        tqdm.write(f"Done.")

        set_key(env_file_name, env_var_mlflow_run, run.info.run_id)

        mlflow.log_param('batch', args.batch)
        mlflow.log_param('learning rate', args.learning_rate)
        mlflow.log_param('epochs', args.epochs)
        mlflow.log_param('seed', args.seed)

        model = pretrain_encoder(
            dataset,
            args.encoder_hidden_dimensions,
            args.batch,
            args.epochs,
            args.learning_rate,
            args.seed,
            lambda epoch, loss: mlflow.log_metric('MSE Loss', loss, epoch)
        )

    tqdm.write(f"Saving encoder to '{args.output_encoder_file}'...")
    torch.save(model.state_dict(), args.output_encoder_file)
