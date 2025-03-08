import argparse
from tqdm.autonotebook import tqdm
import os

from dotenv import load_dotenv, set_key
import torch
import mlflow

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
    parser.add_argument('--batch', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--update_interval', type=int)
    parser.add_argument('--delta_tol', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--env_file_path', type=str)
    parser.add_argument('--mlflow_run_id_env_var_name', type=str)
    parser.add_argument('--mlflow_parent_run_id_env_var_name', type=str)

    args = parser.parse_args()

    tqdm.write(f"Loading features from '{args.input_embeddings_file}'...")
    dataset = load_features_dataset(args.input_embeddings_file)
    tqdm.write(f"Done.")

    tqdm.write(f"Loading initial centroids from '{args.input_centroids_file}'...")
    centroids = torch.load(args.input_centroids_file, weights_only=True)
    tqdm.write(f"Done.")

    tqdm.write(f"Loading initial cluster assignments from '{args.input_assignments_file}'...")
    assignments = torch.load(args.input_assignments_file, weights_only=True)
    tqdm.write(f"Done.")

    tqdm.write(f"Loading encoder from '{args.input_pretrained_encoder_file}'...")
    encoder = load_features_encoder(args.input_pretrained_encoder_file)
    tqdm.write(f"Done.")

    tqdm.write(f"Loading environment variables from '{args.env_file_path}'...")
    load_dotenv(args.env_file_path)
    tqdm.write(f"Done.")

    parent_run_id = os.environ[args.mlflow_parent_run_id_env_var_name]

    tqdm.write(f"Creating a new MLFlow child run (parent run id: {parent_run_id})...")

    with mlflow.start_run(
        parent_run_id=parent_run_id,
        nested=True,
        run_name='DEC fitting'
    ) as child_run:
        tqdm.write(f"Done.")

        set_key(args.env_file_path, args.mlflow_run_id_env_var_name, child_run.info.run_id)

        mlflow.log_param('batch', args.batch)
        mlflow.log_param('learning rate', args.learning_rate)
        mlflow.log_param('steps', args.steps)
        mlflow.log_param('seed', args.seed)
        mlflow.log_param('update interval', args.update_interval)
        mlflow.log_param('delta tolerance', args.delta_tol)

        model = fit_dec(
            encoder,
            dataset,
            centroids,
            assignments,
            args.batch,
            args.learning_rate,
            args.steps,
            args.update_interval,
            args.delta_tol,
            args.seed,
            lambda step, loss: mlflow.log_metric('delta label', loss, step),
            lambda step, loss: mlflow.log_metric('KL loss', loss, step),
            lambda n_clusters: mlflow.log_param('clusters count', n_clusters)
        )

    directory = os.path.dirname(args.output_dec_file)

    if not os.path.exists(directory):
        os.makedirs(directory)
        tqdm.write(f"Created directory '{directory}'.")

    tqdm.write(f"Saving DEC to '{args.output_dec_file}'...")
    torch.save(model.state_dict(), args.output_dec_file)
    tqdm.write(f"Done.")
