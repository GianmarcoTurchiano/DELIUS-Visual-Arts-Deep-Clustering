import argparse
from tqdm.autonotebook import tqdm
import os

from dotenv import load_dotenv
import torch
import mlflow

from delius.clustering.modules.features_encoder import FeaturesEncoder
from delius.clustering.modules.features_dataset import load_features_dataset
from delius.clustering.dec_fitting import fit_dec, DEC


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_centroids_file', type=str)
    parser.add_argument('--input_assignments_file', type=str)
    parser.add_argument('--input_encoder_mlflow_run_id_path', type=str)
    parser.add_argument('--input_embeddings_file', type=str)
    parser.add_argument('--output_dec_mlflow_run_id_file', type=str)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--update_interval', type=int)
    parser.add_argument('--delta_tol', type=float)
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    tqdm.write(f"Loading environment variables...")
    load_dotenv()
    tqdm.write(f"Done.")

    tqdm.write(f"Loading features from '{args.input_embeddings_file}'...")
    dataset = load_features_dataset(args.input_embeddings_file)
    tqdm.write(f"Done.")

    tqdm.write(f"Loading initial centroids from '{args.input_centroids_file}'...")
    centroids = torch.load(args.input_centroids_file, weights_only=True)
    tqdm.write(f"Done.")

    tqdm.write(f"Loading initial cluster assignments from '{args.input_assignments_file}'...")
    assignments = torch.load(args.input_assignments_file, weights_only=True)
    tqdm.write(f"Done.") 

    with open(args.input_encoder_mlflow_run_id_path, 'r') as file:
        parent_run_id = file.read()

    model_uri = f'runs:/{parent_run_id}/{FeaturesEncoder.__name__}'

    tqdm.write(f"Loading encoder from '{model_uri}'...")
    encoder = mlflow.pytorch.load_model(model_uri)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = encoder.to(device)
    tqdm.write(f"Done.")

    tqdm.write(f"Creating a new MLFlow child run (parent run id: {parent_run_id})...")

    with mlflow.start_run(
        parent_run_id=parent_run_id,
        nested=True,
        run_name='DEC fitting'
    ) as child_run:
        tqdm.write(f"Done.")

        mlflow.log_param('batch', args.batch)
        mlflow.log_param('learning rate', args.learning_rate)
        mlflow.log_param('steps', args.steps)
        mlflow.log_param('seed', args.seed)
        mlflow.log_param('update interval', args.update_interval)
        mlflow.log_param('delta tolerance', args.delta_tol)

        n_clusters, _ = centroids.shape

        mlflow.log_param('clusters count', n_clusters)

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
            args.seed
        )

        model = model.to('cpu')
        mlflow.pytorch.log_model(model, DEC.__name__)

        directory = os.path.dirname(args.output_dec_mlflow_run_id_file)

        if not os.path.exists(directory):
            os.makedirs(directory)
            tqdm.write(f"Created directory '{directory}'.")

        tqdm.write(f"Saving the MLFlow run id to '{args.output_dec_mlflow_run_id_file}'...")

        with open(args.output_dec_mlflow_run_id_file, 'w') as file:
            file.write(child_run.info.run_id)

        tqdm.write(f"Done.")
