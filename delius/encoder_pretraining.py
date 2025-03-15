import argparse
from tqdm.autonotebook import tqdm
import os

from dotenv import load_dotenv
import mlflow

from delius.clustering.modules.features_dataset import load_features_dataset
from delius.clustering.encoder_pretraining import pretrain_encoder
from delius.clustering.modules.features_encoder import FeaturesEncoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_embeddings_file', type=str)
    parser.add_argument('--output_encoder_mlflow_run_id_path', type=str)
    parser.add_argument('--encoder_hidden_dimensions', type=int, nargs='+')
    parser.add_argument('--batch', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    tqdm.write(f"Loading features from '{args.input_embeddings_file}'...")
    dataset = load_features_dataset(args.input_embeddings_file)
    tqdm.write(f"Done.")

    tqdm.write(f"Loading environment variables...")
    load_dotenv()
    tqdm.write(f"Done.")

    tqdm.write(f"Creating a new MLFlow run...")

    with mlflow.start_run(run_name='Encoder pre-training') as run:
        tqdm.write(f"Done.")

        mlflow.log_param('batch', args.batch)
        mlflow.log_param('learning rate', args.learning_rate)
        mlflow.log_param('epochs', args.epochs)
        mlflow.log_param('seed', args.seed)
        mlflow.log_param("hidden_dims", str(args.encoder_hidden_dimensions))

        model = pretrain_encoder(
            dataset,
            args.encoder_hidden_dimensions,
            args.batch,
            args.epochs,
            args.learning_rate,
            args.seed
        )

        model = model.to('cpu')
        mlflow.pytorch.log_model(model, FeaturesEncoder.__name__)

        directory = os.path.dirname(args.output_encoder_mlflow_run_id_path)

        if not os.path.exists(directory):
            os.makedirs(directory)
            tqdm.write(f"Created directory '{directory}'.")

        tqdm.write(f"Saving the MLFlow run id to '{args.output_encoder_mlflow_run_id_path}'...")

        with open(args.output_encoder_mlflow_run_id_path, 'w') as file:
            file.write(run.info.run_id)

        tqdm.write(f"Done.")
