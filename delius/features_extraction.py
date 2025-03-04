import argparse
import tarfile

from tqdm.autonotebook import tqdm

from delius.clustering.modules.features_extractor import DenseNetFeaturesExtractor
from delius.clustering.features_extraction import extract_features
from delius.clustering.modules.features_dataset import save_features_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_compressed_image_dir', type=str)
    parser.add_argument('--output_uncompressed_image_dir', type=str)
    parser.add_argument('--output_embedding_file', type=str)
    parser.add_argument('--image_width', type=int)
    parser.add_argument('--image_height', type=int)
    parser.add_argument('--batch_size', type=int)

    args = parser.parse_args()

    input_compressed_image_dir = args.input_compressed_image_dir
    output_uncompressed_image_dir = args.output_uncompressed_image_dir
    batch_size = args.batch_size
    image_width = args.image_width
    image_height = args.image_height
    output_embedding_file = args.output_embedding_file

    tqdm.write(f"Uncompressing files from '{input_compressed_image_dir}' to '{output_uncompressed_image_dir}'...")

    with tarfile.open(input_compressed_image_dir, "r:gz") as tar:
        tar.extractall(output_uncompressed_image_dir)

    tqdm.write(f'Loading model...')

    extractor = DenseNetFeaturesExtractor()

    names, features = extract_features(
        extractor,
        output_uncompressed_image_dir,
        image_width,
        image_height,
        batch_size
    )

    tqdm.write(f"Saving embeddings to file '{output_embedding_file}'...")

    save_features_dataset(names, features, output_embedding_file)
