import argparse
from pathlib import Path
from itertools import chain
from PIL import Image
import tarfile

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from delius.modules.features_extractor import DenseNetFeaturesExtractor


class PicsDataset(Dataset):
    def __init__(
        self,
        image_directory,
        image_width=224,
        image_height=224,
        image_extensions=[
            '*.jpg',
            '*.jpeg',
            '*.png',
            '*.bmp',
            '*.gif',
            '*.tiff'
        ]
    ):
        self.transform = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: x / 255.0) # Min-Max normalization 
        ])

        image_directory_path = Path(image_directory)

        if not image_directory_path.is_dir():
            raise NotADirectoryError(f"'{image_directory}' is not a valid directory.")

        self.image_paths = list(chain.from_iterable(image_directory_path.glob(ext) for ext in image_extensions))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return str(img_path), image
    

def extract_features(
        extractor: nn.Module,
        image_directory: str,
        image_width=224,
        image_height=224,
        batch_size=256
):
    tqdm.write(f"Reading '{image_directory}'...")

    dataset = PicsDataset(
        image_directory,
        image_width,
        image_height
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = extractor.to(device)
    extractor.eval()

    names = []
    embeddings = []

    with torch.no_grad():
        for paths, images in tqdm(loader, desc='Extracting image features'):
            name = list(map(lambda path: path.split('/')[-1], paths))
            names.extend(name)
            
            images = images.to(device)
            embedding = extractor(images)
            embeddings.extend(embedding)

    return dict(zip(names, embeddings))


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

    embeddings = extract_features(
        extractor,
        output_uncompressed_image_dir,
        image_width,
        image_height,
        batch_size
    )

    tqdm.write(f"Saving embeddings to file '{output_embedding_file}'...")

    torch.save(embeddings, output_embedding_file)
