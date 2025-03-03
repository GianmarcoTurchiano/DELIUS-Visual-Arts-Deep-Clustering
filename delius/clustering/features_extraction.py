from pathlib import Path
from itertools import chain
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.autonotebook import tqdm


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
