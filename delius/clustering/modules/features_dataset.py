import torch
from torch.utils.data import Dataset


class FeaturesDataset(Dataset):
    def __init__(
        self,
        names: list[str],
        embeddings: torch.Tensor
    ):
        self.names = names
        self.embeddings = embeddings
        _, self.features_dims = self.embeddings.shape

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        names = self.names[idx]
        embeddings = self.embeddings[idx]
        
        return idx, names, embeddings


def save_features_dataset(names: list[str], features: torch.Tensor, output_embedding_file: str):
    embeddings = dict(zip(names, features))

    torch.save(embeddings, output_embedding_file)


def load_features_dataset(file_path: str):
    data = torch.load(file_path, weights_only=True)

    names = list(data.keys())
    embeddings = list(data.values())
    embeddings = torch.stack(embeddings)

    dataset = FeaturesDataset(names, embeddings)

    return dataset
