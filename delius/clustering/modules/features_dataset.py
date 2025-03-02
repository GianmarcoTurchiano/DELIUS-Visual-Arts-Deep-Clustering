import torch
from torch.utils.data import Dataset


class FeaturesDataset(Dataset):
    def __init__(
        self,
        embeddings: dict
    ):
        self.names = list(embeddings.keys())
        self.embeddings = list(embeddings.values())

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        names = self.names[idx]
        embeddings = self.embeddings[idx]
        
        return idx, names, embeddings


def load_features_dataset(file_path: str):
    data = torch.load(file_path, weights_only=True)

    dataset = FeaturesDataset(data)

    return dataset