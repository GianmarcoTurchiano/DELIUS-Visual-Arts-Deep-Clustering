import torch
from torch.utils.data import Dataset

class EmbeddingsDataset(Dataset):
    def __init__(
        self,
        file_path
    ):
        data = torch.load(file_path)
        self.names = list(data.keys())
        self.embeddings = list(data.values())

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        names = self.names[idx]
        embeddings = self.embeddings[idx]
        
        return names, embeddings
