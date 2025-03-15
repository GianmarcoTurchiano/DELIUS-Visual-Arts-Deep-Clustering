import torch
from torch.utils.data import Dataset
import pandas as pd


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


class LabelledFeaturesDataset(FeaturesDataset):
    def __init__(
        self,
        names: list[str],
        embeddings: torch.Tensor,
        df: pd.DataFrame
    ):
        super().__init__(names, embeddings)

        assert all(df['subject'].isin(self.names)), "All subjects must be in the names list"

        df = df.set_index('subject')
        self.df = pd.DataFrame({col: pd.factorize(df[col])[0] for col in df.columns}, index=df.index)

    def __getitem__(self, idx):
        _, names, embeddings = super().__getitem__(idx)
        labels = self.df.loc[names].values

        return idx, names, embeddings, labels


def save_features_dataset(names: list[str], features: torch.Tensor, output_embedding_file: str):
    embeddings = dict(zip(names, features))

    torch.save(embeddings, output_embedding_file)


def _load_features(file_path):
    data = torch.load(file_path, weights_only=True)

    names = list(data.keys())
    embeddings = list(data.values())
    embeddings = torch.stack(embeddings)

    return names, embeddings


def load_features_dataset(file_path: str):
    names, embeddings = _load_features(file_path)

    dataset = FeaturesDataset(names, embeddings)

    return dataset


def load_labelled_features_dataset(embs_file_path: str, df_file_path: str):
    names, embeddings = _load_features(embs_file_path)
    df = pd.read_csv(df_file_path)

    dataset = LabelledFeaturesDataset(names, embeddings, df)

    return dataset
