import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
from tqdm.auto import tqdm


class SmilesDataset(Dataset):
    def __init__(self, smiles: list[str]):
        self.smiles = smiles

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx]


class HFTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        batch_size: int = 2,
        device: str = "cpu",
        progression_bar: bool = False,
    ):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.batch_size = batch_size
        self.device = device
        self.progression_bar = progression_bar

    def setup(self):
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.pretrained_model_name_or_path,
            torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path
        )
        self.pipeline = pipeline(
            "feature-extraction",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

    def fit(self, X: list[str], y=None):
        self.setup()
        return self

    def _transform(self, smiles: str):
        feat = self.pipeline(smiles)
        feat = [i[0][0] for i in feat]  # type: ignore
        return feat

    def transform(self, X: list[str] | pd.Series | pd.DataFrame, y=None):
        if isinstance(X, pd.Series):
            X = X.to_list()
        elif isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].to_list()

        # create torch dataset
        dataset = SmilesDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # extract features
        features = []
        if self.progression_bar:
            dataloader = tqdm(dataloader, total=len(dataloader))
        for smiles in dataloader:
            features.extend(self._transform(smiles))
        return np.array(features)
