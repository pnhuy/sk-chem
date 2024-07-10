import numpy as np
from mordred import Calculator, descriptors
from rdkit.Chem.rdchem import Mol
from sklearn.base import BaseEstimator, TransformerMixin

class ModredFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, ignore_3D=False, silent=True):
        self.ignore_3D = ignore_3D
        self.silent = silent
        self.feature_names = []

    def get_feature_names_out(self) -> list[str]:
        return self.feature_names

    def fit(self, X: list[Mol], y=None) -> object:
        return self

    def transform(self, X: list[Mol], y=None) -> np.array:
        calculator = Calculator(descriptors, ignore_3D=self.ignore_3D)
        features_df = calculator.pandas(X, quiet=self.silent)
        self.feature_names = list(features_df.columns)
        return features_df.to_numpy()
        
