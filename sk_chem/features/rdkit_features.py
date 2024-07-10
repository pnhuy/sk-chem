import pandas as pd
from rdkit import Chem
from rdkit.Chem.Descriptors import CalcMolDescriptors
from rdkit.Chem.rdchem import Mol
from sklearn.base import BaseEstimator, TransformerMixin


class RdkitFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, missingVal=None, silent=True):
        self.missingVal = missingVal
        self.silent = silent
        self.feature_names = []

    def get_feature_names_out(self) -> list[str]:
        return self.feature_names

    def _calc(self, m: Mol) -> dict[str, float]:
        m = Chem.AddHs(m)
        return CalcMolDescriptors(m, missingVal=self.missingVal, silent=self.silent)

    def fit(self, X: list[Mol], y=None) -> object:
        return self

    def transform(self, X: list[Mol], y=None):
        features_dict = [self._calc(x) for x in X]
        # TODO: remove pandas dependencies
        features_df = pd.DataFrame(features_dict)
        self.feature_names = list(features_df.columns)
        return features_df.to_numpy()
