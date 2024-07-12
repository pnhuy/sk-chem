from rdkit.Chem.rdchem import Mol
from sklearn.base import BaseEstimator, TransformerMixin

from sk_chem.logger import logger
from sk_chem.conformations.molecule_embedder import MoleculeEmbedder


class BaseForceField(BaseEstimator, TransformerMixin):
    def __init__(self, maxAttempts: int = 5000, randomSeed: int = 42):
        self.maxAttempts = maxAttempts
        self.randomSeed = randomSeed

    def fit(self, X: list[Mol], y=None):
        return self

    def _optimize_fn(self, m: Mol) -> Mol:
        return m

    def _optimize(self, m: Mol) -> Mol:
        m = MoleculeEmbedder._process(m)
        m = self._optimize_fn(m)
        return m

    def transform(self, X: list[Mol], y=None):
        mols = [self._optimize(x) for x in X]
        return mols
