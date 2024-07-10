from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Chem.rdchem import Mol
from sklearn.base import BaseEstimator, TransformerMixin

from sk_chem.logger import logger


class BaseForceField(BaseEstimator, TransformerMixin):
    def __init__(self, maxAttempts: int = 5000, randomSeed: int = 42):
        self.maxAttempts = maxAttempts
        self.randomSeed = randomSeed

    def fit(self, X: list[Mol], y=None):
        return self

    def _embed(self, m: Mol):
        res = EmbedMolecule(m, randomSeed=self.randomSeed)
        if res != 0:
            res = EmbedMolecule(m, maxAttempts=self.maxAttempts, randomSeed=self.randomSeed)
            if res != 0:
                res = EmbedMolecule(m, maxAttempts=self.maxAttempts, useRandomCoords=True, randomSeed=self.randomSeed)
            else:
                logger.warning("Unable to generate conformation")
                return -1
        return 0

    def _optimize_fn(self, m: Mol) -> Mol:
        return m

    def _optimize(self, m: Mol) -> Mol:
        embed = self._embed(m)
        if embed == 0:
            m = self._optimize_fn(m)
        return m

    def transform(self, X: list[Mol], y=None):
        mols = [self._optimize(x) for x in X]
        return mols
