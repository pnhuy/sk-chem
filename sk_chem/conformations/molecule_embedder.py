from sklearn.base import BaseEstimator, TransformerMixin
from rdkit.Chem import AddHs, MolToSmiles
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdDistGeom import EmbedMolecule

from sk_chem.logger import logger


class MoleculeEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _process(m: Mol, randomSeed: int = 0xf00d, maxAttempts: int = 5000, useRandomCoords: bool = False):
        m = AddHs(m)
        res = EmbedMolecule(m, randomSeed=randomSeed)
        if res != 0:
            res = EmbedMolecule(m, maxAttempts=maxAttempts, randomSeed=randomSeed)
            if res != 0:
                res = EmbedMolecule(m, maxAttempts=maxAttempts, useRandomCoords=useRandomCoords, randomSeed=randomSeed)
            else:
                logger.info("Smiles: %s", MolToSmiles(m))
                raise Exception("Unable to generate conformation")
        return m
    
    def fit(self, X: list[str | Mol], y=None):
        return self
    
    def transform(self, X: list[Mol]) -> list[Mol]:
        output = [self._process(x) for x in X]
        return output