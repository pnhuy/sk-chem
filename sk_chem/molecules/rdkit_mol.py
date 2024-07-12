from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdchem import Mol
from sklearn.base import BaseEstimator, TransformerMixin


class RdkitMoleculeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, input_format: str = "smiles", addHs: bool = True, sanitize: bool = False):
        self.input_format = input_format
        self.addHs = addHs
        self.sanitize = sanitize

    def _converter(self, inp: str):
        if self.input_format.lower() == "smiles":
            mol = MolFromSmiles(inp, sanitize=self.sanitize)
        else:
            raise ValueError("Not valid input format.")

        if self.addHs:
            mol.UpdatePropertyCache(strict=self.sanitize)
            mol = Chem.AddHs(mol)

        return mol

    def fit(self, X: list[str], y=None):
        return self

    def transform(self, X: list[str], y=None) -> list[Mol]:
        return [self._converter(x) for x in X]
