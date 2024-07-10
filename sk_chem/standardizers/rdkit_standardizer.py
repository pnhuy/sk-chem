from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdchem import Mol
from sklearn.base import BaseEstimator, TransformerMixin


class RdkitMolStandardizer(BaseEstimator, TransformerMixin):
    def __init__(self, input_format: str = "smiles"):
        self.input_format = input_format

    @staticmethod
    def standardize_mol(mol: Mol):
        _ = Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        mol = rdMolStandardize.MetalDisconnector().Disconnect(mol)
        mol = rdMolStandardize.Normalize(mol)
        mol = rdMolStandardize.Reionize(mol)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        return mol

    def _standardize(self, x: list[str | Mol]):
        if isinstance(x, Mol):
            return self.standardize_mol(x)
        elif isinstance(x, str) and self.input_format == "smiles":
            mol = Chem.MolFromSmiles(x)
            mol = self.standardize_mol(mol)
            return Chem.MolToSmiles(mol)

    def fit(self, X: list[str | Mol], y=None):
        return self

    def transform(self, X: list[str | Mol]):
        mols = [self._standardize(x) for x in X]
        return mols
