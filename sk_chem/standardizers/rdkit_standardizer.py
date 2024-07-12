from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdchem import Mol
from sklearn.base import BaseEstimator, TransformerMixin


class RdkitMolStandardizer(BaseEstimator, TransformerMixin):
    def __init__(self, input_format: str = "smiles", add_hs: bool = False):
        self.input_format = input_format
        self.add_hs = add_hs

    @staticmethod
    def standardize_mol(mol: Mol, sanitize=True, add_hs=False):
        if not sanitize:
            mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_PROPERTIES)

        # Iterate over atoms to find and fix valence issues
        for atom in mol.GetAtoms(): # type: ignore
            if atom.GetAtomicNum() == 7:  # Nitrogen
                valence = atom.GetExplicitValence()
                if valence > 3:
                    atom.SetFormalCharge(valence - 3)
                    atom.SetNumExplicitHs(0)
        
        try:
            mol = Chem.AddHs(mol) if add_hs else mol
        except Exception:
            pass

        mol = rdMolStandardize.MetalDisconnector().Disconnect(mol)
        mol = rdMolStandardize.Normalize(mol)
        mol = rdMolStandardize.Reionize(mol)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        return mol

    def _standardize(self, x: list[str | Mol]):
        if isinstance(x, Mol):
            return self.standardize_mol(x, add_hs=self.add_hs)
        elif isinstance(x, str) and self.input_format == "smiles":
            mol = Chem.MolFromSmiles(x)
            mol = self.standardize_mol(mol)
            return Chem.MolToSmiles(mol)

    def fit(self, X: list[str | Mol], y=None):
        return self

    def transform(self, X: list[str | Mol]):
        mols = [self._standardize(x) for x in X]
        return mols
