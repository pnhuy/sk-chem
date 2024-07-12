from rdkit import Chem
from sk_chem.conformations.molecule_embedder import MoleculeEmbedder


SAMPLE_SMILES = [
    "CCO",
    "CC(=O)O",
    "C1=CC=CC=C1",
    "C1=CC=CC=C1C1=CC=CC=C1",
    "C1=CC=CC=C1C1=CC=CC=C1C1=CC=CC=C1",
    "C1=CC=CC=C1C1=CC=CC=C1C1=CC=CC=C1C1=CC=CC=C1",
]

def test_molecule_embedder():
    embedder = MoleculeEmbedder()
    mols = [Chem.MolFromSmiles(smi) for smi in SAMPLE_SMILES]
    mols = embedder.transform(mols)
    assert len(mols) == len(SAMPLE_SMILES)
    assert all([m is not None for m in mols])
    assert all([m.GetNumConformers() > 0 for m in mols])