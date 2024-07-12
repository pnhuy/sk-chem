import tempfile
from rdkit.Chem.rdchem import Mol
from rdkit import Chem as RdkitChem
import CDPL.Chem as CDPLChem


def mol_rdkit_to_cdpkit(mol: Mol, medium='2d'):
    """
    Convert an RDKit molecule to a CDPKit molecule.

    Parameters
    ----------
    mol : Mol
        An RDKit molecule.

    Returns
    -------
    CDPLChem.Molecule
        A CDPKit molecule.
    """
    if medium == '2d':
        mol_block = RdkitChem.MolToMolBlock(mol)
        suffix = '.mol'
    elif medium == '3d':
        mol_block = RdkitChem.MolToXYZBlock(mol)
        suffix = '.xyz'
    else:
        raise ValueError("medium must be either '2d' or '3d'")

    with tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=suffix) as f:
        f.write(mol_block)
        f.flush()
        reader = CDPLChem.MoleculeReader(f.name)
        output = CDPLChem.BasicMolecule()
        reader.read(output)
        reader.close()

    return output

def mol_cdpkit_to_rdkit(mol: CDPLChem.Molecule, medium='2d'):
    """
    Convert a CDPKit molecule to an RDKit molecule.

    Parameters
    ----------
    mol : BasicMolecule
        A CDPKit molecule.

    Returns
    -------
    Mol
        An RDKit molecule.
    """
    if medium == '2d':
        suffix = '.mol'
    elif medium == '3d':
        suffix = '.xyz'
    else:
        raise ValueError("medium must be either '2d' or '3d'")

    with tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=suffix) as f:
        fp = f.name
        writer = CDPLChem.MolecularGraphWriter(fp)
        writer.write(mol)
        writer.close()
        output = RdkitChem.MolFromXYZFile(fp) if medium == '3d' else RdkitChem.MolFromMolFile(fp)

    return output