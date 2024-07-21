from openbabel import pybel


def smiles_to_xzy_block(smiles: str) -> str:
    mol = pybel.readstring("smi", smiles)
    # add hydrogens
    mol.addh()
    # gen 3d
    mol.localopt()
    # save xyz 
    xyz_string = str(mol.write("xyz"))
    return xyz_string