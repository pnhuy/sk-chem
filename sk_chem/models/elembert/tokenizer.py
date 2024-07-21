import json
import os
from typing import Any
from urllib.parse import urljoin

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from cached_path import cached_path
from joblib import load
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from scipy.ndimage import gaussian_filter1d
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm.auto import tqdm

from sk_chem.utils.io import smiles_to_xzy_block

PCAKM_PREFIX = "https://raw.githubusercontent.com/pnhuy/elembert/main/models/"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(CURRENT_DIR, "static", "el2idv1.json")) as f:
    EL2IDV1 = json.load(f)

with open(os.path.join(CURRENT_DIR, "static", "el2idv0.json")) as f:
    EL2IDV0 = json.load(f)


class ElementTokenizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        data_prefix=PCAKM_PREFIX,
        x=np.arange(0, 10, 0.1),
        verbose=True,
    ):
        self.data_prefix = data_prefix
        self.verbose = verbose
        self._load_data(data_prefix)
        self.x = x
        self.v = np.concatenate([[1], 4 * np.pi / 3 * (x[1:] ** 3 - x[:-1] ** 3)])

    def _load_data(self, data_prefix):
        self.km = load(cached_path(urljoin(PCAKM_PREFIX, "km.joblib")))
        self.pca = load(cached_path(urljoin(PCAKM_PREFIX, "pca.joblib")))

    def _tokenize(self, atoms):
        types = atoms.get_chemical_symbols()
        v = self.v
        i, d = neighbor_list("id", atoms, 10.0, self_interaction=False)
        ntypes = []
        for k, _ in enumerate(atoms):
            el = types[k]
            y = np.zeros(100)
            dist = np.round(d[i == k] * 10)
            a, b = np.unique(dist, return_counts=True)
            np.put(y, a.astype(int) - 1, b)
            values = gaussian_filter1d(y / v, 1)
            if (el not in self.km) or (el not in self.pca):
                num = 0
            else:
                num = self.km[el].predict(
                    self.pca[el].transform(
                        np.nan_to_num([values], nan=0, posinf=0, neginf=0)
                    )
                )[0]
            ntypes.append(el + str(num))  # el2id[el+str(num)]
        return ntypes

    @staticmethod
    def rdkit_mol_to_ase_atoms(mol: Mol, optimize: bool = False) -> Atoms:
        """Convert an RDKit molecule to an ASE Atoms object.

        Args:
            rdkit_mol: RDKit molecule object.

        Returns:
            ASE Atoms object.

        Credit: https://gist.github.com/hunter-heidenreich/82c70c980d5b01ca304f270b1e92c824
        """

        ase_atoms = Atoms(
            numbers=[
                atom.GetAtomicNum()
                for atom in mol.GetAtoms()  # type: ignore
            ],
            positions=mol.GetConformer().GetPositions(),
        )
        return ase_atoms

    def tokenize_mol(self, mol: Mol) -> list:
        atoms = self.rdkit_mol_to_ase_atoms(mol)
        return self._tokenize(atoms)

    def __call__(self, mol: Any) -> Any:
        if isinstance(mol, Mol):
            return self.tokenize_mol(mol)
        elif isinstance(mol, Atoms):
            return self._tokenize(mol)
        else:
            raise ValueError("Input must be RDKit molecule or ASE Atoms object.")

    def preprocess(self, smiles: str):
        mapper = EL2IDV1["el2id"]
        unk_id = mapper.get("[UNK]")
        # convert smiles to xyz
        xyz = smiles_to_xzy_block(smiles)

        # convert xyz to mol
        mol = Chem.MolFromXYZBlock(xyz)

        # tokenize mol
        types = self(mol)

        # pad types
        types0 = ["[CLS]"] + types + ["[SEP]"]
        typesNumerical = []
        for i in types0:
            typesNumerical.append(mapper.get(i, unk_id))

        return typesNumerical

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [
            self.preprocess(mol)
            for mol in tqdm(X, disable=not self.verbose, desc="Tokenizing")
        ]
