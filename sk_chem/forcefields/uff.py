from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
from sk_chem.forcefields.base import BaseForceField


class UFF(BaseForceField):
    def __init__(
        self,
        maxIters: int = 200,
        vdwThresh: float = 10.0,
        confId: int = -1,
        ignoreInterfragInteractions: bool = True,
    ):
        super().__init__()
        self.maxIters = maxIters
        self.vdwThresh = vdwThresh
        self.confId = confId
        self.ignoreInterfragInteractions = ignoreInterfragInteractions

    def _optimize_fn(self, m: Mol) -> Mol:
        _ = UFFOptimizeMolecule(
            m,
            maxIters=self.maxIters,
            vdwThresh=self.vdwThresh,
            confId=self.confId,
            ignoreInterfragInteractions=self.ignoreInterfragInteractions,
        )
        return m
