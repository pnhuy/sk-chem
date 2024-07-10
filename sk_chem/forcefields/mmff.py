from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem.rdchem import Mol
from sk_chem.forcefields.base import BaseForceField


class MMFF(BaseForceField):
    def __init__(
        self,
        maxAttempts: int = 5000,
        mmffVariant: str = "MMFF94",
        maxIters: int = 200,
        nonBondedThresh: float = 100.0,
        confId: int = -1,
        ignoreInterfragInteractions: bool = True,
    ):
        super().__init__(maxAttempts=maxAttempts)
        self.mmffVariant = mmffVariant
        self.maxIters = maxIters
        self.nonBondedThresh = nonBondedThresh
        self.confId = confId
        self.ignoreInterfragInteractions = ignoreInterfragInteractions

    def _optimize_fn(self, m: Mol) -> Mol:
        _ = MMFFOptimizeMolecule(
            m,
            mmffVariant=self.mmffVariant,
            maxIters=self.maxIters,
            nonBondedThresh=self.nonBondedThresh,
            confId=self.confId,
            ignoreInterfragInteractions=self.ignoreInterfragInteractions,
        )
        return m

