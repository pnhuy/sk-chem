import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdchem import Mol
from sklearn.base import BaseEstimator, TransformerMixin


class BaseFingerprint(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def get_feature_names_out(self):
        return [f"fp_{idx}" for idx in range(self.fpSize)]

    def transform(self, X: list[Mol], y=None):
        fps = [self.fpgen.GetFingerprintAsNumPy(mol) for mol in X]
        return np.vstack(fps).astype("int")


class AtomPairCountFingerprint(BaseFingerprint):
    def __init__(
        self,
        minDistance: int = 1,  # params for GetAtomPairGenerator
        maxDistance: int = 30,
        includeChirality: bool = False,
        use2D: bool = True,
        countSimulation: bool = True,
        countBounds=None,
        fpSize: int = 2048,
        atomInvariantsGenerator=None,
    ):
        super().__init__()
        self.minDistance = minDistance
        self.maxDistance = maxDistance
        self.includeChirality = includeChirality
        self.use2D = use2D
        self.countSimulation = countSimulation
        self.countBounds = countBounds
        self.fpSize = fpSize
        self.atomInvariantsGenerator = atomInvariantsGenerator

    def fit(self, X: list[Mol], y=None):
        self.fpgen = rdFingerprintGenerator.GetAtomPairGenerator(
            minDistance=self.minDistance,
            maxDistance=self.maxDistance,
            includeChirality=self.includeChirality,
            use2D=self.use2D,
            countSimulation=self.countSimulation,
            countBounds=self.countBounds,
            fpSize=self.fpSize,
            atomInvariantsGenerator=self.atomInvariantsGenerator,
        )

        return self


class RdkitFingerprint(BaseFingerprint):
    def __init__(
        self,
        minPath: int = 1,
        maxPath: int = 7,
        useHs: bool = True,
        branchedPaths: bool = True,
        useBondOrder: bool = True,
        countSimulation: bool = False,
        countBounds=None,
        fpSize: int = 2048,
        numBitsPerFeature: int = 2,
        atomInvariantsGenerator=None,
    ):
        super().__init__()
        self.minPath = minPath
        self.maxPath = maxPath
        self.useHs = useHs
        self.branchedPaths = branchedPaths
        self.useBondOrder = useBondOrder
        self.countSimulation = countSimulation
        self.countBounds = countBounds
        self.fpSize = fpSize
        self.numBitsPerFeature = numBitsPerFeature
        self.atomInvariantsGenerator = atomInvariantsGenerator

    def fit(self, X: list[Mol], y=None):
        self.fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(
            minPath=self.minPath,
            maxPath=self.maxPath,
            useHs=self.useHs,
            branchedPaths=self.branchedPaths,
            useBondOrder=self.useBondOrder,
            countSimulation=self.countSimulation,
            countBounds=self.countBounds,
            fpSize=self.fpSize,
            numBitsPerFeature=self.numBitsPerFeature,
            atomInvariantsGenerator=self.atomInvariantsGenerator,
        )

        return self


class MorganFingerprint(BaseFingerprint):
    def __init__(
        self,
        radius: int = 3,
        countSimulation: bool = False,
        includeChirality: bool = False,
        useBondTypes: bool = True,
        onlyNonzeroInvariants: bool = False,
        includeRingMembership: bool = True,
        countBounds=None,
        fpSize: int = 2048,
        atomInvariantsGenerator=None,
        bondInvariantsGenerator=None,
        includeRedundantEnvironments: bool = False,
    ):
        super().__init__()
        self.radius = radius
        self.countSimulation = countSimulation
        self.includeChirality = includeChirality
        self.useBondTypes = useBondTypes
        self.onlyNonzeroInvariants = onlyNonzeroInvariants
        self.includeRingMembership = includeRingMembership
        self.countBounds = countBounds
        self.fpSize = fpSize
        self.atomInvariantsGenerator = atomInvariantsGenerator
        self.bondInvariantsGenerator = bondInvariantsGenerator
        self.includeRedundantEnvironments = includeRedundantEnvironments

    def fit(self, X: list[Mol], y=None):
        # TODO: Change to MorganGenerator
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            countSimulation=self.countSimulation,
            includeChirality=self.includeChirality,
            useBondTypes=self.useBondTypes,
            onlyNonzeroInvariants=self.onlyNonzeroInvariants,
            includeRingMembership=self.includeRingMembership,
            countBounds=self.countBounds,
            fpSize=self.fpSize,
            atomInvariantsGenerator=self.atomInvariantsGenerator,
            bondInvariantsGenerator=self.bondInvariantsGenerator,
            includeRedundantEnvironments=self.includeRedundantEnvironments,
        )

        return self



class TopologicalTorsionFingerprint(BaseFingerprint):
    def __init__(
        self,
        includeChirality: bool = False,
        torsionAtomCount: int = 4,
        countSimulation: bool = True,
        countBounds=None,
        fpSize: int = 2048,
        atomInvariantsGenerator=None,
    ):
        super().__init__()
        self.includeChirality = includeChirality
        self.torsionAtomCount = torsionAtomCount
        self.countSimulation = countSimulation
        self.countBounds = countBounds
        self.fpSize = fpSize
        self.atomInvariantsGenerator = atomInvariantsGenerator

    def fit(self, X: list[Mol], y=None):
        self.fpgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            includeChirality=self.includeChirality,
            torsionAtomCount=self.torsionAtomCount,
            countSimulation=self.countSimulation,
            countBounds=self.countBounds,
            fpSize=self.fpSize,
            atomInvariantsGenerator=self.atomInvariantsGenerator,
        )

        return self


