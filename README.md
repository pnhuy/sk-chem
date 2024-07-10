# sk-chem: scikit-learn wrapper for cheminformatics

## Installation

```bash
pip install git+https://github.com/pnhuy/sk-chem
```

## Examples

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from sk_chem.features.fingerprint_features import AtomPairCountFingerprint, MorganFingerprint, RdkitFingerprint, TopologicalTorsionFingerprint
from sk_chem.features.rdkit_features import RdkitFeaturizer
from sk_chem.features.mordred_features import ModredFeaturizer
from sk_chem.forcefields.mmff import MMFF
from sk_chem.forcefields.uff import UFF
from sk_chem.molecules.rdkit_mol import RdkitMoleculeTransformer
from sk_chem.standardizers.rdkit_standardizer import RdkitMolStandardizer

DATA_URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"

df = pd.read_csv(DATA_URL)

X = df['smiles']
y = df['ESOL predicted log solubility in mols per litre']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params_grid = {
	'featurizer': [
		RdkitFeaturizer(),
		ModredFeaturizer(),
		AtomPairCountFingerprint(),
		RdkitFingerprint(),
		MorganFingerprint(),
		TopologicalTorsionFingerprint(),
	],
	'forcefield_optimizer': [
		MMFF(),
		UFF(),
	]
}

pipeline = Pipeline([
	('molecule_standardizer', RdkitMolStandardizer()),
	('molecule_transformer', RdkitMoleculeTransformer()),
	('forcefield_optimizer', MMFF()),
	('featurizer', RdkitFeaturizer()),
	('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
	('scaler', StandardScaler()),
	('model', RandomForestRegressor(random_state=42)),
])

grid_search = GridSearchCV(pipeline, params_grid, cv=5, n_jobs=-1, verbose=1)
grid_search = grid_search.fit(X_train, y_train)

best = grid_search.best_estimator_

y_pred = best.predict(X_test)

train_mse : float = mean_squared_error(y_train, best.predict(X_train))
test_mse : float = mean_squared_error(y_test, y_pred)

print("Best Parameters: ", grid_search.best_params_)
print(f"Train MSE: {train_mse}")
print(f"Test MSE: {test_mse}")
```