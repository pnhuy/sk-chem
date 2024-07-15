import torch
from sk_chem.features.deep_features.hf_transformer import HFTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

from tests.helper import get_classification_data

TRANSFORMER_MODEL = "seyonec/ChemBERTa-zinc-base-v1"


def test_hftransformer():
    X_train, X_test, y_train, y_test = get_classification_data()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer_featurizer = HFTransformer(
        TRANSFORMER_MODEL,
        device=device
    )

    pipeline = Pipeline(
        [
            ("transformer", transformer_featurizer),
            ("rf", RandomForestRegressor(n_jobs=-1)),
        ]
    )

    param_grid = {
        "rf__n_estimators": [100],
    }

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, n_jobs=1, verbose=2, error_score="raise"
    )
    grid_search.fit(X_train, y_train)

    best = grid_search.best_estimator_
    y_prob = best.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    assert auc > 0.5
    
    y_pred = [1 if i > 0.5 else 0 for i in y_prob]
    print(classification_report(y_test, y_pred))
