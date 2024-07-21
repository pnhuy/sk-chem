from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.pipeline import Pipeline
from openbabel import openbabel

from sk_chem.models.elembert import (
    ElementTokenizer,
    ElemBertClassifier,
    ElemBertRegressor,
)
from .helper import get_classification_data, get_regression_data

openbabel.obErrorLog.StopLogging()


def test_elembertclassifier():
    X_train, X_test, y_train, y_test = get_classification_data()

    label_encoder = OneHotEncoder()
    y_train = label_encoder.fit_transform(y_train.to_numpy().reshape(-1, 1)).toarray()
    y_test = label_encoder.transform(y_test.to_numpy().reshape(-1, 1)).toarray()

    pipe = Pipeline(
        [
            ("tokenizer", ElementTokenizer()),
            ("classifier", ElemBertClassifier()),
        ]
    )

    pipe.fit(X_train, y_train)
    pred_labels = pipe.predict(X_test)

    true_labels = label_encoder.inverse_transform(y_test)

    print(classification_report(true_labels, pred_labels))


def test_elembertregressor():
    X_train, X_test, y_train, y_test = get_regression_data()

    pipe = Pipeline(
        [
            ("tokenizer", ElementTokenizer()),
            ("regressor", ElemBertRegressor()),
        ]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("MSE:", mean_squared_error(y_test, y_pred))
