import pandas as pd
from sklearn.model_selection import train_test_split

from tests import ESOL_URL, CLINTOX_URL


def get_regression_data():
    df = pd.read_csv(ESOL_URL)
    X = df["smiles"]
    y = df["ESOL predicted log solubility in mols per litre"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test


def get_classification_data():
    df = pd.read_csv(CLINTOX_URL)
    X = df["smiles"]
    y = df["FDA_APPROVED"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test
