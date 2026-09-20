import numpy as np
import pandas as pd


MODELS = ("Logistic Regression", "Random Forest", "XGBoost", "SVM (RBF)", "Stacking Ensemble")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keyed(path, keys, expected=None):
    frame = pd.read_csv(path)
    require(not frame.empty and not frame.isna().any().any(), f"{path.name}: empty or missing values")
    require(not frame.duplicated(keys).any(), f"{path.name}: duplicate keys")
    frame = frame.set_index(keys)
    if expected is not None:
        require(set(frame.index) == set(expected), f"{path.name}: incomplete or unexpected keys")
    return frame


def predictions(frame, context):
    require(set(frame.y_true) == {0, 1}, f"{context}: invalid binary outcomes")
    require(np.isfinite(frame.y_prob).all() and frame.y_prob.between(0, 1).all(),
            f"{context}: invalid probabilities")
