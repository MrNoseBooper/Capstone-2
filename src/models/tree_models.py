"""
Thin wrappers around scikit-learn RandomForestClassifier and
xgboost.XGBClassifier so both expose the same .fit / .predict_proba
interface used by evaluate.py.
"""
from sklearn.ensemble import RandomForestClassifier

import config


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(**config.RF_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    import xgboost as xgb  # imported lazily so the rest of the project
                            # still runs if xgboost isn't installed yet

    # Handle class imbalance explicitly via scale_pos_weight
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = max(n_neg / max(n_pos, 1), 1.0)

    params = dict(config.XGB_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model
