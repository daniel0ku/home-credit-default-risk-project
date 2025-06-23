# custom_transformers.py
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

class CorrelationFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.features_to_keep_ = None
        self.importances_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = X.columns

        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        model = LGBMClassifier(is_unbalance=True, metric='auc', objective='binary')
        model.fit(X, y)
        importances = dict(zip(X.columns, model.feature_importances_))
        self.importances_ = importances

        to_drop = set()

        for col in upper.columns:
            for row in upper.index:
                if upper.loc[row, col] > self.threshold:
                    if importances[row] < importances[col]:
                        to_drop.add(row)
                    else:
                        to_drop.add(col)

        self.features_to_keep_ = [col for col in X.columns if col not in to_drop]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        return X[self.features_to_keep_]