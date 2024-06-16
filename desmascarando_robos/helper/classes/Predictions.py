import numpy as np
import pandas as pd
from category_encoders import HashingEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)


class Predictions:
    def __init__(self, feature_transformations: dict, model: BaseEstimator):
        self.feature_transformations = feature_transformations
        self.model = model
        self.transformers = {
            "log": FunctionTransformer(np.log1p, validate=True),
            "one_hot": OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            "hashing": HashingEncoder(),
            "ordinal": OrdinalEncoder(),
            "standard_scaler": StandardScaler(),
            "min_max_scaler": MinMaxScaler(),
            "robust_scaler": RobustScaler(),
        }
        self.column_transformer = None

    def fit_transform(self, X: pd.DataFrame):
        print("Starting fit_transform...")
        transformers = []
        for transformer_name, features in self.feature_transformations.items():
            transformer = self.transformers[transformer_name]
            transformers.append((transformer_name, transformer, features))
        print("Transformers configured.")

        self.column_transformer = ColumnTransformer(
            transformers=transformers, remainder="passthrough"
        )
        print("ColumnTransformer created.")

        X_transformed = self.column_transformer.fit_transform(X)
        print("fit_transform completed.")
        return X_transformed

    def transform(self, X: pd.DataFrame):
        print("Starting transform...")
        X_transformed = self.column_transformer.transform(X)
        print("transform completed.")
        return X_transformed

    def fit(self, X: pd.DataFrame, y: pd.Series):
        print(f"Training model {type(self.model).__name__}...")
        X_transformed = self.fit_transform(X)
        self.model.fit(X_transformed, y)
        print(f"Model {type(self.model).__name__} trained.")

    def predict_proba(self, X: pd.DataFrame):
        print(f"Predicting probabilities with model {type(self.model).__name__}...")
        X_transformed = self.transform(X)
        y_prob = self.model.predict_proba(X_transformed)[:, 1]
        print(f"Prediction completed with model {type(self.model).__name__}.")
        return y_prob

    def add_predictions(self, X: pd.DataFrame):
        print("Adding predictions to DataFrame...")
        X_copy = X.copy()
        X_transformed = self.transform(X_copy)
        y_prob = self.model.predict_proba(X_transformed)[:, 1]
        X_copy["predicao"] = y_prob
        print("Predictions added to DataFrame.")
        return X_copy
