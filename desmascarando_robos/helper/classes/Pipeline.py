import warnings

import numpy as np
import pandas as pd
from category_encoders import HashingEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

warnings.filterwarnings("ignore")


class MLPipeline:
    def __init__(self, feature_transformations, models):
        self.feature_transformations = feature_transformations
        self.models = models
        self.transformers = {
            "log": FunctionTransformer(np.log1p, validate=True),
            "one_hot": OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            "hashing": HashingEncoder(),
            "ordinal": OrdinalEncoder(),
            "standard_scaler": StandardScaler(),
            "min_max_scaler": MinMaxScaler(),
            "robust_scaler": RobustScaler(),
        }
        self.results = []

    def fit_transform(self, X):
        print("Starting fit_transform...")
        transformers = []
        for transformer_name, features in self.feature_transformations.items():
            transformer = self.transformers[transformer_name]
            transformers.append((transformer_name, transformer, features))
        print("Transformers configured.")

        column_transformer = ColumnTransformer(
            transformers=transformers, remainder="passthrough"
        )
        print("ColumnTransformer created.")

        X_transformed = column_transformer.fit_transform(X)
        print("fit_transform completed.")
        return X_transformed, column_transformer

    def evaluate_model(self, model, X_test, y_test, threshold=0.5):
        print(f"Evaluating model {type(model).__name__}...")
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        scores = {
            "brier_score_loss": brier_score_loss(y_test, y_prob),
            "log_loss": log_loss(y_test, y_prob),
            "f1_score": f1_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        }
        print(f"Evaluation completed for model {type(model).__name__}.")
        return scores

    def run(self, X_train, y_train, X_test, y_test, threshold=0.5):
        print("Starting run...")
        X_train_transformed, column_transformer = self.fit_transform(X_train)
        X_test_transformed = column_transformer.transform(X_test)
        print("Data transformation completed.")

        for model in self.models:
            print(f"Training model {type(model).__name__}...")
            model.fit(X_train_transformed, y_train)
            scores = self.evaluate_model(model, X_test_transformed, y_test, threshold)
            scores["model"] = type(model).__name__
            self.results.append(scores)
            print(f"Model {type(model).__name__} trained and evaluated.")

        print("Run completed.")
        return pd.DataFrame(self.results)
    
    
    def test_transformations(X, feature_transformations):
        transformers = {
            'log': FunctionTransformer(np.log1p, validate=True),
            'one_hot': OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
            'hashing': HashingEncoder(),
            'ordinal': OrdinalEncoder(),
            'standard_scaler': StandardScaler(),
            'min_max_scaler': MinMaxScaler(),
            'robust_scaler': RobustScaler()
        }
        
        for transformer_name, features in feature_transformations.items():
            print(f"Testing {transformer_name} transformation on features: {features}")
            transformer = transformers[transformer_name]
            try:
                transformed = transformer.fit_transform(X[features])
                print(f"Transformation {transformer_name} succeeded. Output shape: {transformed.shape}")
            except Exception as e:
                print(f"Transformation {transformer_name} failed with error: {e}")
