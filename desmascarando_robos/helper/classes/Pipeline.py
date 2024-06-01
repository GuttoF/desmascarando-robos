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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)


class MLPipeline:
    def __init__(self, feature_transformations: dict, models: list):
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

    def fit_transform(self, X: pd.DataFrame):
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

    def evaluate_model(
        self,
        model: list,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5,
    ):
        print(f"Evaluating model {type(model).__name__}...")
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        scores = {
            "model": type(model).__name__,
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

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        print("Starting run...")
        X_train_transformed, column_transformer = self.fit_transform(X_train)
        X_test_transformed = column_transformer.transform(X_test)
        print("Data transformation completed.")

        for model in self.models:
            print(f"Training model {type(model).__name__}...")
            model.fit(X_train_transformed, y_train)
            scores = self.evaluate_model(model, X_test_transformed, y_test, threshold)
            self.results.append(scores)
            print(f"Model {type(model).__name__} trained and evaluated.")

        print("Run completed.")
        return pd.DataFrame(self.results)

    def run_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        print("Starting cross-validation run...")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        all_scores = []

        for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
            print(f"Processing fold {fold+1}/{n_splits}...")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            X_train_transformed, column_transformer = self.fit_transform(X_train)
            X_test_transformed = column_transformer.transform(X_test)
            print("Data transformation completed.")

            for model in self.models:
                print(f"Training model {type(model).__name__} on fold {fold+1}...")
                model.fit(X_train_transformed, y_train)
                scores = self.evaluate_model(
                    model, X_test_transformed, y_test, threshold
                )
                scores["fold"] = fold + 1
                all_scores.append(scores)
                print(
                    f"Model {type(model).__name__} trained and evaluated on fold {fold+1}."
                )

        print("Cross-validation run completed.")
        results_df = pd.DataFrame(all_scores)

        summary = results_df.groupby("model").agg(["mean", "std"])
        summary.columns = ["_".join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()

        return results_df, summary
