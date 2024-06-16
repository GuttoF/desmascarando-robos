import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
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


class CatBoostOptunaPipeline:
    def __init__(self, feature_transformations: dict, n_trials: int = 50):
        self.feature_transformations = feature_transformations
        self.n_trials = n_trials
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
        self.best_params = None  # Adicionado para armazenar os melhores parâmetros

    def fit_transform(self, X: pd.DataFrame):
        transformers = []
        for transformer_name, features in self.feature_transformations.items():
            transformer = self.transformers[transformer_name]
            transformers.append((transformer_name, transformer, features))

        column_transformer = ColumnTransformer(
            transformers=transformers, remainder="passthrough"
        )

        X_transformed = column_transformer.fit_transform(X)
        return X_transformed, column_transformer

    def evaluate_model(
        self, model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5
    ):
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
        return scores

    def find_best_threshold(self, model, X_test, y_test):
        y_prob = model.predict_proba(X_test)[:, 1]
        thresholds = np.arange(0.4, 0.61, 0.01)
        f1_scores = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]
        best_threshold = thresholds[np.argmax(f1_scores)]
        return best_threshold

    def objective(self, trial, X_train, y_train, X_test, y_test):
        param = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "random_strength": trial.suggest_float(
                "random_strength", 1e-3, 10, log=True
            ),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "border_count": trial.suggest_int("border_count", 1, 255),
        }

        model = CatBoostClassifier(**param, verbose=0)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        return brier_score_loss(y_test, y_prob)

    def run_optimization(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_test, y_test),
            n_trials=self.n_trials,
        )

        self.best_params = study.best_params  # Armazena os melhores parâmetros
        return study.best_params

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        X_train_transformed, column_transformer = self.fit_transform(X_train)
        X_test_transformed = column_transformer.transform(X_test)

        best_params = self.run_optimization(
            X_train_transformed, y_train, X_test_transformed, y_test
        )
        best_model = CatBoostClassifier(**best_params)
        best_model.fit(X_train_transformed, y_train)

        best_threshold = self.find_best_threshold(
            best_model, X_test_transformed, y_test
        )
        scores = self.evaluate_model(
            best_model, X_test_transformed, y_test, best_threshold
        )
        scores["best_threshold"] = best_threshold
        scores["model"] = "CatBoostClassifier"
        self.results.append(scores)

        # Retorna os resultados e os melhores parâmetros
        return pd.DataFrame(self.results), self.best_params
