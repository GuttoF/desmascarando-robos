import logging

import numpy as np
import pandas as pd
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)


class MLPipeline:
    def __init__(self, X, y, models):
        self.X = X
        self.y = y
        self.models = models
        self.pipelines = []

    def build_pipeline(
        self,
        log_list=None,
        ohe_list=None,
        ordinal_list=None,
        robust_scaler_list=None,
        min_max_scaler_list=None,
        standard_scaler_list=None,
    ):
        log_list = log_list or []
        ohe_list = ohe_list or []
        ordinal_list = ordinal_list or []
        robust_scaler_list = robust_scaler_list or []
        min_max_scaler_list = min_max_scaler_list or []
        standard_scaler_list = standard_scaler_list or []
        transformers = []

        if log_list:
            logging.info("Log-transforming features: %s", log_list)
            transformers.append(("log", FunctionTransformer(np.log1p), log_list))

        if ohe_list:
            logging.info("One-hot encoding features: %s", ohe_list)
            transformers.append(
                (
                    "ohe",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ohe_list,
                )
            )

        if ordinal_list:
            logging.info("Ordinal encoding features: %s", ordinal_list)
            transformers.append(
                (
                    "ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                    ordinal_list,
                )
            )

        scalers = {
            "robustscaler": RobustScaler(with_centering=False),
            "minmaxscaler": MinMaxScaler(),
            "standardscaler": StandardScaler(),
        }

        for scaler_type, features in [
            ("robustscaler", robust_scaler_list),
            ("minmaxscaler", min_max_scaler_list),
            ("standardscaler", standard_scaler_list),
        ]:
            if features:
                logging.info("Scaling features %s using %s", features, scaler_type)
                transformers.append((scaler_type, scalers[scaler_type], features))

        preprocessor = ColumnTransformer(transformers, remainder="passthrough")

        for model in self.models:
            self.pipelines.append(
                Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            )

    def train_and_evaluate_models(self, X_test, y_test, threshold=0.5):
        metrics = []
        for pipeline in self.pipelines:
            model_name = type(pipeline.named_steps["model"]).__name__
            logging.info(f"Training the {model_name}...")

            # Check the shape of X and y before fitting
            logging.info(f"Shape of X: {self.X.shape}, Shape of y: {self.y.shape}")
            pipeline.fit(self.X, self.y)

            logging.info(f"Evaluating the {model_name}...")
            y_probs = pipeline.predict_proba(X_test)[:, 1]
            y_pred = (y_probs >= threshold).astype(int)

            scores = {
                "Model": model_name,
                "Threshold": threshold,
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred),
                "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
                "ROCAUC": roc_auc_score(y_test, y_probs),
                "Log Loss": log_loss(y_test, y_probs),
                "Brier Score": brier_score_loss(
                    y_test, y_probs
                ),  # Calcular Brier score
            }
            metrics.append(scores)

        return pd.DataFrame(metrics)


class MLPipelineCV(MLPipeline):
    def __init__(self, X, y, models):
        super().__init__(X, y, models)

    def train_and_evaluate_cv(self, threshold=0.5, verbose=True, kfold=5):
        metrics = []
        folds = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)

        for i, pipeline in enumerate(self.pipelines):
            model_name = type(pipeline.named_steps["model"]).__name__
            (
                acc_list,
                precision_list,
                recall_list,
                f1_list,
                roc_auc_list,
                log_loss_list,
                brier_score_list,  # Lista para armazenar os Brier scores
            ) = [], [], [], [], [], [], []

            if verbose:
                print(f"Folding model {i + 1}/{len(self.pipelines)} -> {model_name}")

            for train_index, val_index in folds.split(self.X, self.y):
                X_train_fold, X_val_fold = (
                    self.X.iloc[train_index],
                    self.X.iloc[val_index],
                )
                y_train_fold, y_val_fold = (
                    self.y.iloc[train_index],
                    self.y.iloc[val_index],
                )

                pipeline.fit(X_train_fold, y_train_fold)
                y_probs = pipeline.predict_proba(X_val_fold)[:, 1]
                y_pred = (y_probs >= threshold).astype(int)

                # Coletar métricas
                acc_list.append(balanced_accuracy_score(y_val_fold, y_pred))
                precision_list.append(precision_score(y_val_fold, y_pred))
                recall_list.append(recall_score(y_val_fold, y_pred))
                f1_list.append(f1_score(y_val_fold, y_pred))
                roc_auc_list.append(roc_auc_score(y_val_fold, y_probs))
                log_loss_list.append(log_loss(y_val_fold, y_probs))
                brier_score_list.append(brier_score_loss(y_val_fold, y_probs))

            # Agregar resultados
            scores = {
                "Model": model_name,
                "Threshold": threshold,
                "Balanced Accuracy Mean": np.mean(acc_list),
                "Balanced Accuracy STD": np.std(acc_list),
                "Precision Mean": np.mean(precision_list),
                "Precision STD": np.std(precision_list),
                "Recall Mean": np.mean(recall_list),
                "Recall STD": np.std(recall_list),
                "F1-Score Mean": np.mean(f1_list),
                "F1-Score STD": np.std(f1_list),
                "ROCAUC Mean": np.mean(roc_auc_list),
                "ROCAUC STD": np.std(roc_auc_list),
                "Log Loss Mean": np.mean(log_loss_list),
                "Log Loss STD": np.std(log_loss_list),
                "Brier Score Mean": np.mean(brier_score_list),
                "Brier Score STD": np.std(brier_score_list),
            }
            metrics.append(scores)

        return pd.DataFrame(metrics)
