import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
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
    RobustScaler,
    StandardScaler,
)


class MLPipeline:
    """
    A class representing a machine learning pipeline.

    Parameters:
    - X (array-like): The input features.
    - y (array-like): The target variable.
    - models (list): A list of machine learning models.

    Attributes:
    - X (array-like): The input features.
    - y (array-like): The target variable.
    - models (list): A list of machine learning models.
    - pipelines (list): A list of pipelines built for each model.

    Methods:
    - build_pipeline: Builds the machine learning pipelines.
    - train_and_evaluate_models: Trains and evaluates the models in the pipelines.
    """

    def __init__(self, X, y, models):
        self.X = X
        self.y = y
        self.models = models
        self.pipelines = []

    def build_pipeline(
        self,
        log_list,
        ohe_list,
        robust_scaler_list=[],
        min_max_scaler_list=[],
        standard_scaler_list=[],
    ):
        """
        Builds the machine learning pipelines.

        Parameters:
        - log_list (list): A list of features to be log-transformed.
        - ohe_list (list): A list of features to be one-hot encoded.
        - robust_scaler_list (list, optional): A list of features to be scaled using RobustScaler. Defaults to [].
        - min_max_scaler_list (list, optional): A list of features to be scaled using MinMaxScaler. Defaults to [].
        - standard_scaler_list (list, optional): A list of features to be scaled using StandardScaler. Defaults to [].
        """
        preprocessing_steps = []
        if ohe_list:
            logging.info("One-hot encoding features: %s", ohe_list)
            preprocessing_steps.append(
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            )

        if log_list:
            logging.info("Log-transforming features: %s", log_list)
            preprocessing_steps.append(("logtransform", FunctionTransformer(np.log1p)))

        scalers = {
            "robustscaler": lambda: RobustScaler(with_centering=False),
            "minmaxscaler": MinMaxScaler,
            "standardscaler": StandardScaler,
        }
        for scaler_type, features in [
            ("robustscaler", robust_scaler_list),
            ("minmaxscaler", min_max_scaler_list),
            ("standardscaler", standard_scaler_list),
        ]:
            if features:
                logging.info("Scaling features %s using %s", features, scaler_type)
                preprocessing_steps.append((scaler_type, scalers[scaler_type]()))

        for model in self.models:
            self.pipelines.append(
                Pipeline(steps=[*preprocessing_steps, ("model", model)])
            )

    def train_and_evaluate_models(self, X_test, y_test, threshold=0.5):
        """
        Trains and evaluates the models in the pipelines.

        Parameters:
        - X_test (array-like): The test input features.
        - y_test (array-like): The test target variable.
        - threshold (float, optional): The classification threshold. Defaults to 0.5.

        Returns:
        - pandas.DataFrame: A DataFrame containing the evaluation metrics for each model.
        """
        metrics = []
        for pipeline in self.pipelines:
            model_name = type(pipeline.named_steps["model"]).__name__
            logging.info(f"Training the {model_name}...")
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
            }
            metrics.append(scores)

        return pd.DataFrame(metrics)


class MLPipelineCV(MLPipeline):
    def __init__(self, X, y, models):
        super().__init__(X, y, models)

    def train_and_evaluate_cv(self, threshold=0.5, verbose=True, kfold=5):
        """Evaluates models using cross-validation and returns a dataframe with metrics.

        Args:
            threshold (float, optional): Threshold value for classification. Defaults to 0.5.
            verbose (bool, optional): Whether to print progress information. Defaults to True.
            kfold (int, optional): Number of folds for cross-validation. Defaults to 5.

        Returns:
            pandas.DataFrame: Dataframe with evaluation metrics for each model.
        """
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
            ) = [], [], [], [], [], []

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

                # Collecting metrics
                acc_list.append(balanced_accuracy_score(y_val_fold, y_pred))
                precision_list.append(precision_score(y_val_fold, y_pred))
                recall_list.append(recall_score(y_val_fold, y_pred))
                f1_list.append(f1_score(y_val_fold, y_pred))
                roc_auc_list.append(roc_auc_score(y_val_fold, y_probs))
                log_loss_list.append(log_loss(y_val_fold, y_probs))

            # Aggregate results
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
            }
            metrics.append(scores)

        return pd.DataFrame(metrics)
