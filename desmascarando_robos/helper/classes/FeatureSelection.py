import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)


class FeatureSelection(BaseEstimator, TransformerMixin):
    """
    FeatureSelection class for selecting and transforming features using various transformers and a RandomForest model.

    Parameters:
    ----------
    log_list : list
        List of column names to apply logarithmic transformation.
    ohe_list : list
        List of column names to apply one-hot encoding.
    standard_scaler_list : list
        List of column names to apply standard scaling.
    minmax_list : list
        List of column names to apply min-max scaling.
    robust_scaler_list : list
        List of column names to apply robust scaling.

    Attributes:
    ----------
    model_rf : RandomForestClassifier
        Random Forest model used for feature selection.
    feature_names : list
        List of feature names after transformation.

    Methods:
    -------
    fit(X_train, y_train)
        Fit the FeatureSelection model on the training data.
    transform(X)
        Transform the input data using the fitted transformers.
    plot_feature_importances()
        Plot the feature importances based on the fitted RandomForest model.
    """

    def __init__(
        self, log_list, ohe_list, standard_scaler_list, minmax_list, robust_scaler_list
    ):
        self.log_list = log_list
        self.ohe_list = ohe_list
        self.standard_scaler_list = standard_scaler_list
        self.minmax_list = minmax_list
        self.robust_scaler_list = robust_scaler_list
        self.model_rf = RandomForestClassifier()
        self.feature_names = None

    def fit(self, X_train, y_train):
        transformers = [
            ("log", FunctionTransformer(np.log1p), self.log_list),
            ("ohe", OneHotEncoder(), self.ohe_list),
            ("std_scaler", StandardScaler(), self.standard_scaler_list),
            ("minmax_scaler", MinMaxScaler(), self.minmax_list),
            ("robust_scaler", RobustScaler(), self.robust_scaler_list),
        ]
        if transformers == []:
            logging.warning("No transformers were passed")
            return self

        logging.info("Fitting ColumnTransformer")
        self.col_transform = ColumnTransformer(transformers, remainder="passthrough")
        self.col_transform.fit(X_train, y_train)

        self.feature_names = self._generate_feature_names()

        X_transformed = self.col_transform.transform(X_train)
        self.feature_names = self._generate_feature_names()
        logging.info("Fitting Model")
        self.model_rf.fit(X_transformed, y_train)

        return self

    def _generate_feature_names(self):
        feature_names = []
        for name, transformer, column in self.col_transform.transformers_:
            if hasattr(transformer, "get_feature_names_out"):
                if isinstance(transformer, OneHotEncoder):
                    feature_names.extend(transformer.get_feature_names_out(column))
                else:
                    transformed_names = transformer.get_feature_names_out()
                    if name == "log":
                        transformed_names = [
                            "log_" + feature for feature in column
                        ]  # Prepend "log_" for log transformed features
                    feature_names.extend(transformed_names)
            else:
                if name == "log":
                    feature_names.extend(
                        ["log_" + feature for feature in column]
                    )  # Ensure log features are correctly named
                else:
                    feature_names.extend(column)
        return feature_names

    def transform(self, X):
        return self.col_transform.transform(X)

    def plot_feature_importances(self):
        logging.info("Plotting feature importances")
        importances = self.model_rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = [self.feature_names[i] for i in indices]

        fig, ax = plt.subplots(figsize=(13, 8))
        plt.grid(True, which="major", axis="y", color="#DAD8D7", alpha=0.5, zorder=1)
        plt.xticks(range(len(importances)), sorted_features, rotation="vertical")
        plt.xlabel("Features")
        ax.yaxis.tick_right()
        ax.spines[["top", "left", "bottom", "right"]].set_visible(False)
        ax.spines["right"].set_linewidth(1.1)
        ax.spines["right"].set_color("#DAD8D7")
        ax.plot(
            [0.12, 0.9],
            [0.98, 0.98],
            transform=fig.transFigure,
            clip_on=False,
            color="#1D7865",
            linewidth=0.6,
        )
        ax.add_patch(
            plt.Rectangle(
                (0.12, 0.98),
                0.04,
                -0.02,
                facecolor="#1D7865",
                transform=fig.transFigure,
                clip_on=False,
                linewidth=0,
            )
        )
        plt.text(
            x=0.12,
            y=0.93,
            s="Feature Importance",
            transform=fig.transFigure,
            ha="left",
            fontsize=14,
            weight="bold",
            alpha=0.8,
        )
        ax.text(
            x=0.12,
            y=0.90,
            s="Using RandomForest to select the most important features",
            transform=fig.transFigure,
            ha="left",
            fontsize=12,
            alpha=0.8,
        )
        plt.subplots_adjust(
            left=None, bottom=0.2, right=None, top=0.85, wspace=None, hspace=None
        )
        plt.bar(
            range(len(importances)),
            importances[indices],
            color="#1D7865",
            align="center",
            label="Important",
        )
        plt.show()
