import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as ss
from plotly.subplots import make_subplots


class DataVisualizer:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataVisualizer with a DataFrame.

        Args:
        - data (pd.DataFrame): The DataFrame containing the data for visualization.
        """
        self.data = data

    colors_list = [
        "#1D5B79",
        "#1D7865",
        "#78621D",
        "#784F1D",
        "#1B2F38",
        "#75CDBB",
        "#CDB875",
        "#CDA675",
        "#B7E2F7",
        "#E4FFF9",
    ]

    def multiple_distplots(self, columns: list) -> None:
        """
        Create multiple distribution plots (distplots) with histogram and KDE on the same scale.

        Args:
        - columns (list): A list of column names to be plotted.
        - colors (list, optional): A list of colors for each distribution plot. Defaults to None.
        """
        num_columns = len(columns)
        cols = int(num_columns**0.5)
        if cols * cols >= num_columns:
            rows = cols
        else:
            rows = cols + 1
            if cols * rows < num_columns:
                cols += 1

        fig = make_subplots(rows=rows, cols=cols, subplot_titles=columns)
        row, col = 1, 1

        for i, column in enumerate(columns):
            color_index = i % len(self.colors_list)
            column_data = self.data[column]
            fig.add_trace(
                go.Histogram(
                    x=column_data,
                    name=column + " Histogram",
                    nbinsx=30,
                    opacity=0.75,
                    marker_color=self.colors_list[color_index],
                    histnorm="probability density",
                ),
                row=row,
                col=col,
            )

            kde = ss.gaussian_kde(column_data)
            kde_x = np.linspace(column_data.min(), column_data.max(), 500)
            kde_y = kde(kde_x)

            fig.add_trace(
                go.Scatter(
                    x=kde_x,
                    y=kde_y,
                    mode="lines",
                    name=column + " KDE",
                    line=dict(color=self.colors_list[color_index], width=2),
                ),
                row=row,
                col=col,
            )

            col += 1
            if col > cols:
                col = 1
                row += 1

        fig.update_layout(
            title_text="Distribution Plots", height=200 * rows, showlegend=False
        )
        fig.show()

    def distribution_analysis(self, columns: list) -> None:
        """
        Create multiple distribution plots (distplots) and boxplots for the given columns.

        Args:
        - columns (list): A list of column names to be plotted.
        """
        cols = 2  # two columns per row, one for distplot and one for boxplot
        rows = len(columns)

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"{col}" for col in columns for _ in (0, 1)],
        )

        row = 1

        for i, column in enumerate(columns):
            color_index = i % len(self.colors_list)
            column_data = self.data[column]

            # Distribution Plot
            fig.add_trace(
                go.Histogram(
                    x=column_data,
                    name="Histogram",
                    nbinsx=30,
                    opacity=0.75,
                    marker_color=self.colors_list[color_index],
                    histnorm="probability density",
                ),
                row=row,
                col=1,
            )

            kde = ss.gaussian_kde(column_data)
            kde_x = np.linspace(column_data.min(), column_data.max(), 500)
            kde_y = kde(kde_x)

            fig.add_trace(
                go.Scatter(
                    x=kde_x,
                    y=kde_y,
                    mode="lines",
                    name=f"{column} KDE",
                    line=dict(color=self.colors_list[color_index], width=2),
                ),
                row=row,
                col=1,
            )

            # Boxplot
            fig.add_trace(
                go.Box(
                    x=column_data,
                    name="",
                    marker_color=self.colors_list[color_index],
                ),
                row=row,
                col=2,
            )

            row += 1

        fig.update_layout(
            title_text="Distribution and Boxplot Analysis",
            height=250 * rows,
            showlegend=False,
        )
        fig.show()

    def multiple_barplots(self, columns: list) -> None:
        """
        Generate multiple bar plots for specified columns in the DataFrame.

        Args:
        - columns (list): A list of column names to be plotted.
        """
        num_columns = len(columns)
        cols = int(num_columns**0.5)
        rows = (num_columns + cols - 1) // cols

        fig = make_subplots(rows=rows, cols=cols, subplot_titles=columns)

        index = 0
        for row in range(1, rows + 1):
            for col in range(1, cols + 1):
                if index < num_columns:
                    column_data = self.data[columns[index]].value_counts().reset_index()
                    column_data.columns = ["category", "count"]

                    fig.add_trace(
                        go.Bar(
                            x=column_data["category"],
                            y=column_data["count"],
                            name=columns[index],
                            marker_color=self.colors_list[
                                index % len(self.colors_list)
                            ],
                        ),
                        row=row,
                        col=col,
                    )
                index += 1

        fig.update_layout(title_text="Barplots", height=200 * rows, showlegend=False)
        fig.show()

    def correlation_heatmap(self, columns: list) -> None:
        """
        Generate a heatmap for the correlation matrix of specified columns in the DataFrame.
        """
        corr_matrix = self.data[columns].corr()
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="Greys",
                colorbar=dict(title="Correlation Coefficient"),
            )
        )

        annotations = []
        for i, row in enumerate(corr_matrix.values):
            for j, value in enumerate(row):
                annotations.append(
                    go.layout.Annotation(
                        x=corr_matrix.columns[j],
                        y=corr_matrix.columns[i],
                        text=str(round(value, 2)),
                        showarrow=False,
                        font=dict(color="black"),
                    )
                )

        fig.update_layout(
            title_text="Correlation Heatmap",
            height=1200,
            width=1200,
            annotations=annotations,
        )
        fig.show()

    def scatter_plot_matrix(self, columns: list, color_column=None) -> None:
        """
        Generate a scatter plot matrix for specified columns, optionally colored by another column.
        """
        if color_column:
            fig = px.scatter_matrix(
                self.data, dimensions=columns, color=self.data[color_column]
            )
        else:
            fig = px.scatter_matrix(
                self.data,
                dimensions=columns,
                color_discrete_sequence=self.colors_list,
            )

        fig.update_layout(
            title_text="Scatter Plot Matrix",
            height=1000,
            width=1000,
            xaxis_tickangle=-45,
            yaxis_tickangle=-45,
        )
        fig.show()

    def cramers_v(self, x, y):
        """
        Calculate Cramér's V statistic for categorical-categorical association.
        """
        confusion_matrix = pd.crosstab(x, y)
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        r_corr = r - ((r - 1) ** 2) / (n - 1)
        k_corr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2_corr / min((k_corr - 1), (r_corr - 1)))

    def categorical_heatmap(self, columns: list) -> None:
        """
        Generate a heatmap for the association between categorical columns.
        """
        cramer_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
        for col1 in columns:
            for col2 in columns:
                if col1 == col2:
                    cramer_matrix.loc[col1, col2] = 1.0
                else:
                    cramer_matrix.loc[col1, col2] = self.cramers_v(
                        self.data[col1], self.data[col2]
                    )

        fig = go.Figure(
            data=go.Heatmap(
                z=cramer_matrix.values,
                x=cramer_matrix.columns,
                y=cramer_matrix.columns,
                colorscale="Greys",
                colorbar=dict(title="Cramér's V"),
            )
        )

        annotations = []
        for i, row in enumerate(cramer_matrix.values):
            for j, value in enumerate(row):
                annotations.append(
                    go.layout.Annotation(
                        x=cramer_matrix.columns[j],
                        y=cramer_matrix.columns[i],
                        text=str(round(value, 2)),
                        showarrow=False,
                        font=dict(color="black"),
                    )
                )

        fig.update_layout(
            title_text="Cramer's Heatmap",
            height=1200,
            width=1200,
            annotations=annotations,
        )
        fig.show()

    def statistic_test(self, featureA: str, featureB: str) -> None:
        """
        Perform a chi-square test of independence between two categorical variables.

        Parameters:
        - featureA: str
            The name of the first categorical variable.
        - featureB: str
            The name of the second categorical variable.

        Returns:
        None

        Prints the chi-square statistic, p-value, and the result of the hypothesis test.
        """
        table = pd.crosstab(self.data[featureA], self.data[featureB])
        result = ss.chi2_contingency(table)
        print(
            f"Chi2 Statistic: {round(result.statistic, 3)}\nP-value: {round(result.pvalue, 3)}"
        )

        if result.pvalue < 0.05:
            print("Reject the null hypothesis: The variables are dependent")
        else:
            print("Fail to reject the null hypothesis: The variables are independent")
