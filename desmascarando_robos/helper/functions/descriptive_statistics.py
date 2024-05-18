from typing import Union

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

colors_list = ["#DE9776", "#9192B3", "#3D8221", "#823F21", "#BFC2FF", "#97D77D"]

def multiple_boxplots(
    data: Union[float, int, str], columns: list, colors: list = colors_list
) -> None:
    """
    Create multiple boxplots in subplots.

    Args:
        data (Union[float, int, str]): The data to be plotted.
        columns (list): A list of column names to be plotted.
        colors (list, optional): A list of colors for each boxplot. Defaults to None.

    Returns:
        None
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
        fig.add_trace(
            go.Box(y=data[column], name=column, marker_color=colors[i]),
            row=row,
            col=col,
        )

        col += 1
        if col > cols:
            col = 1
            row += 1

    fig.update_layout(
        title_text="Boxplots",
        height=200 * rows,
        showlegend=False,
    )

    fig.show()


def categorical_metrics(data: Union[int, str], col: str):
    """
    Shows the the absolute and percent values in categorical variables.

    Args:
        data ([dataframe]): [Insert all categorical attributes in the dataset]

    Returns:
        [dataframe]: [A dataframe with absolute and percent values]
    """

    return pd.DataFrame(
        {
            "absolute": data[col].value_counts(),
            "percent %": data[col].value_counts(normalize=True) * 100,
        }
    )
