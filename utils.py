from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
from gradio import Interface
from gradio.inputs import Slider, Radio
from gradio.outputs import JSON, Plot
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix


def build_rule_configurator(features_dict: Dict[str, pd.DataFrame],
                            targets_dict: Dict[str, pd.Series]) -> Interface:
    def eval_fn(a, b, c, d):
        return predict_and_evaluate(features_dict, targets_dict, a, b, c, d)

    return Interface(
        fn=eval_fn,
        inputs=[
            Slider(minimum=0.01, maximum=0.99, step=0.01, label="Maximum relative length difference"),
            Slider(minimum=0.01, maximum=0.99, step=0.01, label="Maximum relative number of question marks difference"),
            Slider(minimum=0.01, maximum=0.99, step=0.01, label="Minimum common words ratio"),
            Radio(choices=["Train", "Test"], label="Dataset name")
        ],
        outputs=[
            JSON(label="Evaluation metrics"),
            Plot(type="matplotlib", label="Confusion matrix")
        ],
        live=True,
        allow_flagging="never",
        examples=[[0.50, 0.99, 0.20, "Test"]]
    )


def predict_and_evaluate(features_dict: Dict[str, pd.DataFrame], targets_dict: Dict[str, pd.Series],
                         max_relative_length_diff: float, min_common_words_ratio: float, min_other_ratio: float,
                         dataset_name: str):
    y_true = targets_dict[dataset_name.lower()]
    y_pred = predict(features_dict[dataset_name.lower()],
                     max_relative_length_diff,
                     min_common_words_ratio,
                     min_other_ratio)
    return compute_scores(y_true, y_pred), compute_confusion_matrix(y_true, y_pred)


def predict(features: pd.DataFrame, max_relative_length_diff: float, max_relative_nb_question_marks_diff: float,
            min_common_words_ratio: float) -> pd.Series:
    y_pred = (
            (features["relative_length_diff"] < max_relative_length_diff) &
            (features["relative_nb_question_marks_diff"] < max_relative_nb_question_marks_diff) &
            (features["common_words_ratio"] > min_common_words_ratio)
    ).astype(int)
    return y_pred


def compute_scores(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, str]:
    return {
        "Accuracy": "{:.2%}".format(accuracy_score(y_true, y_pred)),
        "Recall": "{:.2%}".format(recall_score(y_true, y_pred)),
        "Precision": "{:.2%}".format(precision_score(y_true, y_pred)),
    }


def compute_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, dpi: int = 200, figsize: tuple = (5, 5)) -> Figure:
    cf_matrix = confusion_matrix(y_true, y_pred)

    vmin = np.min(cf_matrix)
    vmax = np.max(cf_matrix)
    off_diag_mask = np.eye(*cf_matrix.shape, dtype=bool)

    fig = plt.figure(clear=True, dpi=dpi, figsize=figsize)

    # True predictions
    ax = sns.heatmap(data=cf_matrix, annot=True, mask=~off_diag_mask, cmap='Greens', vmin=vmin, vmax=vmax,
                     cbar=False, square=True, fmt="n")

    # False predictions
    ax = sns.heatmap(data=cf_matrix, annot=True, mask=off_diag_mask, cmap='OrRd', vmin=vmin, vmax=vmax, cbar=False,
                     square=True, fmt="n", xticklabels=["Not duplicate", "Duplicate"],
                     yticklabels=["Not duplicate", "Duplicate"], ax=ax)

    ax.set_xlabel("Predictions", labelpad=10, fontsize="large", fontweight="bold")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    ax.set_ylabel("Reality", labelpad=10, fontsize="large", fontweight="bold")

    plt.close(fig)

    return fig
