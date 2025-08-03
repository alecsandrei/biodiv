from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.figure import Figure
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

if t.TYPE_CHECKING:
    from pathlib import Path


@dataclass
class EDA:
    variable: pd.Series

    def plot(self, show: bool = False, out_file: Path | None = None) -> Figure:
        fig = plt.figure(figsize=(15, 5))

        hist_ax = fig.add_subplot(1, 3, 1)
        box_ax = fig.add_subplot(1, 3, 2)
        proportion_ax = fig.add_subplot(1, 3, 3)

        sns.histplot(self.variable, kde=True, ax=hist_ax)
        sns.boxplot(self.variable, width=0.4, ax=box_ax)
        sns.ecdfplot(self.variable, ax=proportion_ax)

        plt.tight_layout()
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        logger.info('Saved EDA plot at %s' % out_file)
        return fig


@dataclass
class RegressorDiagnostic:
    true: np.ndarray
    preds: np.ndarray
    name: str | None = None
    data: pd.DataFrame = field(init=False)

    def __post_init__(self):
        if self.true.ndim == 1:
            self.true = self.true.reshape(-1, 1)
        if self.preds.ndim == 1:
            self.preds = self.preds.reshape(-1, 1)
        self.data = pd.DataFrame(
            np.concatenate([self.true, self.preds], axis=1),
            columns=['Actual', 'Predicted'],
        )

    def log_results(self) -> None:
        metrics = (
            mean_squared_error,
            mean_absolute_error,
            root_mean_squared_error,
            r2_score,
        )
        for metric in metrics:
            result = metric(self.true, self.preds)
            if self.name is not None:
                logger.info(
                    '%s of the model is %.2f for %s'
                    % (metric.__name__, result, self.name)
                )
            else:
                logger.info('%s of the model is %.2f' % metric.__name__)

    def predicted_vs_actual(self, ax: plt.Axes):
        sns.scatterplot(
            self.data,
            x='Actual',
            y='Predicted',
            edgecolor='black',
            s=60,
            alpha=0.8,
            ax=ax,
        )
        ax.set_title('Predicted vs actual')

    def residuals_vs_actual(self, ax: plt.Axes):
        self.data['Residuals'] = self.true - self.preds
        sns.scatterplot(
            self.data,
            x='Actual',
            y='Residuals',
            edgecolor='black',
            s=60,
            alpha=0.8,
            ax=ax,
        )
        ax.set_title('Residuals vs actual')

    def plot(self, show: bool = False, out_file: Path | None = None) -> Figure:
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        self.predicted_vs_actual(ax1)
        self.residuals_vs_actual(ax2)
        ax1.set_box_aspect(1)
        ax2.set_box_aspect(1)
        plt.tight_layout()
        if out_file:
            logger.info('Saved regressor diagnostic plot at %s' % out_file)
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig
