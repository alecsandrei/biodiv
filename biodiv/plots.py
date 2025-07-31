from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


@dataclass
class Diagnostic:
    true: np.ndarray
    preds: np.ndarray
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

    def plot(self, show: bool = False) -> Figure:
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        self.predicted_vs_actual(ax1)
        self.residuals_vs_actual(ax2)
        ax1.set_box_aspect(1)
        ax2.set_box_aspect(1)
        plt.tight_layout()
        if show:
            plt.show()
        return fig
