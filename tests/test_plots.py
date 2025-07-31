from __future__ import annotations

import numpy as np

from biodiv.config import RANDOM_SEED
from biodiv.plots import Diagnostic


def test_diagnostic_plots():
    n = 100
    rng = np.random.default_rng(RANDOM_SEED)
    actual = rng.normal(scale=1, size=n)
    print(actual.shape)
    eps = rng.normal(scale=0.25, size=n)
    predicted = actual + eps
    Diagnostic(actual, predicted).plot()
