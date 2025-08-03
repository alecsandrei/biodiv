from __future__ import annotations

import typing as t

import numpy as np

from biodiv.config import RANDOM_SEED
from biodiv.dataset import BiodiversityDataset
from biodiv.plots import EDA, RegressorDiagnostic

if t.TYPE_CHECKING:
    from pathlib import Path


def test_diagnostic_plots(tmp_path: Path):
    out_file = tmp_path / 'regressor_diagnostic.png'
    n = 100
    rng = np.random.default_rng(RANDOM_SEED)
    actual = rng.normal(scale=1, size=n)
    eps = rng.normal(scale=0.25, size=n)
    predicted = actual + eps
    RegressorDiagnostic(actual, predicted).plot(show=False, out_file=out_file)


def test_eda_plots(tmp_path: Path):
    out_file = tmp_path / 'margalef_eda.png'
    data = BiodiversityDataset.read_data()
    eda = EDA(data.margalef)
    eda.plot(show=False, out_file=out_file)
    assert out_file.exists()
