from __future__ import annotations

from pathlib import Path

import pandas as pd

from biodiv.dataset import (
    Bathymetry,
    BiodiversityDataset,
    GeomorphometricVariable,
)


def test_read_biodiversity_dataset():
    data = BiodiversityDataset.read_data().data
    assert isinstance(data, pd.DataFrame)


def test_bathymetry_fetch(tmp_path: Path):
    Bathymetry.fetch(
        bbox=(9.943984127, 10.488734293, 43.128960089, 43.450396522),
        out_file=tmp_path / 'bathymetry.tif',
    )


def test_geomorphometric_variables():
    variables = GeomorphometricVariable._get_path_to_variables()
    df = GeomorphometricVariable._as_df()
    assert df.shape[1] == len(variables)
