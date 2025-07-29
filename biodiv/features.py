from __future__ import annotations

import collections.abc as c
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio as rio

from biodiv.config import INTERIM_DATA_DIR
from biodiv.dataset import BiodiversityDataset


def sample_raster_values(
    raster: Path,
    coords: c.Sequence[tuple[float, float]],
    sample_kwargs: dict | None = None,
) -> np.ndarray:
    with rio.open(raster) as src:
        return np.array(list(src.sample(coords, **sample_kwargs)))


def main() -> pd.DataFrame:
    data = (
        BiodiversityDataset.read_data()
        .data.groupby(by=['Latitude', 'Longitude'])
        .mean(numeric_only=True)
        .reset_index()
    )
    coords = [(row.Longitude, row.Latitude) for row in data.itertuples()]

    for raster in INTERIM_DATA_DIR.glob('*.tif'):
        sampled = sample_raster_values(
            raster, coords, sample_kwargs={'indexes': 1}
        )
        data[raster.stem] = sampled
    return data


if __name__ == '__main__':
    main()
