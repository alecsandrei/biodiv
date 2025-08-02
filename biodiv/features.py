from __future__ import annotations

import collections.abc as c
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from loguru import logger

from biodiv.config import EPSG, INTERIM_DATA_DIR
from biodiv.dataset import BiodiversityDataset, GeomorphometricVariable


def sample_raster_values(
    raster: Path,
    coords: c.Sequence[tuple[float, float]],
    sample_kwargs: dict | None = None,
) -> np.ndarray:
    with rio.open(raster) as src:
        return np.array(list(src.sample(coords, **sample_kwargs)))


def get_features() -> pd.DataFrame:
    # data = (
    #    BiodiversityDataset.read_data()
    #    .data
    #    # .to_crs(EPSG)
    #    .groupby(by=['geometry'])
    #    .mean(numeric_only=True)
    #    .reset_index()
    # )
    # coords = [(row.geometry.x, row.geometry.y) for row in data.itertuples()]
    biodiversity_data = BiodiversityDataset.read_data()
    biodiversity_data.label_features()
    data = (
        biodiversity_data.data.groupby(by=['Latitude', 'Longitude'])
        .mean(numeric_only=True)
        .reset_index()
    )

    data = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(
            x=data.Longitude, y=data.Latitude, crs=4326
        ),
        crs=4326,
    )
    data = data.to_crs(EPSG)
    coords = [(row.geometry.x, row.geometry.y) for row in data.itertuples()]
    for raster in INTERIM_DATA_DIR.glob('*.tif'):
        # if raster.stem == 'dem':
        #    continue
        if raster.stem in GeomorphometricVariable._value2member_map_:
            sampled = sample_raster_values(
                raster, coords, sample_kwargs={'indexes': 1}
            )
            data[raster.stem] = sampled

    logger.info('Shape of features: %s/%s' % (data.shape[0], data.shape[1]))

    return data
