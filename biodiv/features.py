from __future__ import annotations

import collections.abc as c
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import shapely
from loguru import logger

from biodiv.config import COMPUTE_VARIABLES, EPSG
from biodiv.dataset import (
    BiodiversityDataset,
    GeomorphometricVariable,
    compute_geomorphometric_variables,
)


def sample_raster_values(
    raster: Path,
    coords: c.Sequence[tuple[float, float]],
    sample_kwargs: dict | None = None,
) -> np.ndarray:
    with rio.open(raster) as src:
        return np.array(list(src.sample(coords, **sample_kwargs)))


def dissolve_dataset(
    biodiversity_dataset: BiodiversityDataset,
) -> gpd.GeoDataFrame:
    data = (
        biodiversity_dataset.data.groupby(by=['Latitude', 'Longitude'])
        .mean(numeric_only=True)
        .reset_index()
    )

    return gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(
            x=data.Longitude, y=data.Latitude, crs=4326
        ),
        crs=4326,
    )


def get_bbox(data: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    bounds = shapely.Polygon.from_bounds(*data.total_bounds).buffer(0.1).bounds
    return (bounds[0], bounds[2], bounds[1], bounds[3])


def get_features(compute_variables: int = COMPUTE_VARIABLES) -> pd.DataFrame:
    data = dissolve_dataset(BiodiversityDataset.read_data())
    if compute_variables:
        compute_geomorphometric_variables(get_bbox(data))
    data.to_crs(EPSG, inplace=True)
    coords = [(row.geometry.x, row.geometry.y) for row in data.itertuples()]
    for raster in GeomorphometricVariable._get_path_to_variables():
        sampled = sample_raster_values(
            raster, coords, sample_kwargs={'indexes': 1}
        )
        data[raster.stem] = sampled

    logger.info('Shape of features: %s/%s' % (data.shape[0], data.shape[1]))

    return data
