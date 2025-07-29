from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio as rio
from shapely import Polygon, box

from biodiv.features import sample_raster_values

FILE = Path(__file__).parent


def random_points_in_polygon(
    polygon: Polygon, number: int
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    minx, miny, maxx, maxy = polygon.bounds
    for _ in range(number):
        points.append(
            (np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        )
    return points


def test_sample_raster_values():
    raster = FILE / 'data/raster.tif'
    with rio.open(raster) as src:
        bounds = box(*src.bounds)
    n = 20
    points = random_points_in_polygon(bounds, n)
    sampled = sample_raster_values(raster, points, sample_kwargs={'indexes': 1})
    assert sampled.shape == (n, 1)
