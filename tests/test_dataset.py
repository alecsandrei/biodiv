from __future__ import annotations

import geopandas as gpd

from biodiv.dataset import Bathymetry, BiodiversityDataset


def test_read_biodiversity_dataset():
    data = BiodiversityDataset.read_data()
    assert isinstance(data, gpd.GeoDataFrame)


def test_bathymetry_fetch():
    Bathymetry.fetch(
        bbox=(9.943984127, 10.488734293, 43.128960089, 43.450396522)
    )
