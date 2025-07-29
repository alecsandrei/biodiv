from __future__ import annotations

import geopandas as gpd

from biodiv.dataset import BiodiversityDataset


def test_read_biodiversity_dataset():
    data = BiodiversityDataset.read_data()
    assert isinstance(data, gpd.GeoDataFrame)
