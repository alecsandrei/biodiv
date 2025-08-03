from __future__ import annotations

import typing as t

import rasterio as rio

from biodiv.config import INTERIM_DATA_DIR
from biodiv.dataset import GeomorphometricVariable

if t.TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from xgboost import XGBRegressor


def predict(classifier: XGBRegressor, out_file: Path):
    df = GeomorphometricVariable._as_df()
    predictions = classifier.predict(df)
    save_predictions(predictions, out_file)


def save_predictions(predictions: np.ndarray, out_file: Path):
    dem = (
        INTERIM_DATA_DIR
        / f'{GeomorphometricVariable.DIGITAL_ELEVATION_MODEL.value}.tif'
    )
    with rio.open(dem) as src:
        rows = src.height
        cols = src.width
        predictions_reshaped = predictions.reshape(rows, cols)
        with rio.open(out_file, mode='w', **src.meta) as dst:
            dst.write(predictions_reshaped, 1)
