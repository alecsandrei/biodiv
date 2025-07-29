from __future__ import annotations

import typing as t
from dataclasses import dataclass

import pandas as pd

from biodiv.config import RAW_DATA_DIR


@dataclass
class BiodiversityRecords:
    data: pd.DataFrame

    @classmethod
    def read_data(cls) -> t.Self:
        return cls(
            pd.read_parquet(
                RAW_DATA_DIR / 'eurobis_obisenv_view_2025-03-20.parquet'
            )
        )


if __name__ == '__main__':
    records = BiodiversityRecords.read_data()
    breakpoint()
