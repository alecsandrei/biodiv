from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info('PROJ_ROOT path is: %s' % PROJ_ROOT)

DATA_DIR = PROJ_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
EXTERNAL_DATA_DIR = DATA_DIR / 'external'

MODELS_DIR = PROJ_ROOT / 'models'

REPORTS_DIR = PROJ_ROOT / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

EPSG = 6875
RANDOM_SEED = 0
BATHYMETRY_RESOLUTION = (500, 500)
SAGA_CMD = 'saga_cmd'  # path to saga_cmd file
COMPUTE_VARIABLES = bool(
    int(os.getenv('COMPUTE_VARIABLES', 1))
)  # Should be true when you run the first time
