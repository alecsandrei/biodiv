from __future__ import annotations

import warnings

from loguru import logger

from biodiv import config

__all__ = ['config']

try:
    import matplotlib

    matplotlib.use('Qt5Agg')
except Exception as e:
    logger.warning(
        'Failed to import and configure matplotlib to use Qt6Agg. Error: %s' % e
    )

# Silences pkg_resources deprecation warning
warnings.simplefilter('ignore', DeprecationWarning)
