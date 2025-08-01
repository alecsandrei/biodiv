from __future__ import annotations

import collections.abc as c
import typing as t
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import shapely
from loguru import logger
from owslib.wcs import WebCoverageService
from PySAGA_cmd.saga import SAGA
from rasterio import warp
from rasterio.enums import Resampling

from biodiv.config import (
    EPSG,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
)

if t.TYPE_CHECKING:
    from os import PathLike

    from PySAGA_cmd.saga import Library, ToolOutput


@dataclass
class BiodiversityDataset:
    data: gpd.GeoDataFrame

    @classmethod
    def read_data(cls) -> t.Self:
        indices = pd.read_csv(RAW_DATA_DIR / 'biodiversity' / 'indices.csv')
        stations = pd.read_csv(RAW_DATA_DIR / 'biodiversity' / 'stations.csv')
        data = pd.merge(indices, stations, on='Station')
        return cls(
            gpd.GeoDataFrame(
                data,
                geometry=gpd.points_from_xy(data.Longitude, data.Latitude),
                crs=4326,
            )
        )


@dataclass
class Bathymetry:
    path: Path
    url: str = 'https://ows.emodnet-bathymetry.eu/wcs?SERVICE=WCS&REQUEST=GetCapabilities&VERSION=2.0.1'

    @classmethod
    def fetch(
        cls,
        bbox: tuple[float, float, float, float],
        out_file: Path,
    ) -> t.Self:
        wcs = WebCoverageService(cls.url)
        dataset_id = 'emodnet__mean'
        data = wcs.getCoverage(
            identifier=[dataset_id],
            format='image/tiff',
            timeout=120,
            subsets=[('Lat', bbox[2], bbox[3]), ('Long', bbox[0], bbox[1])],
        ).read()

        if out_file is not None:
            with open(out_file, 'wb') as outfile:
                outfile.write(data)
        # return cls(reproject(io.BytesIO(data), out_file, BATHYMETRY_RESOLUTION))

        return cls(out_file)


def reproject(
    src,
    dest: Path,
    resolution: tuple[float, float],
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    dst_crs = rio.crs.CRS.from_epsg(EPSG)
    with rio.open(src) as ds:
        src_crs = ds.crs
        arr = ds.read(1)
        meta = ds.meta.copy()
        newaff, width, height = warp.calculate_default_transform(
            src_crs,
            dst_crs,
            ds.width,
            ds.height,
            *ds.bounds,
            resolution=resolution,
        )
        newarr = np.empty((height, width), dtype=arr.dtype)
        warp.reproject(
            arr,
            newarr,
            src_transform=ds.transform,
            src_nodata=-9999,
            dst_nodata=-9999,
            dst_transform=newaff,
            width=width,
            height=height,
            src_crs=src_crs,
            dst_crs=f'EPSG:{EPSG}',
            resample=resampling,
            dst_resolution=resolution,
        )
        meta.update(
            {
                'transform': newaff,
                'width': width,
                'height': height,
                'crs': dst_crs,
            }
        )
        with rio.open(dest, mode='w', **meta) as dest_raster:
            dest_raster.write(newarr, 1)
    return dest


class GeomorphometricVariable(Enum):
    SLOPE = 'slope'
    HILLSHADE = 'shade'
    INDEX_OF_CONVERGENCE = 'ioc'
    TERRAIN_SURFACE_CONVEXITY = 'conv'
    POSITIVE_TOPOGRAPHIC_OPENNESS = 'poso'
    NEGATIVE_TOPOGRAPHIC_OPENNESS = 'nego'
    ASPECT = 'aspect'
    NORTHNESS = 'northness'
    EASTNESS = 'eastness'
    PROFILE_CURVATURE = 'cprof'
    PLAN_CURVATURE = 'cplan'
    GENERAL_CURVATURE = 'cgene'
    FLOW_LINE_CURVATURE = 'croto'
    TANGENTIAL_CURVATURE = 'ctang'
    LONGITUDINAL_CURVATURE = 'clong'
    CROSS_SECTIONAL_CURVATURE = 'ccros'
    MINIMAL_CURVATURE = 'cmini'
    MAXIMAL_CURVATURE = 'cmaxi'
    TOTAL_CURVATURE = 'ctota'
    DIGITAL_ELEVATION_MODEL = 'dem'
    REAL_SURFACE_AREA = 'area'
    TOPOGRAPHIC_POSITION_INDEX = 'tpi'
    VALLEY_DEPTH = 'vld'
    TERRAIN_RUGGEDNESS_INDEX = 'tri'
    VECTOR_RUGGEDNESS_MEASURE = 'vrm'
    LOCAL_CURVATURE = 'clo'
    UPSLOPE_CURVATURE = 'cup'
    LOCAL_UPSLOPE_CURVATURE = 'clu'
    DOWNSLOPE_CURVATURE = 'cdo'
    LOCAL_DOWNSLOPE_CURVATURE = 'cdl'
    FLOW_ACCUMULATION = 'flow'
    FLOW_PATH_LENGTH = 'fpl'
    SLOPE_LENGTH = 'spl'
    CELL_BALANCE = 'cbl'
    TOPOGRAPHIC_WETNESS_INDEX = 'twi'
    WIND_EXPOSITION_INDEX = 'wind'


class TerrainAnalysis:
    def __init__(
        self,
        dem: PathLike,
        saga: SAGA,
        verbose: bool,
        infer_obj_type: bool,
        ignore_stderr: bool,
        variables: c.Sequence[GeomorphometricVariable] | None = None,
    ):
        self.dem = Path(dem)
        self.saga = saga
        self.verbose = verbose
        self.infer_obj_type = infer_obj_type
        self.ignore_stderr = ignore_stderr
        self.variables = variables

        self.tools: list[c.Callable[..., ToolOutput | None]] = [
            self.analytical_hillshading,
            self.index_of_convergence,
            self.terrain_surface_convexity,
            self.topographic_openness,
            self.slope_aspect_curvature,
            self.real_surface_area,
            self.wind_exposition_index,
            self.topographic_position_index,
            self.valley_depth,
            self.terrain_ruggedness_index,
            self.vector_ruggedness_measure,
            self.upslope_and_downslope_curvature,
            self.flow_accumulation_parallelizable,
            self.flow_path_length,
            self.slope_length,
            self.cell_balance,
            self.topographic_wetness_index,
        ]

    def execute(self) -> c.Generator[tuple[str, ToolOutput | None]]:
        for tool in self.tools:
            yield (tool.__name__, tool())

    @property
    def morphometry(self) -> Library:
        return self.saga / 'ta_morphometry'

    @property
    def lighting(self) -> Library:
        return self.saga / 'ta_lighting'

    @property
    def channels(self) -> Library:
        return self.saga / 'ta_channels'

    @property
    def hydrology(self) -> Library:
        return self.saga / 'ta_hydrology'

    def get_out_path(self, variable: GeomorphometricVariable) -> Path | None:
        if self.should_compute(variable):
            return (INTERIM_DATA_DIR / variable.value).with_suffix('.tif')
        return None

    def should_compute(self, variable: GeomorphometricVariable) -> bool:
        if self.variables is None:
            return True
        elif variable in self.variables:
            return True
        return False

    def index_of_convergence(self) -> ToolOutput | None:
        """Requires 1 or 2 units of buffer depending on the neighbours parameter."""
        if not self.should_compute(
            GeomorphometricVariable.INDEX_OF_CONVERGENCE
        ):
            return None
        tool = self.morphometry / 'Convergence Index'
        return tool.execute(
            elevation=self.dem,
            method=0,
            neighbours=0,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
            result=self.get_out_path(
                GeomorphometricVariable.INDEX_OF_CONVERGENCE
            ),
        )

    def terrain_surface_convexity(self) -> ToolOutput | None:
        """Requires 1 unit of buffer."""
        if not self.should_compute(
            GeomorphometricVariable.TERRAIN_SURFACE_CONVEXITY
        ):
            return None
        tool = self.morphometry / 'Terrain Surface Convexity'
        return tool.execute(
            dem=self.dem,
            convexity=self.get_out_path(
                GeomorphometricVariable.TERRAIN_SURFACE_CONVEXITY,
            ),
            kernel=0,
            type=0,
            epsilon=0,
            scale=10,
            method=1,
            dw_weighting=3,
            dw_idw_power=2,
            dw_bandwidth=0.7,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def analytical_hillshading(self) -> ToolOutput | None:
        if not self.should_compute(GeomorphometricVariable.HILLSHADE):
            return None
        tool = self.lighting / 'Analytical Hillshading'
        return tool.execute(
            elevation=self.dem,
            method='5',
            shade=self.get_out_path(GeomorphometricVariable.HILLSHADE),
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def topographic_openness(self) -> ToolOutput | None:
        """Uses radius / resolution units of buffer."""
        if not any(
            self.should_compute(variable)
            for variable in (
                GeomorphometricVariable.POSITIVE_TOPOGRAPHIC_OPENNESS,
                GeomorphometricVariable.NEGATIVE_TOPOGRAPHIC_OPENNESS,
            )
        ):
            return None
        tool = self.lighting / 'Topographic Openness'
        kwargs = {
            'dem': self.dem,
            'radius': 100,
            'directions': 1,
            'direction': 315,
            'ndirs': 8,
            'method': 0,
            'dlevel': 3.0,
            'unit': 0,
            'nadir': 1,
            'verbose': self.verbose,
            'infer_obj_type': self.infer_obj_type,
            'ignore_stderr': self.ignore_stderr,
            'pos': self.get_out_path(
                GeomorphometricVariable.POSITIVE_TOPOGRAPHIC_OPENNESS
            ),
            'neg': self.get_out_path(
                GeomorphometricVariable.NEGATIVE_TOPOGRAPHIC_OPENNESS
            ),
        }
        return tool.execute(**kwargs)

    def slope_aspect_curvature(self) -> ToolOutput | None:
        """Requires 1 unit of buffer."""
        if not any(
            self.should_compute(variable)
            for variable in (
                GeomorphometricVariable.ASPECT,
                GeomorphometricVariable.NORTHNESS,
                GeomorphometricVariable.EASTNESS,
                GeomorphometricVariable.SLOPE,
                GeomorphometricVariable.GENERAL_CURVATURE,
                GeomorphometricVariable.PROFILE_CURVATURE,
                GeomorphometricVariable.PLAN_CURVATURE,
                GeomorphometricVariable.TANGENTIAL_CURVATURE,
                GeomorphometricVariable.LONGITUDINAL_CURVATURE,
                GeomorphometricVariable.CROSS_SECTIONAL_CURVATURE,
                GeomorphometricVariable.MINIMAL_CURVATURE,
                GeomorphometricVariable.MAXIMAL_CURVATURE,
                GeomorphometricVariable.TOTAL_CURVATURE,
                GeomorphometricVariable.FLOW_LINE_CURVATURE,
            )
        ):
            return None

        tool = self.morphometry / 'Slope, Aspect, Curvature'
        return tool.execute(
            elevation=self.dem,
            aspect=self.get_out_path(GeomorphometricVariable.ASPECT),
            northness=self.get_out_path(GeomorphometricVariable.NORTHNESS),
            eastness=self.get_out_path(GeomorphometricVariable.EASTNESS),
            slope=self.get_out_path(GeomorphometricVariable.SLOPE),
            c_gene=self.get_out_path(GeomorphometricVariable.GENERAL_CURVATURE),
            c_prof=self.get_out_path(GeomorphometricVariable.PROFILE_CURVATURE),
            c_plan=self.get_out_path(GeomorphometricVariable.PLAN_CURVATURE),
            c_tang=self.get_out_path(
                GeomorphometricVariable.TANGENTIAL_CURVATURE
            ),
            c_long=self.get_out_path(
                GeomorphometricVariable.LONGITUDINAL_CURVATURE
            ),
            c_cros=self.get_out_path(
                GeomorphometricVariable.CROSS_SECTIONAL_CURVATURE,
            ),
            c_mini=self.get_out_path(GeomorphometricVariable.MINIMAL_CURVATURE),
            c_maxi=self.get_out_path(GeomorphometricVariable.MAXIMAL_CURVATURE),
            c_tota=self.get_out_path(GeomorphometricVariable.TOTAL_CURVATURE),
            c_roto=self.get_out_path(
                GeomorphometricVariable.FLOW_LINE_CURVATURE
            ),
            method=6,
            unit_slope=0,
            unit_aspect=0,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def real_surface_area(self) -> ToolOutput | None:
        tool = self.morphometry / 'Real Surface Area'
        if not self.should_compute(GeomorphometricVariable.REAL_SURFACE_AREA):
            return None
        return tool.execute(
            dem=self.dem,
            area=self.get_out_path(GeomorphometricVariable.REAL_SURFACE_AREA),
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def wind_exposition_index(self) -> ToolOutput | None:
        if not self.should_compute(
            GeomorphometricVariable.WIND_EXPOSITION_INDEX
        ):
            return None
        tool = self.morphometry / 'Wind Exposition Index'
        return tool.execute(
            dem=self.dem,
            exposition=self.get_out_path(
                GeomorphometricVariable.WIND_EXPOSITION_INDEX
            ),
            maxdist=0.1,
            step=90,
            oldver=0,
            accel=1.5,
            pyramids=0,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def topographic_position_index(self) -> ToolOutput | None:
        """Uses radius_max / resolution units of buffer."""
        if not self.should_compute(
            GeomorphometricVariable.TOPOGRAPHIC_POSITION_INDEX
        ):
            return None
        tool = self.morphometry / 'Multi-Scale Topographic Position Index (TPI)'
        return tool.execute(
            dem=self.dem,
            tpi=self.get_out_path(
                GeomorphometricVariable.TOPOGRAPHIC_POSITION_INDEX
            ),
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def valley_depth(self) -> ToolOutput | None:
        if not self.should_compute(GeomorphometricVariable.VALLEY_DEPTH):
            return None
        tool = self.channels / 'Valley Depth'
        return tool.execute(
            elevation=self.dem,
            valley_depth=self.get_out_path(
                GeomorphometricVariable.VALLEY_DEPTH
            ),
            threshold=1,
            maxiter=0,
            nounderground=1,
            order=4,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def terrain_ruggedness_index(self) -> ToolOutput | None:
        """Uses radius units of buffer."""
        if not self.should_compute(
            GeomorphometricVariable.TERRAIN_RUGGEDNESS_INDEX
        ):
            return None
        tool = self.morphometry / 'Terrain Ruggedness Index (TRI)'
        return tool.execute(
            dem=self.dem,
            tri=self.get_out_path(
                GeomorphometricVariable.TERRAIN_RUGGEDNESS_INDEX,
            ),
            mode=1,
            radius=1,
            dw_weighting=0,
            dw_idw_power=2,
            dw_bandwidth=75,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def vector_ruggedness_measure(self) -> ToolOutput | None:
        """Uses radius units of buffer."""
        if not self.should_compute(
            GeomorphometricVariable.VECTOR_RUGGEDNESS_MEASURE
        ):
            return None
        tool = self.morphometry / 'Vector Ruggedness Measure (VRM)'
        return tool.execute(
            dem=self.dem,
            vrm=self.get_out_path(
                GeomorphometricVariable.VECTOR_RUGGEDNESS_MEASURE,
            ),
            mode=1,
            radius=1,
            dw_weighting=0,
            dw_idw_power=2,
            dw_bandwidth=75,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def upslope_and_downslope_curvature(self) -> ToolOutput | None:
        """Uses one unit of buffer."""
        if not any(
            self.should_compute(variable)
            for variable in (
                GeomorphometricVariable.LOCAL_CURVATURE,
                GeomorphometricVariable.UPSLOPE_CURVATURE,
                GeomorphometricVariable.LOCAL_UPSLOPE_CURVATURE,
                GeomorphometricVariable.DOWNSLOPE_CURVATURE,
                GeomorphometricVariable.LOCAL_DOWNSLOPE_CURVATURE,
            )
        ):
            return None
        tool = self.morphometry / 'Upslope and Downslope Curvature'
        return tool.execute(
            dem=self.dem,
            c_local=self.get_out_path(GeomorphometricVariable.LOCAL_CURVATURE),
            c_up=self.get_out_path(GeomorphometricVariable.UPSLOPE_CURVATURE),
            c_up_local=self.get_out_path(
                GeomorphometricVariable.LOCAL_UPSLOPE_CURVATURE
            ),
            c_down=self.get_out_path(
                GeomorphometricVariable.DOWNSLOPE_CURVATURE
            ),
            c_down_local=self.get_out_path(
                GeomorphometricVariable.LOCAL_DOWNSLOPE_CURVATURE,
            ),
            weighting=0.5,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def flow_accumulation_parallelizable(self) -> ToolOutput | None:
        if not self.should_compute(GeomorphometricVariable.FLOW_ACCUMULATION):
            return None
        tool = self.hydrology / 'Flow Accumulation (Parallelizable)'
        return tool.execute(
            dem=self.dem,
            flow=self.get_out_path(GeomorphometricVariable.FLOW_ACCUMULATION),
            update=0,
            method=2,
            convergence=1.1,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def flow_path_length(self) -> ToolOutput | None:
        if not self.should_compute(GeomorphometricVariable.FLOW_PATH_LENGTH):
            return None
        tool = self.hydrology / 'Flow Path Length'
        return tool.execute(
            elevation=self.dem,
            # seed=None,
            length=self.get_out_path(GeomorphometricVariable.FLOW_PATH_LENGTH),
            seeds_only=0,
            method=1,
            convergence=1.1,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def slope_length(self) -> ToolOutput | None:
        if not self.should_compute(GeomorphometricVariable.SLOPE_LENGTH):
            return None
        tool = self.hydrology / 'Slope Length'
        return tool.execute(
            dem=self.dem,
            length=self.get_out_path(GeomorphometricVariable.SLOPE_LENGTH),
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def cell_balance(self) -> ToolOutput | None:
        if not self.should_compute(GeomorphometricVariable.CELL_BALANCE):
            return None
        tool = self.hydrology / 'Cell Balance'
        return tool.execute(
            dem=self.dem,
            # weights=None,
            weights_default=1,
            balance=self.get_out_path(GeomorphometricVariable.CELL_BALANCE),
            method=1,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def topographic_wetness_index(self) -> ToolOutput | None:
        if not self.should_compute(
            GeomorphometricVariable.TOPOGRAPHIC_WETNESS_INDEX
        ):
            return None
        tool = self.hydrology / 'twi'
        return tool.execute(
            dem=self.dem,
            twi=self.get_out_path(
                GeomorphometricVariable.TOPOGRAPHIC_WETNESS_INDEX
            ),
            flow_method=0,
        )


if __name__ == '__main__':
    records = BiodiversityDataset.read_data()
    bounds = (
        shapely.Polygon.from_bounds(*records.data.total_bounds)
        .buffer(0.1)
        .bounds
    )
    bbox = (bounds[0], bounds[2], bounds[1], bounds[3])

    logger.info('bbox of biodiversity records: %s' % str(bbox))
    bathymetry = Bathymetry.fetch(
        bbox=bbox,
        out_file=INTERIM_DATA_DIR / 'dem.tif',
    )
    terrain_analysis = TerrainAnalysis(
        bathymetry.path,
        saga=SAGA(),
        verbose=True,
        infer_obj_type=False,
        ignore_stderr=True,
    )
    for tool in terrain_analysis.execute():
        logger.info('Computed %s geomorphometric variable' % tool[0])
