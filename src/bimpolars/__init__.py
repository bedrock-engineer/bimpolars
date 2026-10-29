"""Building Information Models in polars DataFrames with geometry columns.

A :class:`BimDataFrame` holds one BIM model: a polars DataFrame with one row per element and
one model CRS, stored as PROJJSON. :func:`read_ifc` fills it from an IFC file and
:func:`read_parquet` reads one back from the Parquet that :meth:`BimDataFrame.write_parquet`
wrote.
"""

from bimpolars.crs import LinearUnit, engineering_crs, linear_unit
from bimpolars.frame import BBOX_DTYPE, MESH_DTYPE, STYLE_DTYPE, BimDataFrame, read_parquet
from bimpolars.ifc import read_ifc
from bimpolars.properties import properties_series

__version__ = "0.1.0"

__all__ = [
    "BBOX_DTYPE",
    "MESH_DTYPE",
    "STYLE_DTYPE",
    "BimDataFrame",
    "LinearUnit",
    "engineering_crs",
    "linear_unit",
    "properties_series",
    "read_ifc",
    "read_parquet",
]
