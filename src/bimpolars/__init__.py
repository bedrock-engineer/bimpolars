"""Building Information Models in polars DataFrames with geometry columns.

A :class:`BimDataFrame` holds one BIM model: a polars DataFrame with one row per element and
one model CRS, stored as PROJJSON. :func:`read_ifc` fills it from an IFC file,
:func:`simple_features` adds the plan footprint of every element, :func:`element_meshes_glb`
writes the meshes as glTF, and :func:`read_parquet` reads a model back from the (Geo)Parquet
that :meth:`BimDataFrame.write_parquet` wrote.
"""

from bimpolars.crs import LinearUnit, engineering_crs, linear_unit
from bimpolars.features import FEATURE_COLUMNS, FOOTPRINT_DTYPE, plan_outline, simple_features
from bimpolars.frame import BBOX_DTYPE, MESH_DTYPE, STYLE_DTYPE, BimDataFrame, read_parquet
from bimpolars.gltf import element_meshes_glb, triangles_glb
from bimpolars.ifc import read_ifc
from bimpolars.properties import properties_series

__version__ = "0.1.0"

__all__ = [
    "BBOX_DTYPE",
    "FEATURE_COLUMNS",
    "FOOTPRINT_DTYPE",
    "MESH_DTYPE",
    "STYLE_DTYPE",
    "BimDataFrame",
    "LinearUnit",
    "element_meshes_glb",
    "engineering_crs",
    "linear_unit",
    "plan_outline",
    "properties_series",
    "read_ifc",
    "read_parquet",
    "simple_features",
    "triangles_glb",
]
