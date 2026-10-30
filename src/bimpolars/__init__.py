"""Building Information Models in polars DataFrames with geometry columns.

A :class:`BimDataFrame` holds one BIM model: a polars DataFrame with one row per element and
one model CRS, stored as PROJJSON. :func:`read_ifc` fills it from an IFC file and
:func:`read_speckle` from the v3 objects of a Speckle version (:func:`version_objects`),
:func:`simple_features` adds the plan footprint of every element, :func:`element_meshes_glb`
writes the meshes as glTF, and :func:`read_parquet` reads a model back from the (Geo)Parquet
that :meth:`BimDataFrame.write_parquet` wrote.

:func:`read_ifc` needs ifcopenshell, the extra ``bimpolars[ifc]``: ``bimpolars.ifc`` is imported
when ``read_ifc`` is first looked up, so the rest of the package works without it.
"""

from typing import TYPE_CHECKING, Any

from bimpolars.crs import LinearUnit, engineering_crs, linear_unit
from bimpolars.features import FEATURE_COLUMNS, FOOTPRINT_DTYPE, plan_outline, simple_features
from bimpolars.frame import (
    BBOX_DTYPE,
    MESH_DTYPE,
    NATIVE_GEOM_DTYPE,
    STYLE_DTYPE,
    SWEEP_DTYPE,
    BimDataFrame,
    read_parquet,
)
from bimpolars.gltf import element_meshes_glb, triangles_glb
from bimpolars.lightbim import LIGHTBIM_COLUMNS
from bimpolars.properties import properties_series
from bimpolars.speckle import read_speckle, version_objects

if TYPE_CHECKING:
    from bimpolars.ifc import read_ifc

__version__ = "0.1.0"

__all__ = [
    "BBOX_DTYPE",
    "FEATURE_COLUMNS",
    "FOOTPRINT_DTYPE",
    "LIGHTBIM_COLUMNS",
    "MESH_DTYPE",
    "NATIVE_GEOM_DTYPE",
    "STYLE_DTYPE",
    "SWEEP_DTYPE",
    "BimDataFrame",
    "LinearUnit",
    "element_meshes_glb",
    "engineering_crs",
    "linear_unit",
    "plan_outline",
    "properties_series",
    "read_ifc",
    "read_parquet",
    "read_speckle",
    "simple_features",
    "triangles_glb",
    "version_objects",
]


def __getattr__(name: str) -> Any:
    """Imports :mod:`bimpolars.ifc`, and so ifcopenshell, when ``read_ifc`` is first used."""
    if name == "read_ifc":
        from bimpolars.ifc import read_ifc

        return read_ifc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
