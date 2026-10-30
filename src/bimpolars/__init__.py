"""Building Information Models in polars DataFrames with geometry columns.

A :class:`BimDataFrame` holds one BIM model: a polars DataFrame with one row per element and
one model CRS, stored as PROJJSON and read as a ``pyproj.CRS``. :func:`read_ifc` fills it from
an IFC file and :func:`read_speckle` from the v3 objects of a Speckle version
(:func:`version_objects`), :func:`simple_features` adds the plan footprint of every element,
:func:`element_meshes_glb` writes the meshes as glTF, and :func:`read_parquet` reads a model
back from the (Geo)Parquet that :meth:`BimDataFrame.write_parquet` wrote.
:func:`derived_engineering_crs`, :func:`set_crs`, :func:`to_crs` and :func:`with_geog` (from
:mod:`bimpolars.geobim`) georeference a model, and :mod:`bimpolars.data` fetches the test
models.

:func:`read_ifc` needs ifcopenshell, the extra ``bimpolars[ifc]``: ``bimpolars.ifc`` is imported
when ``read_ifc`` is first looked up, so the rest of the package works without it.
"""

from typing import TYPE_CHECKING, Any

from bimpolars import data
from bimpolars.crs import CRSLike, LinearUnit, engineering_crs
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
from bimpolars.geobim import (
    affine_transformation,
    derived_engineering_crs,
    linearized_site_to_ecef,
    set_crs,
    to_crs,
    with_geog,
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
    "CRSLike",
    "LinearUnit",
    "affine_transformation",
    "data",
    "derived_engineering_crs",
    "element_meshes_glb",
    "engineering_crs",
    "linearized_site_to_ecef",
    "plan_outline",
    "properties_series",
    "read_ifc",
    "read_parquet",
    "read_speckle",
    "set_crs",
    "simple_features",
    "to_crs",
    "triangles_glb",
    "version_objects",
    "with_geog",
]


def __getattr__(name: str) -> Any:
    """Imports :mod:`bimpolars.ifc`, and so ifcopenshell, when ``read_ifc`` is first used."""
    if name == "read_ifc":
        from bimpolars.ifc import read_ifc

        return read_ifc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
