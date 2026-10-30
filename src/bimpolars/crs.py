"""Model coordinate reference systems as PROJJSON, without pyproj.

A BIM model lives in its own Cartesian coordinates. Until it is georeferenced, that is a plain
`EngineeringCRS <https://proj.org/en/stable/specifications/projjson.html>`_: three axes in the
model's length unit, without a relation to the earth. The core keeps every CRS as a PROJJSON
``dict`` so that it needs no pyproj; the ``[geobim]`` extra turns it into a ``pyproj.CRS``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

PROJJSON_SCHEMA = "https://proj.org/schemas/v0.7/projjson.schema.json"

CRS84H: dict[str, Any] = {
    "$schema": PROJJSON_SCHEMA,
    "type": "GeographicCRS",
    "name": "WGS 84 longitude-latitude-height",
    "datum_ensemble": {
        "name": "World Geodetic System 1984 ensemble",
        "members": [
            {
                "name": "World Geodetic System 1984 (Transit)",
                "id": {"authority": "EPSG", "code": 1166},
            },
            {
                "name": "World Geodetic System 1984 (G730)",
                "id": {"authority": "EPSG", "code": 1152},
            },
            {
                "name": "World Geodetic System 1984 (G873)",
                "id": {"authority": "EPSG", "code": 1153},
            },
            {
                "name": "World Geodetic System 1984 (G1150)",
                "id": {"authority": "EPSG", "code": 1154},
            },
            {
                "name": "World Geodetic System 1984 (G1674)",
                "id": {"authority": "EPSG", "code": 1155},
            },
            {
                "name": "World Geodetic System 1984 (G1762)",
                "id": {"authority": "EPSG", "code": 1156},
            },
            {
                "name": "World Geodetic System 1984 (G2139)",
                "id": {"authority": "EPSG", "code": 1309},
            },
            {
                "name": "World Geodetic System 1984 (G2296)",
                "id": {"authority": "EPSG", "code": 1383},
            },
        ],
        "ellipsoid": {
            "name": "WGS 84",
            "semi_major_axis": 6378137,
            "inverse_flattening": 298.257223563,
        },
        "accuracy": "2.0",
        "id": {"authority": "EPSG", "code": 6326},
    },
    "coordinate_system": {
        "subtype": "ellipsoidal",
        "axis": [
            {
                "name": "Geodetic longitude",
                "abbreviation": "Lon",
                "direction": "east",
                "unit": "degree",
            },
            {
                "name": "Geodetic latitude",
                "abbreviation": "Lat",
                "direction": "north",
                "unit": "degree",
            },
            {
                "name": "Ellipsoidal height",
                "abbreviation": "h",
                "direction": "up",
                "unit": "metre",
            },
        ],
    },
    "id": {"authority": "OGC", "code": "CRS84h"},
}
"""OGC:CRS84h as PROJJSON: longitude, latitude (degrees) and ellipsoidal height (metres) on WGS 84.

The CRS of the ``geog`` column that :func:`bimpolars.geobim.with_geog` adds, and what GeoJSON
means by its coordinates (RFC 7946: longitude, latitude and an elevation in metres above the
WGS 84 ellipsoid). It is written out here, as ``pyproj.CRS("OGC:CRS84h").to_json_dict()`` gives
it (PROJ 9.8), so that the core can label the column without pyproj.
"""

LINEAR_UNITS: dict[str, float] = {
    "metre": 1.0,
    "millimetre": 0.001,
    "centimetre": 0.01,
    "decimetre": 0.1,
    "kilometre": 1000.0,
    "foot": 0.3048,
    "inch": 0.0254,
    "US survey foot": 1200 / 3937,
}
"""Conversion factors to the metre of the linear units ``engineering_crs`` knows by name."""


class LinearUnit(NamedTuple):
    """A length unit of a CRS axis.

    Attributes:
        name: The unit's name, such as ``"millimetre"``.
        conversion_factor: How many metres one unit is.
    """

    name: str
    conversion_factor: float


def engineering_crs(
    name: str = "Model coordinates",
    *,
    unit: str | LinearUnit = "metre",
    datum: str = "Model origin",
) -> dict[str, Any]:
    """Builds a plain engineering CRS as PROJJSON.

    The CRS is a right-handed Cartesian system with X east, Y north and Z up, all three axes in
    ``unit``. It says nothing about where the model is on earth.

    Args:
        name: The CRS name, typically derived from the model's name.
        unit: The length unit of the axes: a name from :data:`LINEAR_UNITS`, or a
            :class:`LinearUnit` for any other.
        datum: The name of the engineering datum: what the origin is anchored to.

    Returns:
        The CRS as a PROJJSON dict.

    Raises:
        ValueError: If ``unit`` is a name :data:`LINEAR_UNITS` does not know.
    """
    unit_json = unit_projjson(as_linear_unit(unit))
    axes = [
        {"name": "Easting", "abbreviation": "X", "direction": "east", "unit": unit_json},
        {"name": "Northing", "abbreviation": "Y", "direction": "north", "unit": unit_json},
        {"name": "Height", "abbreviation": "Z", "direction": "up", "unit": unit_json},
    ]
    return {
        "$schema": PROJJSON_SCHEMA,
        "type": "EngineeringCRS",
        "name": name,
        "datum": {"type": "EngineeringDatum", "name": datum},
        "coordinate_system": {"subtype": "Cartesian", "axis": axes},
    }


def as_linear_unit(unit: str | LinearUnit) -> LinearUnit:
    """Resolves a unit given by name or as a :class:`LinearUnit`.

    Args:
        unit: A name from :data:`LINEAR_UNITS`, or a :class:`LinearUnit` for any other unit.

    Returns:
        The unit with its conversion factor to the metre.

    Raises:
        ValueError: If ``unit`` is a name :data:`LINEAR_UNITS` does not know.
    """
    if isinstance(unit, LinearUnit):
        return unit
    if unit not in LINEAR_UNITS:
        known = ", ".join(LINEAR_UNITS)
        raise ValueError(f"unknown unit {unit!r}; pass a LinearUnit or one of {known}")
    return LinearUnit(unit, LINEAR_UNITS[unit])


def unit_projjson(unit: LinearUnit) -> dict[str, Any]:
    """The PROJJSON of a linear unit, as an axis or a parameter carries it.

    Args:
        unit: The unit.

    Returns:
        ``{"type": "LinearUnit", "name": ..., "conversion_factor": ...}``.
    """
    return {
        "type": "LinearUnit",
        "name": unit.name,
        "conversion_factor": float(unit.conversion_factor),
    }


def linear_unit(crs: Mapping[str, Any]) -> LinearUnit:
    """Reads the length unit of a CRS's first axis.

    Args:
        crs: A CRS as PROJJSON. A compound CRS is read through its first component.

    Returns:
        The unit's name and its conversion factor to the metre.

    Raises:
        ValueError: If the CRS has no Cartesian axis with a linear unit.
    """
    node: Mapping[str, Any] = crs
    while "components" in node:
        node = node["components"][0]
    try:
        axis = node["coordinate_system"]["axis"][0]
    except (KeyError, IndexError):
        raise ValueError("the CRS has no coordinate system axis") from None
    unit = axis.get("unit", "metre")
    if isinstance(unit, str):
        if unit not in LINEAR_UNITS:
            raise ValueError(f"the CRS axis unit {unit!r} is not a known linear unit")
        return LinearUnit(unit, LINEAR_UNITS[unit])
    if unit.get("type") != "LinearUnit":
        raise ValueError(f"the CRS axis unit is a {unit.get('type')}, not a LinearUnit")
    return LinearUnit(str(unit["name"]), float(unit["conversion_factor"]))
