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
    unit: str = "metre",
    unit_factor: float | None = None,
    datum: str = "Model origin",
) -> dict[str, Any]:
    """Builds a plain engineering CRS as PROJJSON.

    The CRS is a right-handed Cartesian system with X east, Y north and Z up, all three axes in
    ``unit``. It says nothing about where the model is on earth.

    Args:
        name: The CRS name, typically derived from the model's name.
        unit: The length unit of the axes. A name from :data:`LINEAR_UNITS`, or any name when
            ``unit_factor`` is given.
        unit_factor: How many metres one ``unit`` is. Defaults to the factor of a known
            ``unit``.
        datum: The name of the engineering datum: what the origin is anchored to.

    Returns:
        The CRS as a PROJJSON dict.

    Raises:
        ValueError: If ``unit`` is unknown and no ``unit_factor`` is given.
    """
    if unit_factor is None:
        try:
            unit_factor = LINEAR_UNITS[unit]
        except KeyError:
            known = ", ".join(LINEAR_UNITS)
            raise ValueError(f"unknown unit {unit!r}; pass unit_factor or one of {known}") from None
    unit_json = {"type": "LinearUnit", "name": unit, "conversion_factor": float(unit_factor)}
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
