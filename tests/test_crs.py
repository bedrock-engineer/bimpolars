import json

import pytest

from bimpolars import engineering_crs, linear_unit
from bimpolars.crs import CRS84H, LINEAR_UNITS, LinearUnit


def test_engineering_crs_defaults() -> None:
    crs = engineering_crs()
    assert crs["type"] == "EngineeringCRS"
    assert crs["name"] == "Model coordinates"
    assert crs["datum"] == {"type": "EngineeringDatum", "name": "Model origin"}
    axes = crs["coordinate_system"]["axis"]
    assert [axis["abbreviation"] for axis in axes] == ["X", "Y", "Z"]
    assert [axis["direction"] for axis in axes] == ["east", "north", "up"]
    assert axes[0]["unit"] == {"type": "LinearUnit", "name": "metre", "conversion_factor": 1.0}


def test_engineering_crs_in_another_unit() -> None:
    crs = engineering_crs("Feet", unit="foot")
    assert linear_unit(crs) == LinearUnit("foot", LINEAR_UNITS["foot"])
    custom = engineering_crs("Custom", unit=LinearUnit("half metre", 0.5))
    assert linear_unit(custom) == LinearUnit("half metre", 0.5)
    with pytest.raises(ValueError, match="unknown unit"):
        engineering_crs(unit="furlong")


def test_linear_unit_of_a_compound_crs() -> None:
    crs = {
        "type": "CompoundCRS",
        "components": [engineering_crs(unit="millimetre"), {"type": "VerticalCRS"}],
    }
    assert linear_unit(crs) == LinearUnit("millimetre", 0.001)
    assert linear_unit({"coordinate_system": {"axis": [{"unit": "metre"}]}}) == LinearUnit(
        "metre", 1.0
    )
    with pytest.raises(ValueError):
        linear_unit({"type": "GeographicCRS"})
    with pytest.raises(ValueError):
        linear_unit(
            {"coordinate_system": {"axis": [{"unit": {"type": "AngularUnit", "name": "degree"}}]}}
        )


def test_pyproj_reads_the_projjson() -> None:
    pyproj = pytest.importorskip("pyproj")
    crs = pyproj.CRS.from_json_dict(engineering_crs("Sixty5", unit="millimetre"))
    assert crs.type_name == "Engineering CRS"
    assert crs.name == "Sixty5"
    assert [axis.unit_name for axis in crs.axis_info] == ["millimetre"] * 3
    assert [axis.unit_conversion_factor for axis in crs.axis_info] == [0.001] * 3
    assert [axis.direction for axis in crs.axis_info] == ["east", "north", "up"]


def test_crs84h_is_what_pyproj_calls_it() -> None:
    """The PROJJSON of the geog column is OGC:CRS84h as PROJ defines it."""
    json.dumps(CRS84H)
    assert CRS84H["type"] == "GeographicCRS"
    assert CRS84H["id"] == {"authority": "OGC", "code": "CRS84h"}
    with pytest.raises(ValueError, match="not a known linear unit"):
        linear_unit(CRS84H)  # degrees: a geog column has no length unit
    pyproj = pytest.importorskip("pyproj")
    crs = pyproj.CRS.from_json_dict(CRS84H)
    assert crs == pyproj.CRS("OGC:CRS84h")
    assert crs.to_authority() == ("OGC", "CRS84h")
    assert crs.is_geographic and len(crs.axis_info) == 3
    assert [axis.abbrev for axis in crs.axis_info] == ["Lon", "Lat", "h"]
    assert [axis.unit_name for axis in crs.axis_info] == ["degree", "degree", "metre"]
