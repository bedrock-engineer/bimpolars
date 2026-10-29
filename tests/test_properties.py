import json

import polars as pl

from bimpolars.ifc import _properties_series


def test_properties_dtypes_follow_the_values() -> None:
    rows = [
        {"Pset_A": {"flag": True, "count": 1, "ratio": 2, "text": "a", "codes": [1, 2]}},
        {
            "Pset_A": {"flag": False, "count": 2, "ratio": 0.5, "text": "b", "codes": [3]},
            "Qto": {"Area": 1.5},
        },
        {},
    ]
    series = _properties_series(rows)
    schema = series.dtype.to_schema()
    pset_a = schema["Pset_A"].to_schema()
    assert pset_a["flag"] == pl.Boolean
    assert pset_a["count"] == pl.Int64
    assert pset_a["ratio"] == pl.Float64
    assert pset_a["text"] == pl.String
    assert pset_a["codes"] == pl.List(pl.Int64)
    assert schema["Qto"].to_schema()["Area"] == pl.Float64
    assert series[0]["Pset_A"]["ratio"] == 2.0
    assert series[0]["Qto"] is None
    assert series[2] is None


def test_values_whose_type_differs_between_rows_become_strings() -> None:
    rows = [
        {"P": {"mixed": True, "nested": {"a": 1}, "lists": [1, 2], "none": None}},
        {"P": {"mixed": "UNKNOWN", "nested": "text", "lists": ["x"], "none": None}},
    ]
    series = _properties_series(rows)
    fields = series.dtype.to_schema()["P"].to_schema()
    assert fields["mixed"] == pl.String
    assert fields["nested"] == pl.String
    assert fields["lists"] == pl.String
    assert "none" not in fields
    assert series[0]["P"]["mixed"] == "True"
    assert series[0]["P"]["nested"] == json.dumps({"a": 1})
    assert series[1]["P"]["lists"] == json.dumps(["x"])


def test_no_properties_at_all() -> None:
    series = _properties_series([{}, {"Empty": {}}])
    assert series.dtype == pl.Null
    assert len(series) == 2
