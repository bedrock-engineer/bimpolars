import json

import polars as pl

from bimpolars import properties_series


def test_properties_dtypes_follow_the_values() -> None:
    rows = [
        {"Pset_A": {"flag": True, "count": 1, "ratio": 2, "text": "a", "codes": [1, 2]}},
        {
            "Pset_A": {"flag": False, "count": 2, "ratio": 0.5, "text": "b", "codes": [3]},
            "Qto": {"Area": 1.5},
        },
        {},
    ]
    series = properties_series(rows)
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
    assert series[2] == {"Pset_A": None, "Qto": None}


def test_values_whose_type_differs_between_rows_become_strings() -> None:
    rows = [
        {"P": {"mixed": True, "nested": {"a": 1}, "lists": [1, 2], "none": None}},
        {"P": {"mixed": "UNKNOWN", "nested": "text", "lists": ["x"], "none": None}},
    ]
    series = properties_series(rows)
    fields = series.dtype.to_schema()["P"].to_schema()
    assert fields["mixed"] == pl.String
    assert fields["nested"] == pl.String
    assert fields["lists"] == pl.String
    assert "none" not in fields
    assert series[0]["P"]["mixed"] == "True"
    assert series[0]["P"]["nested"] == json.dumps({"a": 1})
    assert series[1]["P"]["lists"] == json.dumps(["x"])


def test_lists() -> None:
    rows = [
        {"P": {"ints": [1, 2], "widened": [1, 2], "words": ["a"], "empty": [], "nested": [[1]]}},
        {"P": {"ints": [], "widened": [0.5], "words": [], "empty": [], "nested": [1]}},
    ]
    series = properties_series(rows)
    fields = series.dtype.to_schema()["P"].to_schema()
    assert fields["ints"] == pl.List(pl.Int64)
    assert fields["widened"] == pl.List(pl.Float64)
    assert fields["words"] == pl.List(pl.String)
    assert fields["empty"] == pl.List(pl.String)
    assert fields["nested"] == pl.String
    assert series[0]["P"]["widened"] == [1.0, 2.0]
    assert series[1]["P"]["ints"] == []
    assert series[0]["P"]["nested"] == "[[1]]"


def test_no_properties_at_all() -> None:
    series = properties_series([{}, {"Empty": {}}])
    assert series.dtype == pl.Null
    assert len(series) == 2


def test_the_column_name() -> None:
    series = properties_series([{"P": {"a": 1}}], name="psets")
    assert series.name == "psets"
    assert series.dtype.to_schema()["P"].to_schema()["a"] == pl.Int64
