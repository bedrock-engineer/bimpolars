import ifcopenshell
import pytest

from bimpolars import BimDataFrame, data, read_ifc, simple_features


@pytest.fixture(scope="session")
def sixty5_structural_model() -> ifcopenshell.file:
    return ifcopenshell.open(str(data.fetch(data.SIXTY5_STRUCTURAL)))


@pytest.fixture(scope="session")
def sixty5_structural(sixty5_structural_model: ifcopenshell.file) -> BimDataFrame:
    return read_ifc(sixty5_structural_model)


@pytest.fixture(scope="session")
def sixty5_structural_features(sixty5_structural: BimDataFrame) -> BimDataFrame:
    return simple_features(sixty5_structural)
