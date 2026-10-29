import ifcopenshell
import pytest

from bimpolars import BimDataFrame, data, read_ifc


@pytest.fixture(scope="session")
def sixty5_structural_model() -> ifcopenshell.file:
    return ifcopenshell.open(str(data.fetch(data.SIXTY5_STRUCTURAL)))


@pytest.fixture(scope="session")
def sixty5_structural(sixty5_structural_model: ifcopenshell.file) -> BimDataFrame:
    return read_ifc(sixty5_structural_model)
