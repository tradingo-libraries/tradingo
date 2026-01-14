import pytest
import pandas as pd
from arcticdb import Arctic

from tradingo.symbols import symbol_provider, symbol_publisher


@pytest.fixture
def arctic() -> Arctic:
    return Arctic("mem://test-db")


def test_symbol_provider(arctic: Arctic) -> None:

    lib = arctic.get_library("my-lib", create_if_missing=True)

    @symbol_provider(input_1="my-lib/symbol")
    def provider(input_1: pd.DataFrame) -> pd.DataFrame:
        return input_1

    @symbol_publisher("my-lib/symbol")
    def publisher(input_1: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()

    publisher(arctic=arctic, input_1="my-lib/symbol")
    res = provider(arctic=arctic, input_1="my-lib/symbol")

    assert res.empty
