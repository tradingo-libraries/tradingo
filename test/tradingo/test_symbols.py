import pandas as pd
import pytest
from arcticdb import Arctic

from tradingo.symbols import symbol_provider, symbol_publisher


@pytest.fixture
def arctic() -> Arctic:
    return Arctic("mem://test-db")


def test_symbol_provider(arctic: Arctic) -> None:

    @symbol_provider(input_1="my-lib/symbol")
    def provider(input_1: pd.DataFrame) -> pd.DataFrame:
        return input_1

    @symbol_publisher("my-lib/symbol")
    def publisher(input_1: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            1,
            index=pd.date_range("2025-01-01", "2026-01-01"),
            columns=["A"],
        )

    publisher(arctic=arctic, input_1=pd.DataFrame(), dry_run=False)
    res = provider(arctic=arctic, input_1="my-lib/symbol")  # type: ignore

    assert res["A"].mean() == 1


def test_symbols_function_is_provided_for_and_published(arctic: Arctic) -> None:

    data = pd.DataFrame(
        1,
        index=pd.date_range("2026-01-27", "2026-01-28"),
        columns=["A"],
    )

    @symbol_provider(input_1="my-lib/symbol")  # type: ignore
    @symbol_publisher("my-lib/out-symbol")
    def provider_publisher(input_1: pd.DataFrame) -> pd.DataFrame:
        return input_1

    lib = arctic.get_library(name="my-lib", create_if_missing=True)
    lib.write("symbol", data)

    provider_publisher(arctic=arctic, input_1="my-lib/symbol", dry_run=False)  # type: ignore

    assert lib.list_symbols(regex="out-symbol")


def test_symbols_provider_fallsback_to_pickle(arctic: Arctic) -> None:

    data = pd.DataFrame(
        1,
        index=pd.date_range("2026-01-27", "2026-01-28"),
        columns=["A"],
    )

    @symbol_provider(input_1="my-lib/symbol")  # type: ignore
    @symbol_publisher("my-lib/out-symbol")
    def provider_publisher(input_1: pd.DataFrame) -> pd.DataFrame:
        return input_1

    lib = arctic.get_library(name="my-lib", create_if_missing=True)
    lib.write_pickle("symbol", data)

    provider_publisher(arctic=arctic, input_1="my-lib/symbol", dry_run=False)  # type: ignore

    assert lib.list_symbols(regex="out-symbol")


def test_symbols_provider_pass_a_list(arctic: Arctic) -> None:

    data1 = pd.DataFrame(
        2.0,
        index=pd.date_range("2026-01-25", "2026-01-26", tz="utc"),
        columns=["A"],
    )
    data2 = pd.DataFrame(
        1.0,
        index=pd.date_range("2026-01-27", "2026-01-28", tz="utc"),
        columns=["A"],
    )

    @symbol_provider(input_1=["my-lib/symbol-1", "my-lib/symbol-2"])  # type: ignore
    @symbol_publisher("my-lib/out-symbol")
    def provider_publisher(input_1: pd.DataFrame) -> pd.DataFrame:
        return input_1

    lib = arctic.get_library(name="my-lib", create_if_missing=True)
    lib.write("symbol-1", data1)
    lib.write("symbol-2", data2)

    provider_publisher(arctic=arctic, input_1=["my-lib/symbol-1", "my-lib/symbol-2"], dry_run=False)  # type: ignore

    assert lib.list_symbols(regex="out-symbol")

    data = lib.read("out-symbol")
    pd.testing.assert_frame_equal(
        pd.concat((data1, data2)), data.data, check_freq=False
    )
