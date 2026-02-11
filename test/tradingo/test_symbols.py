from typing import cast

import numpy as np
import pandas as pd
import pytest
from arcticdb import Arctic, LibraryOptions
from arcticdb.exceptions import NoSuchVersionException

from tradingo.symbols import (
    Symbol,
    SymbolParseError,
    lib_provider,
    symbol_provider,
    symbol_publisher,
)


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


# =============================================================================
# Symbol.parse tests
# =============================================================================


def test_symbol_parse_basic() -> None:
    symbol = Symbol.parse("my-lib/my-symbol", {})
    assert symbol.library == "my-lib"
    assert symbol.symbol == "my-symbol"
    assert symbol.kwargs == {}


def test_symbol_parse_with_kwargs() -> None:
    symbol = Symbol.parse("my-lib/{name}", {"name": "test-symbol"})
    assert symbol.library == "my-lib"
    assert symbol.symbol == "test-symbol"


def test_symbol_parse_with_prefix_and_postfix() -> None:
    symbol = Symbol.parse(
        "my-lib/symbol",
        {},
        symbol_prefix="pre-",
        symbol_postfix="-post",
    )
    assert symbol.symbol == "pre-symbol-post"


def test_symbol_parse_with_query_params() -> None:
    symbol = Symbol.parse("my-lib/symbol?columns=A,B,C", {})
    assert symbol.kwargs == {"columns": ["A", "B", "C"]}


def test_symbol_parse_with_as_of() -> None:
    symbol = Symbol.parse("my-lib/symbol?as_of=123", {})
    assert symbol.kwargs == {"as_of": 123}


def test_symbol_parse_missing_param_raises() -> None:
    with pytest.raises(SymbolParseError, match="Missing parameter"):
        Symbol.parse("my-lib/{missing}", {})


def test_symbol_parse_invalid_format_raises() -> None:
    with pytest.raises(SymbolParseError, match="invalid"):
        Symbol.parse("invalid-no-slash", {})


# =============================================================================
# symbol_provider tests
# =============================================================================


def test_symbol_provider_with_symbol_prefix(arctic: Arctic) -> None:
    lib = arctic.get_library(name="my-lib", create_if_missing=True)
    data = pd.DataFrame(
        1.0,
        index=pd.date_range("2026-01-01", "2026-01-05", tz="utc"),
        columns=["A"],
    )
    lib.write("prefix-symbol", data)

    @symbol_provider(symbol_prefix="prefix-", input_1="my-lib/symbol")
    def provider(input_1: pd.DataFrame) -> pd.DataFrame:
        return input_1

    result = provider(arctic=arctic)  # type: ignore
    pd.testing.assert_frame_equal(result, data, check_freq=False)


def test_symbol_provider_with_no_date(arctic: Arctic) -> None:
    lib = arctic.get_library(name="my-lib", create_if_missing=True)
    data = pd.DataFrame(
        {"A": [1, 2, 3]},
        index=[0, 1, 2],  # Non-datetime index
    )
    lib.write("no-date-symbol", data)

    @symbol_provider(no_date=True, input_1="my-lib/no-date-symbol")
    def provider(input_1: pd.DataFrame) -> pd.DataFrame:
        return input_1

    result = provider(arctic=arctic)  # type: ignore
    pd.testing.assert_frame_equal(result, data)


def test_symbol_provider_raise_if_missing_false(arctic: Arctic) -> None:
    @symbol_provider(input_1="my-lib/nonexistent")  # type: ignore
    def provider(input_1: pd.DataFrame | None) -> pd.DataFrame | None:
        return input_1

    result = provider(arctic=arctic, raise_if_missing=False)  # type: ignore
    assert result is None


def test_symbol_provider_raise_if_missing_true(arctic: Arctic) -> None:
    @symbol_provider(input_1="my-lib/nonexistent")
    def provider(input_1: pd.DataFrame) -> pd.DataFrame:
        return input_1

    with pytest.raises(NoSuchVersionException):
        provider(arctic=arctic, raise_if_missing=True)  # type: ignore


def test_symbol_provider_date_range_filtering(arctic: Arctic) -> None:
    lib = arctic.get_library(name="my-lib", create_if_missing=True)
    data = pd.DataFrame(
        {"A": range(10)},
        index=pd.date_range("2026-01-01", periods=10, tz="utc"),
    )
    lib.write("range-symbol", data)

    @symbol_provider(input_1="my-lib/range-symbol")
    def provider(input_1: pd.DataFrame) -> pd.DataFrame:
        return input_1

    result = provider(  # type: ignore
        arctic=arctic,
        start_date=pd.Timestamp("2026-01-03", tz="utc"),
        end_date=pd.Timestamp("2026-01-07", tz="utc"),
    )
    assert len(result) == 5
    assert result.index[0] == pd.Timestamp("2026-01-03", tz="utc")
    assert result.index[-1] == pd.Timestamp("2026-01-07", tz="utc")


def test_symbol_provider_dict_of_symbols(arctic: Arctic) -> None:
    lib = arctic.get_library(name="my-lib", create_if_missing=True)
    data1 = pd.DataFrame(
        {"A": [1.0]},
        index=pd.date_range("2026-01-01", periods=1, tz="utc"),
    )
    data2 = pd.DataFrame(
        {"B": [2.0]},
        index=pd.date_range("2026-01-01", periods=1, tz="utc"),
    )
    lib.write("sym1", data1)
    lib.write("sym2", data2)

    @symbol_provider(input_1={"first": "my-lib/sym1", "second": "my-lib/sym2"})  # type: ignore
    def provider(input_1: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        return input_1

    result = provider(arctic=arctic)  # type: ignore
    assert "first" in result
    assert "second" in result
    pd.testing.assert_frame_equal(result["first"], data1, check_freq=False)
    pd.testing.assert_frame_equal(result["second"], data2, check_freq=False)


def test_symbol_provider_passes_dataframe_directly(arctic: Arctic) -> None:
    direct_data = pd.DataFrame({"A": [1, 2, 3]})

    @symbol_provider(input_1="my-lib/symbol")
    def provider(input_1: pd.DataFrame) -> pd.DataFrame:
        return input_1

    result = provider(arctic=arctic, input_1=direct_data)
    pd.testing.assert_frame_equal(result, direct_data)


def test_symbol_provider_with_none_symbol_skips(arctic: Arctic) -> None:
    @symbol_provider(input_1="my-lib/symbol")  # type: ignore
    def provider(input_1: pd.DataFrame | None = None) -> pd.DataFrame | None:
        return input_1

    result = provider(arctic=arctic, input_1=None)
    assert result is None


def test_symbol_provider_passes_dates_to_function(arctic: Arctic) -> None:
    lib = arctic.get_library(name="my-lib", create_if_missing=True)
    data = pd.DataFrame(
        {"A": [1.0]},
        index=pd.date_range("2026-01-01", periods=1, tz="utc"),
    )
    lib.write("symbol", data)

    @symbol_provider(input_1="my-lib/symbol")  # type: ignore
    def provider(
        input_1: pd.DataFrame,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
        return input_1, start_date, end_date

    start = pd.Timestamp("2026-01-01")
    end = pd.Timestamp("2026-01-10")
    result = provider(arctic=arctic, start_date=start, end_date=end)  # type: ignore
    assert result[1] == start
    assert result[2] == end


def test_symbol_provider_with_columns_query_param(arctic: Arctic) -> None:
    lib = arctic.get_library(name="my-lib", create_if_missing=True)
    data = pd.DataFrame(
        {"A": [1.0], "B": [2.0], "C": [3.0]},
        index=pd.date_range("2026-01-01", periods=1, tz="utc"),
    )
    lib.write("multi-col-symbol", data)

    @symbol_provider(input_1="my-lib/multi-col-symbol?columns=A,B")
    def provider(input_1: pd.DataFrame) -> pd.DataFrame:
        return input_1

    result = provider(arctic=arctic)  # type: ignore
    assert list(result.columns) == ["A", "B"]


# =============================================================================
# symbol_publisher tests
# =============================================================================


def test_symbol_publisher_dry_run_returns_concat(arctic: Arctic) -> None:
    @symbol_publisher("lib1/sym1", "lib2/sym2")
    def publisher() -> tuple[pd.DataFrame, pd.DataFrame]:
        df1 = pd.DataFrame({"A": [1]}, index=pd.date_range("2026-01-01", periods=1))
        df2 = pd.DataFrame({"B": [2]}, index=pd.date_range("2026-01-01", periods=1))
        return df1, df2

    result = publisher(arctic=arctic, dry_run=True)
    assert isinstance(result, pd.DataFrame)
    assert ("lib1/sym1", "A") in result.columns
    assert ("lib2/sym2", "B") in result.columns


def test_symbol_publisher_with_symbol_prefix(arctic: Arctic) -> None:
    @symbol_publisher("my-lib/symbol", symbol_prefix="pre-")
    def publisher() -> pd.DataFrame:
        return pd.DataFrame(
            {"A": [1.0]},
            index=pd.date_range("2026-01-01", periods=1),
        )

    publisher(arctic=arctic, dry_run=False)
    lib = arctic.get_library("my-lib")
    assert "pre-symbol" in lib.list_symbols()


def test_symbol_publisher_with_symbol_postfix(arctic: Arctic) -> None:
    @symbol_publisher("my-lib/symbol", symbol_postfix="-post")
    def publisher() -> pd.DataFrame:
        return pd.DataFrame(
            {"A": [1.0]},
            index=pd.date_range("2026-01-01", periods=1),
        )

    publisher(arctic=arctic, dry_run=False)
    lib = arctic.get_library("my-lib")
    assert "symbol-post" in lib.list_symbols()


def test_symbol_publisher_with_prefix_and_postfix(arctic: Arctic) -> None:
    @symbol_publisher("my-lib/symbol", symbol_prefix="pre-", symbol_postfix="-post")
    def publisher() -> pd.DataFrame:
        return pd.DataFrame(
            {"A": [1.0]},
            index=pd.date_range("2026-01-01", periods=1),
        )

    publisher(arctic=arctic, dry_run=False)
    lib = arctic.get_library("my-lib")
    assert "pre-symbol-post" in lib.list_symbols()


def test_symbol_publisher_with_astype_single(arctic: Arctic) -> None:
    @symbol_publisher("my-lib/typed", astype=np.dtype("float32"))
    def publisher() -> pd.DataFrame:
        return pd.DataFrame(
            {"A": [1.0, 2.0]},
            index=pd.date_range("2026-01-01", periods=2),
        )

    publisher(arctic=arctic, dry_run=False)
    lib = arctic.get_library("my-lib")
    result = cast(pd.DataFrame, lib.read("typed").data)
    assert result["A"].dtype == np.float32


def test_symbol_publisher_with_astype_dict(arctic: Arctic) -> None:
    @symbol_publisher(
        "my-lib/typed-dict",
        astype={"A": np.dtype("float32"), "B": np.dtype("int32")},
    )
    def publisher() -> pd.DataFrame:
        return pd.DataFrame(
            {"A": [1.0, 2.0], "B": [3.0, 4.0]},
            index=pd.date_range("2026-01-01", periods=2),
        )

    publisher(arctic=arctic, dry_run=False)
    lib = arctic.get_library("my-lib")
    result = cast(pd.DataFrame, lib.read("typed-dict").data)
    assert result["A"].dtype == np.float32
    assert result["B"].dtype == np.int32


def test_symbol_publisher_with_template(arctic: Arctic) -> None:
    @symbol_publisher(template="{0}/{1}")  # type: ignore
    def publisher() -> (
        tuple[
            tuple[pd.DataFrame, tuple[str, str]], tuple[pd.DataFrame, tuple[str, str]]
        ]
    ):
        df1 = pd.DataFrame({"A": [1]}, index=pd.date_range("2026-01-01", periods=1))
        df2 = pd.DataFrame({"B": [2]}, index=pd.date_range("2026-01-01", periods=1))
        return (df1, ("lib1", "sym1")), (df2, ("lib2", "sym2"))

    publisher(arctic=arctic, dry_run=False)
    assert "sym1" in arctic.get_library("lib1").list_symbols()
    assert "sym2" in arctic.get_library("lib2").list_symbols()


def test_symbol_publisher_write_pickle(arctic: Arctic) -> None:
    @symbol_publisher("my-lib/pickled", write_pickle=True)
    def publisher() -> pd.DataFrame:
        return pd.DataFrame(
            {"A": [1, 2, 3]},
            index=[0, 1, 2],  # Non-datetime index
        )

    publisher(arctic=arctic, dry_run=False)
    lib = arctic.get_library("my-lib")
    result = lib.read("pickled").data
    assert isinstance(result, pd.DataFrame)
    assert list(result["A"]) == [1, 2, 3]


def test_symbol_publisher_with_snapshot(arctic: Arctic) -> None:
    @symbol_publisher("my-lib/snap-symbol")
    def publisher() -> pd.DataFrame:
        return pd.DataFrame(
            {"A": [1.0]},
            index=pd.date_range("2026-01-01", periods=1),
        )

    publisher(arctic=arctic, dry_run=False, snapshot="my-snapshot")
    lib = arctic.get_library("my-lib")
    assert "my-snapshot" in lib.list_snapshots()


def test_symbol_publisher_snapshot_replaces_existing(arctic: Arctic) -> None:
    @symbol_publisher("my-lib/snap-replace")
    def publisher(value: int) -> pd.DataFrame:
        return pd.DataFrame(
            {"A": [float(value)]},
            index=pd.date_range("2026-01-01", periods=1),
        )

    publisher(arctic=arctic, value=1, dry_run=False, snapshot="replace-snap")
    publisher(arctic=arctic, value=2, dry_run=False, snapshot="replace-snap")

    lib = arctic.get_library("my-lib")
    snapshots = lib.list_snapshots()
    assert "replace-snap" in snapshots


def test_symbol_publisher_with_clean(arctic: Arctic) -> None:
    lib = arctic.get_library("my-lib", create_if_missing=True)
    old_data = pd.DataFrame(
        {"A": [0.0]},
        index=pd.date_range("2025-01-01", periods=1, tz="utc"),
    )
    lib.write("clean-symbol", old_data)

    @symbol_publisher("my-lib/clean-symbol")
    def publisher() -> pd.DataFrame:
        return pd.DataFrame(
            {"A": [1.0]},
            index=pd.date_range("2026-01-01", periods=1),
        )

    publisher(arctic=arctic, dry_run=False, clean=True)
    lib = arctic.get_library("my-lib", create_if_missing=True)
    result = cast(pd.DataFrame, lib.read("clean-symbol").data)
    assert len(result) == 1
    assert result.index[0].year == 2026


def test_symbol_publisher_skips_empty_dataframe(arctic: Arctic) -> None:
    @symbol_publisher("my-lib/empty-symbol")
    def publisher() -> pd.DataFrame:
        return pd.DataFrame()

    publisher(arctic=arctic, dry_run=False)
    lib = arctic.get_library("my-lib", create_if_missing=True)
    assert "empty-symbol" not in lib.list_symbols()


def test_symbol_publisher_rejects_positional_args(arctic: Arctic) -> None:
    @symbol_publisher("my-lib/symbol")
    def publisher(value: int) -> pd.DataFrame:
        return pd.DataFrame(
            {"A": [float(value)]},
            index=pd.date_range("2026-01-01", periods=1),
        )

    with pytest.raises(ValueError, match="Keyword only arguments"):
        publisher(arctic, False, None, False, 1)


def test_symbol_publisher_with_formatted_symbol(arctic: Arctic) -> None:
    @symbol_publisher("my-lib/{name}")
    def publisher(name: str) -> pd.DataFrame:
        return pd.DataFrame(
            {"A": [1.0]},
            index=pd.date_range("2026-01-01", periods=1),
        )

    publisher(arctic=arctic, name="formatted-sym", dry_run=False)
    lib = arctic.get_library("my-lib")
    assert "formatted-sym" in lib.list_symbols()


def test_symbol_publisher_with_library_options(arctic: Arctic) -> None:
    options = LibraryOptions(rows_per_segment=10)

    @symbol_publisher("options-lib/symbol", library_options=options)
    def publisher() -> pd.DataFrame:
        return pd.DataFrame(
            {"A": range(100)},
            index=pd.date_range("2026-01-01", periods=100),
        )

    publisher(arctic=arctic, dry_run=False)
    assert "options-lib" in arctic.list_libraries()


# =============================================================================
# lib_provider tests
# =============================================================================


def test_lib_provider_basic(arctic: Arctic) -> None:
    from arcticdb.version_store.library import Library

    @lib_provider(my_lib="test-lib")  # type: ignore
    def func(arctic: Arctic, my_lib: Library) -> str:
        return str(my_lib.name)

    result = func(arctic=arctic)  # type: ignore
    assert result == "test-lib"


def test_lib_provider_creates_library_if_missing(arctic: Arctic) -> None:
    from arcticdb.version_store.library import Library

    @lib_provider(new_lib="brand-new-lib")  # type: ignore
    def func(arctic: Arctic, new_lib: Library) -> str:
        return str(new_lib.name)

    result = func(arctic=arctic)  # type: ignore
    assert result == "brand-new-lib"
    assert "brand-new-lib" in arctic.list_libraries()


def test_lib_provider_override_via_kwarg(arctic: Arctic) -> None:
    from arcticdb.version_store.library import Library

    @lib_provider(my_lib="default-lib")  # type: ignore
    def func(arctic: Arctic, my_lib: Library) -> str:
        return str(my_lib.name)

    result = func(arctic=arctic, my_lib="override-lib")
    assert result == "override-lib"


def test_lib_provider_multiple_libs(arctic: Arctic) -> None:
    from arcticdb.version_store.library import Library

    @lib_provider(lib_a="lib-a", lib_b="lib-b")  # type: ignore
    def func(arctic: Arctic, lib_a: Library, lib_b: Library) -> tuple[str, str]:
        return lib_a.name, lib_b.name

    result = func(arctic=arctic)  # type: ignore
    assert result == ("lib-a", "lib-b")


# =============================================================================
# Combined decorator tests
# =============================================================================


def test_lib_provider_with_symbol_publisher(arctic: Arctic) -> None:
    from arcticdb.version_store.library import Library

    @lib_provider(output_lib="combined-lib")  # type: ignore
    @symbol_publisher("combined-lib/symbol")
    def func(arctic: Arctic, output_lib: Library) -> pd.DataFrame:
        return pd.DataFrame(
            {"A": [1.0]},
            index=pd.date_range("2026-01-01", periods=1),
        )

    func(arctic=arctic, dry_run=False)  # type: ignore
    lib = arctic.get_library("combined-lib")
    assert "symbol" in lib.list_symbols()


def test_provider_and_publisher_passes_arctic_to_function(arctic: Arctic) -> None:
    lib = arctic.get_library("my-lib", create_if_missing=True)
    data = pd.DataFrame(
        {"A": [1.0]},
        index=pd.date_range("2026-01-01", periods=1, tz="utc"),
    )
    lib.write("input-sym", data)

    @symbol_provider(input_1="my-lib/input-sym")  # type: ignore
    @symbol_publisher("my-lib/output-sym")
    def func(input_1: pd.DataFrame, arctic: Arctic) -> pd.DataFrame:
        assert arctic is not None
        return input_1

    func(arctic=arctic, dry_run=False)  # type: ignore
    assert "output-sym" in lib.list_symbols()
