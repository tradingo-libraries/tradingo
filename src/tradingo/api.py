from __future__ import annotations
import re

from typing import Any

import arcticdb as adb
from arcticdb.options import DEFAULT_ENCODING_VERSION, EncodingVersion
from arcticdb.version_store.library import AsOf
import pandas as pd
from pandas.core.arrays.period import NaTType


class _Read:
    def __init__(
        self,
        path_so_far: tuple[str, ...],
        library: adb.library.Library,
        assets: list[str],
        root: Tradingo,
    ):
        self._path_so_far = path_so_far
        self._library = library
        self._assets = assets
        self._path = ".".join(self._path_so_far)
        self._root = root

    def __dir__(self) -> list[str]:
        return [*self.list(), *super().__dir__()]

    def __repr__(self) -> str:
        return f'Namespace("{self._path}")'

    def __getattr__(self, symbol: str) -> _Read:
        return self.__class__(
            (*self._path_so_far, symbol),
            library=self._library,
            assets=self._assets,
            root=self._root,
        )

    def __getitem__(self, symbol: str) -> _Read:
        return self.__getattr__(symbol)

    def __call__(
        self,
        as_of: AsOf | None = None,
        date_range: tuple[pd.Timestamp | None, pd.Timestamp | None] | None = None,
        row_range: tuple[int, int] | None = None,
        columns: list[str] | None = None,
        query_builder: adb.QueryBuilder | None = None,
    ) -> pd.DataFrame:
        result = self._library.read(
            symbol=".".join(self._path_so_far),
            as_of=as_of,
            date_range=date_range,
            row_range=row_range,
            columns=columns,
            query_builder=query_builder,
            lazy=False,
        )
        assert isinstance(result, adb.VersionedItem)
        return result.data

    def update(
        self,
        data: pd.DataFrame | pd.Series,
        metadata: Any = None,
        upsert: bool = False,
        date_range: tuple[pd.Timestamp | None, pd.Timestamp | None] | None = None,
        prune_previous_versions: bool = False,
    ) -> None:

        self._library.update(
            self._path,
            data,
            metadata,
            upsert,
            date_range,
            prune_previous_versions,
        )

    def list_symbols(
        self,
        snapshot_name: str | None = None,
        regex: str = "",
    ) -> list[str]:
        sub_symbol = re.escape(".".join(self._path_so_far))
        if regex and sub_symbol:
            sub_symbol = re.escape(".").join((sub_symbol, regex))
        elif regex:
            sub_symbol = regex

        return list(
            dict.fromkeys(
                [
                    i.replace(
                        f"{sub_symbol}." if sub_symbol else "",
                        "",
                    )
                    for i in self._library.list_symbols(
                        regex=sub_symbol, snapshot_name=snapshot_name
                    )
                ]
            )
        )

    def head(
        self,
        n: int = 5,
        as_of: AsOf | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        columns = columns or []
        result = self._library.head(self._path, n, as_of, columns, lazy=False)
        assert isinstance(result, adb.VersionedItem)
        return result.data

    def tail(
        self,
        n: int = 5,
        as_of: int | str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        columns = columns or []
        result = self._library.tail(self._path, n, as_of, columns, lazy=False)
        assert isinstance(result, adb.VersionedItem)
        return result.data

    def exists(self) -> bool:
        """Return true if symbol exists"""
        return bool(self._library.list_symbols(regex=f"^{re.escape(self._path)}$"))


class Tradingo(adb.Arctic):  # type: ignore
    def __init__(
        self,
        uri: str,
        provider: str | None = None,
        universe: str | None = None,
        encoding_version: EncodingVersion = DEFAULT_ENCODING_VERSION,
        output_format: adb.OutputFormat | str = adb.OutputFormat.PANDAS,
    ):
        super().__init__(
            uri=uri,
            encoding_version=encoding_version,
            output_format=output_format,
        )
        self._provider = provider
        self._universe = universe

    def _get_path_so_far(self, library: str) -> list[str]:
        path_so_far: list[str] = []
        if library == "instruments":
            return path_so_far
        if self._provider:
            path_so_far.append(self._provider)
        if self._universe:
            path_so_far.append(self._universe)
        return path_so_far

    def __getattr__(self, library: str) -> _Read:
        if library.startswith("_"):
            raise AttributeError(library)
        if library not in self.list_libraries():
            raise AttributeError(library)
        path_so_far = self._get_path_so_far(library)
        if library == "instruments":
            return _Read(
                library=self.get_library(library),
                path_so_far=tuple(path_so_far),
                assets=[],
                root=self,
            )

        assets = []
        if self._universe:
            assets = getattr(self.instruments, self._universe)().index.to_list()

        return _Read(
            library=self.get_library(library),
            path_so_far=tuple(path_so_far),
            assets=assets,
            root=self,
        )

    def __dir__(self) -> list[str]:
        return [*self.list_libraries(), *super().__dir__()]


class VolSurface(Tradingo):
    def get(
        self,
        symbol: str,
        start_date: NaTType | pd.Timestamp = pd.Timestamp("1970-01-01 00:00+00:00"),
        end_date: pd.Timestamp = pd.Timestamp.now("utc"),
    ) -> pd.DataFrame:
        assert isinstance(start_date, pd.Timestamp)
        futures = (
            pd.concat(
                (
                    self.futures.cboe.VX.expiration(date_range=(start_date, end_date)),
                    self.futures.cboe.VX.price(date_range=(start_date, end_date)),
                ),
                axis=1,
                keys=("expiration", "price"),
            )
            .stack()
            .astype({"expiration": "datetime64[ns]"})
            .reset_index()
            .set_index(["timestamp", "symbol"])
        )

        library = getattr(self.options.cboe, symbol)
        option_chains = pd.concat(
            (
                library.expiration(date_range=(start_date, end_date)),
                library.option_type(date_range=(start_date, end_date)),
                library.strike(date_range=(start_date, end_date)),
                library.bid(date_range=(start_date, end_date)),
                library.ask(date_range=(start_date, end_date)),
                library.implied_volatility(date_range=(start_date, end_date)),
                library.delta(date_range=(start_date, end_date)),
                library.vega(date_range=(start_date, end_date)),
                library.gamma(date_range=(start_date, end_date)),
                library.theta(date_range=(start_date, end_date)),
                library.rho(date_range=(start_date, end_date)),
            ),
            axis=1,
            keys=(
                "expiration",
                "option_type",
                "strike",
                "bid",
                "ask",
                "implied_volatility",
                "delta",
                "vega",
                "gamma",
                "theta",
                "rho",
            ),
        ).stack(future_stack=True)

        return option_chains.merge(
            futures, on=["timestamp", "expiration"], how="left"
        ).set_index(["expiration", "option_type", "strike"], append=True)
