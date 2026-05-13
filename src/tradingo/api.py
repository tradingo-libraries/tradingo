from __future__ import annotations

import os
import re
from typing import Any, cast

import arcticdb as adb
import pandas as pd
from arcticdb.version_store.library import AsOf


class _Read:
    def __init__(
        self,
        *,
        path_so_far: tuple[str, ...] = (),
        library: adb.library.Library,
        root: Tradingo,
    ):
        self._path_so_far = path_so_far
        self._library = library
        self._path = ".".join(self._path_so_far)
        self._root = root

    def __dir__(self) -> list[str]:
        return [*self.list_symbols(), *super().__dir__()]

    def __repr__(self) -> str:
        return f'Namespace("{self._path}")'

    def __getattr__(self, symbol: str) -> _Read:
        return self.__class__(
            path_so_far=(*self._path_so_far, symbol),
            library=self._library,
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
        return cast(pd.DataFrame, result.data)

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
        sub_symbol = re.escape(self._path)
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
                    ).split(
                        "."
                    )[0]
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
        return cast(pd.DataFrame, result.data)

    def tail(
        self,
        n: int = 5,
        as_of: int | str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        columns = columns or []
        result = self._library.tail(self._path, n, as_of, columns, lazy=False)
        assert isinstance(result, adb.VersionedItem)
        return cast(pd.DataFrame, result.data)

    def exists(self) -> bool:
        """Return true if symbol exists"""
        return bool(self._library.list_symbols(regex=f"^{re.escape(self._path)}$"))


class Tradingo(adb.Arctic):  # type: ignore

    def _get_path_so_far(self, library: str) -> list[str]:
        path_so_far: list[str] = []
        if library == "instruments":
            return path_so_far
        return path_so_far

    def __getattr__(self, library: str) -> _Read:
        if library.startswith("_"):
            raise AttributeError(library)
        if library not in self.list_libraries():
            raise AttributeError(library)
        return _Read(
            library=self.get_library(library),
            root=self,
        )

    def __dir__(self) -> list[str]:
        return [*self.list_libraries(), *super().__dir__()]


_GLOBAL_INSTANCE: Tradingo | None = None


def from_env() -> Tradingo:
    global _GLOBAL_INSTANCE
    if _GLOBAL_INSTANCE is not None:
        return _GLOBAL_INSTANCE
    return (_GLOBAL_INSTANCE := Tradingo(os.environ["TP_ARCTIC_URI"]))
