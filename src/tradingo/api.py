from __future__ import annotations
from copy import deepcopy
import inspect
from os import pathconf_names
from typing import Optional
import contextlib
import pandas as pd
from datetime import datetime
import re
import arcticdb as adb

import tradingo.utils


READ_SIG = inspect.signature(adb.arctic.Library.read)


class _Read:

    def __init__(
        self,
        path_so_far,
        library: adb.library.Library,
        assets,
        common_args,
        common_kwargs,
        root: Tradingo,
    ):
        self.path_so_far = path_so_far
        self.library = library
        self.assets = assets
        self.path = ".".join(self.path_so_far)
        self.common_args = common_args
        self.common_kwargs = common_kwargs
        self.root = root

    def __dir__(self):
        return [*self.list(), *super().__dir__()]

    def __repr__(self):
        return f'Namespace("{self.path}")'

    def __getattr__(self, symbol):
        return self.__class__(
            (*self.path_so_far, symbol),
            library=self.library,
            assets=self.assets,
            common_args=self.common_args,
            common_kwargs=self.common_kwargs,
            root=self.root,
        )

    def __getitem__(self, symbol):
        return self.__getattr__(symbol)

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        operations = ["merge", "transpose", "concat", "with_instrument_details"]
        operation, index = next(
            (
                (elem, i)
                for i, elem in enumerate(self.path_so_far)
                if elem in operations
            ),
            (None, len(self.path_so_far)),
        )
        part_one_path = self.path_so_far[0:index]
        part_two_path = self.path_so_far[index + 1 :]

        lib_kwargs = {
            k: v for k, v in kwargs.items() if k in READ_SIG.parameters.keys()
        }

        callback_kwargs = {}

        def get_callback(operation):

            for i in (pd.DataFrame, tradingo.utils, pd):
                try:
                    return getattr(i, operation)
                except AttributeError as ex:
                    continue
            raise AttributeError(operation)

        if operation:
            callback = get_callback(operation)
            callback_sig = inspect.signature(callback)
            callback_kwargs = {
                k: v for k, v in kwargs.items() if k in callback_sig.parameters.keys()
            }

        def get_data_at_path(path, kw):

            kw.setdefault(
                "columns",
                (
                    self.assets
                    if all(i in path for i in ("backtest", "portfolio"))
                    else None
                ),
            )

            for k in self.common_kwargs:
                kw.pop(k, None)
            return self.library.read(
                ".".join(path),
                *args,
                *self.common_args,
                **kw,
                **self.common_kwargs,
            ).data

        lhs = get_data_at_path(part_one_path, lib_kwargs)

        if operation:
            callback_args = (lhs,)
            if part_two_path:
                callback_lib = part_two_path[0]
                callback_args = (
                    lhs,
                    self.__class__(
                        (*self.root._get_path_so_far(callback_lib), *part_two_path[1:]),
                        getattr(self.root, (callback_lib)).library,
                        self.assets,
                        self.common_args,
                        self.common_kwargs,
                        self.root,
                    )(*args, **kwargs),
                )

            if operation == "concat":
                callback_args = (callback_args,)

            callback = get_callback(operation)
            callback_sig = inspect.signature(pd.DataFrame)
            return callback(*callback_args, **callback_kwargs)
        return lhs

    def update(self, *args, **kwargs):
        self.library.update(self.path, *args, **kwargs)

    def list(self, *args, **kwargs):
        regex = re.escape(".".join((self.path_so_far)) + ".")
        kwargs["regex"] = regex + kwargs.setdefault("regex", "")
        return list(
            dict.fromkeys(
                [
                    i.replace(f'{".".join(self.path_so_far)}.', "").split(".")[0]
                    for i in self.library.list_symbols(*args, **kwargs)
                ]
            )
        )

    def head(self, *args, **kwargs):
        return self.library.head(self.path, *args, **kwargs).data

    def tail(self, *args, **kwargs):
        return self.library.tail(self.path, *args, **kwargs).data

    def exists(self):
        pass


class Tradingo(adb.Arctic):

    def __init__(
        self,
        name,
        *args,
        provider: Optional[str] = None,
        universe: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.name = name
        self.provider = provider
        self.instrument_symbol = universe
        self.universe = universe
        self._context_args = ()
        self._context_kwargs = {}

    @contextlib.contextmanager
    def common_args(self, *args, **kwargs):
        try:
            self._context_args = args
            self._context_kwargs = kwargs
            yield self
        finally:
            self._context_args = ()
            self._context_kwargs = {}

    def _get_path_so_far(self, library):
        path_so_far = []
        if library == "instruments":
            return path_so_far
        if self.provider:
            path_so_far.append(self.provider)
        if self.universe:
            path_so_far.append(self.universe)
        return path_so_far

    def __getattr__(self, library):

        if library in self.list_libraries():
            path_so_far = self._get_path_so_far(library)
            if library == "instruments":
                return _Read(
                    library=self.get_library(library),
                    path_so_far=path_so_far,
                    assets=[],
                    common_args=(),
                    common_kwargs={},
                    root=self,
                )

            assets = []
            if self.universe:
                assets = getattr(self.instruments, self.universe)().index.to_list()

            return _Read(
                library=self.get_library(library),
                path_so_far=path_so_far,
                assets=assets,
                common_args=self._context_args,
                common_kwargs=self._context_kwargs,
                root=self,
            )

        return super().__getattr__(library)

    def __dir__(self):
        return [*self.list_libraries(), *super().__dir__()]


class VolSurface(Tradingo):

    def get(
        self,
        symbol: str,
        start_date: pd.Timestamp = pd.Timestamp("1970-01-01 00:00+00:00"),
        end_date: pd.Timestamp = pd.Timestamp.now("utc"),
        **kwargs,
    ):

        with self.common_args(date_range=(start_date, end_date)):

            futures = (
                pd.concat(
                    (self.futures.cboe.VX.expiration(), self.futures.cboe.VX.price()),
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
                    library.expiration(),
                    library.option_type(),
                    library.strike(),
                    library.bid(),
                    library.ask(),
                    library.implied_volatility(),
                    library.delta(),
                    library.vega(),
                    library.gamma(),
                    library.theta(),
                    library.rho(),
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
