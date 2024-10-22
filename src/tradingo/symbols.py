import os
from collections import defaultdict
from typing import Optional
import functools
import logging
from typing import NamedTuple
from urllib.parse import urlparse, parse_qsl

import pandas as pd

import arcticdb as adb


logger = logging.getLogger(__name__)

ENVIRONMENT = os.environ.get("ENVIRONMENT", "test")
ARCTIC_URL = os.environ.get(
    "TRADINGO_ARCTIC_URL",
    f"lmdb:///home/rory/dev/airflow/{ENVIRONMENT}/arctic.db",
)


class Symbol(NamedTuple):

    library: str
    symbol: str
    kwargs: dict


def lib_provider(**libs):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            arctic = adb.Arctic(ARCTIC_URL)
            libs_ = {
                k: arctic.get_library(v, create_if_missing=True)
                for k, v in libs.items()
            }
            return func(*args, **libs_, **kwargs)

        return wrapper

    return decorator


def parse_symbol(symbol, kwargs, symbol_prefix="", symbol_postfix=""):
    string_kwargs = {k: str(v) for k, v in kwargs.items()}
    symbol = symbol.format(**string_kwargs)
    parsed_symbol = urlparse(symbol)
    lib, sym = parsed_symbol.path.split("/")
    kwargs = dict(parse_qsl(parsed_symbol.query))
    symbol_prefix = symbol_prefix.format(**string_kwargs)
    symbol_postfix = symbol_postfix.format(**string_kwargs)
    for key, value in kwargs.items():
        if key == "as_of":
            try:
                kwargs[key] = int(value)
            except TypeError:
                continue
    return Symbol(lib, symbol_prefix + sym + symbol_postfix, kwargs)


def symbol_provider(
    symbol_prefix="",
    no_date=False,
    **symbols,
):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(
            *args,
            start_date=None,
            end_date=None,
            arctic: Optional[adb.Arctic] = None,
            **kwargs,
        ):

            arctic = arctic or adb.Arctic(ARCTIC_URL)
            orig_symbol_data = {}
            requested_symbols = symbols.copy()

            for symbol in symbols:
                if symbol in kwargs and isinstance(
                    kwargs[symbol], (pd.DataFrame, pd.Series)
                ):
                    orig_symbol_data[symbol] = kwargs.pop(symbol)
                if symbol in kwargs and isinstance(kwargs[symbol], str):
                    requested_symbols[symbol] = kwargs[symbol]
                if symbol in kwargs and kwargs[symbol] is None:
                    requested_symbols.pop(symbol)

            symbols_data = {
                k: (
                    arctic.get_library(
                        parse_symbol(v, kwargs, symbol_prefix=symbol_prefix).library,
                        create_if_missing=True,
                    )
                    .read(
                        parse_symbol(v, kwargs, symbol_prefix=symbol_prefix).symbol,
                        date_range=(
                            None
                            if no_date
                            else (
                                pd.Timestamp(start_date) if start_date else None,
                                pd.Timestamp(end_date) if end_date else None,
                            )
                        ),
                        **parse_symbol(v, kwargs, symbol_prefix=symbol_prefix).kwargs,
                    )
                    .data
                )
                for k, v in requested_symbols.items()
                if k not in orig_symbol_data
            }
            kwargs.update(symbols_data)
            kwargs.update(orig_symbol_data)

            logger.info("Providing %s symbols from %s", symbols_data.keys(), arctic)

            return func(
                *args, **kwargs, arctic=arctic, start_date=start_date, end_date=end_date
            )

        return wrapper

    return decorator


def symbol_publisher(
    *symbols,
    symbol_prefix="",
    symbol_postfix="",
    astype=None,
    template=None,
    library_options=None,
    write_pickle=False,
):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(
            *args,
            dry_run=False,
            arctic: Optional[adb.Arctic] = None,
            snapshot: Optional[str] = None,
            clean: bool = False,
            **kwargs,
        ):

            arctic = arctic or adb.Arctic(ARCTIC_URL)
            out = func(*args, arctic=arctic, **kwargs)
            logger.info("Publishing %s to %s", symbols or template, arctic)

            # yf kws:
            #    kwdout = out.pop(-1)
            #
            if template:
                out, symbols_ = list(zip(*out))
                symbols_ = [template.format(*s) for s in symbols_]
            else:
                symbols_ = symbols

            libraries = defaultdict(dict)

            for data, symbol in zip(out, symbols_):

                if astype:
                    data = data.astype(
                        astype
                        if not isinstance(astype, dict)
                        else {k: v for k, v in astype.items() if k in data.columns}
                    )

                logger.info(
                    "writing symbol=%s rows=%s",
                    parse_symbol(symbol, kwargs, symbol_prefix, symbol_postfix),
                    len(data.index),
                )

                lib, sym, params = parse_symbol(
                    symbol,
                    kwargs,
                    symbol_prefix=symbol_prefix,
                    symbol_postfix=symbol_postfix,
                )

                if not dry_run:
                    lib = arctic.get_library(
                        lib, create_if_missing=True, library_options=library_options
                    )
                    if isinstance(data.index, pd.DatetimeIndex):
                        if clean:
                            lib.delete(sym)
                        result = lib.update(
                            sym,
                            data,
                            upsert=True,
                            date_range=(data.index[0], data.index[-1]),
                            **params,
                        )
                    elif write_pickle:
                        result = lib.write_pickle(sym, data, **params)

                    else:
                        result = lib.write(sym, data, **params)

                    libraries[lib.name][result.symbol] = result.version

            if not dry_run and snapshot:
                for lib_name, versions in libraries.items():
                    logging.info(
                        "Snapshotting %s for %s %s", lib_name, snapshot, versions
                    )
                    lib = arctic.get_library(lib_name)
                    if snapshot in lib.list_snapshots():
                        lib.delete_snapshot(snapshot)
                    lib.snapshot(snapshot_name=snapshot, versions=versions)

            if dry_run:
                return pd.concat(out, keys=symbols_, axis=1)

            return None

        wrapper.published_fields = symbols
        return wrapper

    return decorator
