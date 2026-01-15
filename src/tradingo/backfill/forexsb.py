import pathlib

import pandas as pd
from arcticdb.version_store.library import Library

from tradingo import symbols


def load_backfill(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    symbol: str,
    interval: str,
    data_dir: pathlib.Path,
) -> pd.DataFrame:

    return (
        pd.read_csv(
            f"{data_dir}/{symbol}{interval}.csv",
            sep="\t",
            parse_dates=True,
            date_format="%Y-%m-%d %H:%M",
            index_col=0,
        )
        .set_axis(["open", "high", "low", "close", "volume"], axis=1)
        .rename_axis("timestamp")
        .tz_localize("utc")
        .loc[start_date:end_date]
    )


@symbols.lib_provider(pricelib="{raw_price_lib}")
def create_universe(
    pricelib: Library,
    symbols: list[str],
    end_date: pd.Timestamp,
    start_date: pd.Timestamp,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    result = pd.concat(
        (
            (
                pd.DataFrame(
                    pricelib.read(symbol, date_range=(start_date, end_date)).data
                )
                for symbol in symbols
            )
        ),
        axis=1,
        keys=(s.rsplit(".", maxsplit=1)[0] for s in symbols),
    ).reorder_levels([1, 0], axis=1)
    return (
        result["open"],
        result["high"],
        result["low"],
        result["close"],
        result["volume"],
    )
