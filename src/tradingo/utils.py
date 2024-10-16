import json
import os
import pathlib
import pandas as pd


def get_instruments(config, key="equity") -> pd.DataFrame:
    if "file" in config[key]:
        return pd.read_csv(
            config[key]["file"],
            index_col=config[key]["index_col"],
        ).rename_axis("Symbol")
    if "html" in config[key]:
        return pd.read_html(config[key]["html"], index_col=config[key]["index_col"])[0]
    raise ValueError(config[key])


def with_instrument_details(
    dataframe: pd.DataFrame,
    instruments: pd.DataFrame,
    fields: list[str],
):
    """Add instrument details to column index"""
    return (
        dataframe.transpose()
        .rename_axis("Symbol")
        .merge(instruments[fields], left_index=True, right_index=True)
        .reset_index()
        .set_index([*fields, "Symbol"])
        .sort_index()
        .transpose()
    ).dropna()


def null_instruments(symbols):
    return pd.DataFrame(
        data="",
        index=symbols,
        columns=[
            "Name",
            "SEDOL",
            "ISIN",
            "CUSIP",
            "Incept. Date",
            "Gross Expense Ratio (%)",
            "Net Expense Ratio (%)",
            "Net Assets (USD)",
            "Net Assets as of",
            "Asset Class",
            "Sub Asset Class",
            "Region",
            "Market",
            "Location",
            "Investment Style",
            "Key Facts",
            "Avg. Annual Return: NAV Quarterly",
            "Avg. Annual Return: Price Quarterly",
            "Avg. Annual Return: NAV Monthly",
            "Avg. Annual Return: Price Monthly",
            "Yield",
            "Fixed Income Characteristics",
            "Sustainability Characteristics (MSCI ESG Fund Ratings)",
        ],
    )


def buffer(
    series: pd.Series,
):
    pass
