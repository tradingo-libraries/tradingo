"""
Factor model and risk decomposition.
"""

__all__ = [
    "factor_betas",
    "idiosyncratic_alpha",
    "idiosyncratic_residuals",
    "factor_model",
]

import logging

import pandas as pd

from tradingo.analytics import cov, expectation

logger = logging.getLogger(__name__)


def factor_betas(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    how: str = "exponential",
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Regress asset vs factor returns (dependent vs independent variables)
    to calculate alpha (residuals) and beta (loadings).

    :param asset_returns: Asset returns DataFrame.
    :param factor_returns: Factor returns DataFrame.
    :param how: Method of calculating covariance, either "exponential" or "rolling",
        by default "exponential".
    :param **kwargs: Additional arguments for the ewm (exponentially weighted mean) method.
        For rolling covariance, the window size must be specified in kwargs.
        For example, `window=10` for a 10-period rolling covariance.
        For exponential covariance, the halflife can be specified in kwargs.
        For example, `halflife=10` for a 10-period halflife.

    :returns: tuple with betas, and factor covariances.
    """

    if set(asset_returns.columns).intersection(factor_returns.columns):
        logger.warning("Asset returns and factor returns have overlapping columns. ")
    if asset_returns.index.freq != factor_returns.index.freq:
        raise ValueError("Asset and factor returns must have the same frequency.")

    start = asset_returns.first_valid_index()
    returns = pd.concat((asset_returns.loc[start:], factor_returns.loc[start:]), axis=1)

    cov_mtx = cov(
        returns,
        annualisation=1,
        how=how,
        **kwargs,
    ).reindex(asset_returns.index, level=0)

    factor_covs = (
        cov_mtx.loc[(slice(None), factor_returns.columns), factor_returns.columns]
        .sort_index(level=0)
        .reindex(asset_returns.index, level=0, axis=0)
    )

    # cov to beta
    betas = (
        pd.concat(
            (
                cov_mtx.loc[(slice(None), asset_returns.columns), factor]
                .sort_index(level=0)
                .unstack(level=1)
                .divide(cov_mtx.loc[(slice(None), factor), factor].to_numpy(), axis=0)
                .stack(dropna=False)
                for factor in factor_returns.columns
            ),
            axis=1,
            keys=factor_returns.columns,
        )
        .unstack(level=1)
        .stack(level=0)
    ).reindex(index=factor_returns.columns, level=1)

    return betas, factor_covs


def idiosyncratic_alpha(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    betas: pd.DataFrame = None,
    how: str = "exponential",
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate idiosyncratic alphas from asset returns and factor returns.
    Alphas are calculated as the difference between asset returns and
    the product of betas and factor returns.

    :param asset_returns: Asset returns DataFrame.
    :param factor_returns: Factor returns DataFrame.
    :param betas: Betas DataFrame, by default None.
    :param how: Method of calculating covariance, either "exponential" or "rolling",
        by default "exponential".
    :param **kwargs: Additional arguments for the ewm (exponentially weighted mean) method.
        For rolling covariance, the window size must be specified in kwargs.
        For example, `window=10` for a 10-period rolling covariance.
        For exponential covariance, the halflife can be specified in kwargs.
        For example, `halflife=10` for a 10-period halflife.

    :returns: tuple with alphas and mean returns dataframes
    """

    if betas is None:
        betas, _ = factor_betas(
            asset_returns,
            factor_returns,
            how=how,
            **kwargs,
        )
    elif not betas.index.get_level_values(0).difference(asset_returns.index).empty:
        raise ValueError(
            "Asset returns and betas must have the same index. "
            f"Got {asset_returns.index} and {betas.index}"
        )

    if set(asset_returns.columns).intersection(factor_returns.columns):
        logger.warning("Asset returns and factor returns have overlapping columns. ")
    if asset_returns.index.freq != factor_returns.index.freq:
        raise ValueError("Asset and factor returns must have the same frequency.")

    start = asset_returns.first_valid_index()
    returns = pd.concat((asset_returns.loc[start:], factor_returns.loc[start:]), axis=1)

    # calculate alpha = E[asset] - beta * E[factor]
    means = expectation(
        returns,
        annualisation=1,
        how=how,
        **kwargs,
    ).dropna(axis=0, how="all")

    index = means.index.intersection(
        betas.index.get_level_values(0).unique().sort_values()
    )
    betas = betas.reindex(index=index, level=0)
    means = means.reindex(index=index, level=0)

    alphas = (
        pd.concat(
            (
                means.loc[:, asset_returns.columns]
                .sort_index(level=0)
                .subtract(
                    betas.loc[(slice(None), factor), asset_returns.columns]
                    .reindex(
                        pd.MultiIndex.from_product([index, [factor]]), fill_value=None
                    )
                    .mul(means.loc[:, [factor]].to_numpy()),
                    axis=0,
                )
                for factor in factor_returns.columns
            ),
            axis=0,
        )
        .sort_index()
        .reindex(index=factor_returns.columns, level=1)
        .dropna(how="all")
    )

    return alphas, means


def idiosyncratic_residuals(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    betas: pd.DataFrame,
    alphas: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate regression residuals: observed return minus model-predicted return.
    """

    idx = asset_returns.index.intersection(betas.index.get_level_values(0))
    asset_returns = asset_returns.reindex(idx)
    factor_returns = factor_returns.reindex(idx)
    betas = betas.reindex(index=idx, level=0)
    alphas = alphas.reindex(index=idx, level=0)

    predicted_returns = pd.concat(
        [
            betas.loc[(slice(None), factor), :]
            .multiply(factor_returns[factor], axis=0)
            .add(alphas.loc[(slice(None), factor), :])
            for factor in factor_returns.columns
        ],
        axis=0,
    ).sort_index()

    residuals = asset_returns - predicted_returns
    return residuals


def factor_model(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    how: str = "exponential",
    **kwargs,
):
    """
    end-to-end multi-factor model regression, including confidence intervals.

    :param asset_returns: Asset returns DataFrame.
    :param factor_returns: Factor returns DataFrame.
    :param how: Method of calculating covariance, either "exponential" or "rolling",
        by default "exponential".
    :param **kwargs: Additional arguments for the ewm (exponentially weighted mean) method.
        For rolling covariance, the window size must be specified in kwargs.
        For example, `window=10` for a 10-period rolling covariance.
        For exponential covariance, the halflife can be specified in kwargs.
        For example, `halflife=10` for a 10-period halflife.

    :returns: tuple with alphas, betas, and their standard error dataframes.

    """

    betas, covs_factors = factor_betas(
        asset_returns,
        factor_returns,
        how=how,
        **kwargs,
    )

    alphas, means_returns = idiosyncratic_alpha(
        asset_returns,
        factor_returns,
        **kwargs,
    )

    # model confidence
    residuals = idiosyncratic_residuals(
        asset_returns,
        factor_returns,
        betas,
        alphas,
    )

    cov_mtx = (
        cov(
            residuals.unstack(),
            annualisation=1,
            how=how,
            **kwargs,
        )
        .unstack(level=(1, 2))
        .reindex(asset_returns.index, level=0)
    )
    cov_residuals = (
        cov_mtx[[idx for idx in cov_mtx.columns if idx[:2] == idx[2:]]]
        .droplevel((0, 1), axis=1)
        .stack()
    )

    covs_factors = covs_factors.unstack(fill_value=None)
    covs_factors = covs_factors[
        [c for c in covs_factors.columns if c[0] == c[1]]
    ].droplevel(0, axis=1)

    # alpha and beta confidence is measured through standard errors:
    # se_betas = std_residuals / std_factors
    # se_alphas = std_residuals * sqrt(1/n + mean_factors**2 / var_factors)

    se_betas = pd.concat(
        [
            cov_residuals.loc[(slice(None), factor), :].divide(
                covs_factors[factor], axis=0
            )
            ** 0.5
            for factor in factor_returns.columns
        ],
        axis=0,
    ).sort_index()

    means_factors = means_returns[factor_returns.columns]
    se_alphas = pd.concat(
        [
            cov_residuals.loc[(slice(None), factor), :].multiply(
                (1.0 / len(means_factors[factor]))
                + (means_factors[factor] ** 2).div(covs_factors[factor]),
                axis=0,
            )
            ** 0.5
            for factor in factor_returns.columns
        ],
        axis=0,
    ).sort_index()

    return alphas, betas, se_alphas, se_betas
