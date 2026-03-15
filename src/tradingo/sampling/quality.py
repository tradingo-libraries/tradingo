"""Data quality checks for sampled price data."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataQualityError(Exception):
    """Raised when data staleness has reached or exceeded ffill_limit."""


def _trailing_nan_count(series: pd.Series) -> int:
    """Count consecutive trailing NaN values at the end of a series."""
    return next(
        (i for i, v in enumerate(series.iloc[::-1]) if not pd.isna(v)), len(series)
    )


def check_price_staleness(
    prices: pd.DataFrame,
    universe_name: str,
    ffill_limit: int = 10,
    alert_recipient: str = "",
) -> None:
    """
    Check trailing NaN count per instrument in prices.

    - Sends email + raises DataQualityError if any instrument >= ffill_limit
      (ffill has stopped carrying forward — NaN is now in stored data)
    - Sends email + logs WARNING if any instrument >= ffill_limit - 1
      (one day away from ffill expiry)

    :param prices: DataFrame with prices (columns = tickers)
    :param universe_name: Name of the universe (for log/alert messages)
    :param ffill_limit: The same limit used in convert_prices_to_ccy; alert
        fires at ffill_limit - 1, error fires at ffill_limit.
    :param alert_recipient: Email address to notify; no email sent if empty.
    """
    staleness: dict[str, int] = {}
    for col in prices.columns:
        staleness[col] = _trailing_nan_count(prices[col])

    staleness_series = pd.Series(staleness, name="trailing_stale_days")

    def _notify(heading: str, stale_instruments: pd.Series) -> None:
        if not alert_recipient:
            return
        try:
            from tradingo.notifications.email import send_email

            send_email(
                body=f"<h3>{heading}</h3>{stale_instruments.to_frame().to_html()}",
                subject=f"[tradingo] Data Quality Alert: {universe_name}",
                recipient=alert_recipient,
            )
            logger.info("Staleness alert sent to %s", alert_recipient)
        except Exception as exc:
            logger.error("Failed to send staleness alert email: %s", exc)

    exceeded = staleness_series[staleness_series >= ffill_limit]
    if not exceeded.empty:
        msg = (
            f"Universe {universe_name}: ffill exhausted for {len(exceeded)} "
            f"instrument(s) (>= {ffill_limit} days):\n" + exceeded.to_string()
        )
        _notify(
            f"ffill exhausted for {len(exceeded)} instrument(s) (>= {ffill_limit} days)",
            exceeded,
        )
        raise DataQualityError(msg)

    approaching = staleness_series[staleness_series >= ffill_limit - 1]
    if not approaching.empty:
        msg = (
            f"Universe {universe_name}: {len(approaching)} instrument(s) "
            f"approaching ffill limit ({ffill_limit - 1}/{ffill_limit} days):\n"
            + approaching.to_string()
        )
        logger.warning(msg)
        _notify(
            f"{len(approaching)} instrument(s) approaching ffill limit ({ffill_limit - 1}/{ffill_limit} days)",
            approaching,
        )
