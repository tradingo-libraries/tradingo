"""RSS feed point-in-time snapshot collector."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)


def collect_rss_snapshot(
    source: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch an RSS feed and return all articles as a DataFrame.

    Index: published_at (UTC DatetimeIndex)
    Columns: link, title, summary, collected_at

    start_date / end_date are accepted for pipeline compatibility but not used
    to filter articles — all articles present in the feed are captured. Filter
    on read via ArcticDB date_range or as_of for point-in-time recall.
    """
    try:
        import feedparser
    except ImportError as e:
        raise ImportError("feedparser is required: pip install feedparser") from e

    logger.info("fetching RSS feed source=%s", source)
    feed = feedparser.parse(source)

    collected_at = datetime.now(tz=timezone.utc)
    records = []

    for entry in feed.entries:
        published = entry.get("published_parsed")
        if published is None:
            continue
        pub_dt = datetime(
            published[0],
            published[1],
            published[2],
            published[3],
            published[4],
            published[5],
            tzinfo=timezone.utc,
        )
        records.append(
            {
                "published_at": pub_dt,
                "link": entry.get("link", ""),
                "title": entry.get("title", ""),
                "summary": entry.get("summary", "")[:1000],
                "collected_at": collected_at,
            }
        )

    if not records:
        logger.info("no articles found in feed source=%s", source)
        return pd.DataFrame(
            columns=["link", "title", "summary", "collected_at"],
            index=pd.DatetimeIndex([], name="published_at", tz="UTC"),
        )

    df = (
        pd.DataFrame(records)
        .drop_duplicates(subset=["link"])
        .set_index("published_at")
        .sort_index()
    )
    df.index = pd.DatetimeIndex(df.index, tz="UTC")

    logger.info("collected %d articles from %s", len(df), source)
    return df
