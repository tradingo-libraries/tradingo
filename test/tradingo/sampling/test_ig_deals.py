"""Tests for IG deal sampling and live position reconstruction."""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from tradingo.sampling.ig import (
    ACTIVITY_PAGE_SIZE,
    _ig_query_time,
    activities_to_deals,
    deals_to_positions,
    sample_ig_positions,
)

NG = "CC.D.NG.UMP.IP"
SPX = "IX.D.SPTRD.IFS.IP"


def activity(
    date: str,
    epic: str,
    direction: str,
    size: float,
    actions: list[str] | None = None,
    status: str = "ACCEPTED",
    deal_id: str | None = None,
) -> dict[str, Any]:
    return {
        "date": date,
        "epic": epic,
        "period": "-",
        "dealId": deal_id or f"D{date}{epic}",
        "channel": "PUBLIC_WEB_API",
        "type": "POSITION",
        "status": status,
        "description": "test",
        "details": {
            "dealReference": "REF",
            "actions": [
                {"actionType": a, "affectedDealId": "X"}
                for a in (actions if actions is not None else ["POSITION_OPENED"])
            ],
            "marketName": "m",
            "currency": "GBP",
            "size": size,
            "direction": direction,
            "level": 100.0,
            "stopLevel": None,
            "limitLevel": None,
        },
    }


class TestActivitiesToDeals:
    def test_signed_position_change_only_for_position_moves(self) -> None:
        deals = activities_to_deals(
            [
                activity("2026-10-26T10:00:00", NG, "BUY", 2),
                activity("2026-10-26T11:00:00", NG, "SELL", 1, ["POSITION_CLOSED"]),
                activity("2026-10-26T12:00:00", NG, "BUY", 5, [], status="REJECTED"),
                activity("2026-10-26T13:00:00", NG, "BUY", 1, ["STOP_LIMIT_AMENDED"]),
            ]
        )
        assert deals["position_change"].tolist() == [2.0, -1.0, 0.0, 0.0]
        assert deals["status"].tolist()[2] == "REJECTED"
        assert str(pd.DatetimeIndex(deals.index).tz) == "UTC"
        assert deals.index[0] == pd.Timestamp("2026-10-26T10:00:00Z")

    def test_empty_has_fixed_schema(self) -> None:
        empty = activities_to_deals([])
        full = activities_to_deals([activity("2026-10-26T10:00:00", NG, "BUY", 1)])
        assert list(empty.columns) == list(full.columns)
        assert (empty.dtypes == full.dtypes).all()

    def test_dedupes_on_deal_id(self) -> None:
        a = activity("2026-10-26T10:00:00", NG, "BUY", 1, deal_id="X")
        assert len(activities_to_deals([a, a])) == 1


class TestDealsToPositions:
    def test_anchors_to_final_position(self) -> None:
        deals = activities_to_deals(
            [
                activity("2026-10-26T10:00:00", NG, "BUY", 2),
                activity("2026-10-26T11:00:00", NG, "BUY", 3),
                activity("2026-10-26T12:00:00", SPX, "SELL", 1),
            ]
        )
        # Account already held 4 NG before the window.
        final = pd.Series({NG: 9.0, SPX: -1.0})
        pos = deals_to_positions(
            deals,
            final,
            pd.Timestamp("2026-10-26T09:00:00Z"),
            pd.Timestamp("2026-10-26T13:00:00Z"),
        )
        assert pos[NG].tolist() == [4.0, 6.0, 9.0, 9.0, 9.0]
        assert pos[SPX].tolist() == [0.0, 0.0, 0.0, -1.0, -1.0]
        assert pos.index[0] == pd.Timestamp("2026-10-26T09:00:00Z")
        assert pos.index[-1] == pd.Timestamp("2026-10-26T13:00:00Z")

    def test_end_before_snapshot_excludes_later_deals(self) -> None:
        deals = activities_to_deals(
            [
                activity("2026-10-26T10:00:00", NG, "BUY", 2),
                activity("2026-10-26T15:00:00", NG, "SELL", 2, ["POSITION_CLOSED"]),
            ]
        )
        pos = deals_to_positions(
            deals,
            pd.Series({NG: 0.0}),
            pd.Timestamp("2026-10-26T09:00:00Z"),
            pd.Timestamp("2026-10-26T12:00:00Z"),
        )
        assert pos[NG].tolist() == [0.0, 2.0, 2.0]
        assert pos.index.max() == pd.Timestamp("2026-10-26T12:00:00Z")

    def test_no_deals(self) -> None:
        pos = deals_to_positions(
            activities_to_deals([]),
            pd.Series({NG: 3.0}),
            pd.Timestamp("2026-10-26T09:00:00Z"),
            pd.Timestamp("2026-10-26T12:00:00Z"),
        )
        assert pos[NG].tolist() == [3.0, 3.0]


class TestSampleIgPositions:
    @pytest.fixture
    def instruments(self) -> pd.DataFrame:
        return pd.DataFrame(index=pd.Index([NG, SPX], name="Symbol"))

    def test_outputs(self, instruments: pd.DataFrame) -> None:
        svc = MagicMock()
        svc.return_dataframe = True
        svc.fetch_open_positions.return_value = pd.DataFrame(
            {
                "epic": [NG, NG, "OTHER"],
                "dealId": ["a", "b", "c"],
                "direction": ["BUY", "BUY", "SELL"],
                "size": [2.0, 3.0, 1.0],
                "level": [100.0, 110.0, 5.0],
            }
        )
        svc.fetch_account_activity.return_value = {
            "activities": [
                activity("2026-10-26T10:00:00", NG, "BUY", 3),
                activity("2026-10-26T10:30:00", "OTHER", "SELL", 1),
            ]
        }
        deals, position, avg = sample_ig_positions(
            instruments,
            start_date=pd.Timestamp("2026-10-26T09:00:00Z"),
            end_date=pd.Timestamp("2026-10-26T11:00:00Z"),
            service=svc,
        )
        assert deals["epic"].unique().tolist() == [NG]
        assert list(position.columns) == [NG, SPX]
        assert position[NG].tolist() == [2.0, 5.0, 5.0]
        assert avg.loc[avg.index[0], NG] == pytest.approx(106.0)
        assert np.isnan(avg.loc[avg.index[0], SPX])
        # caller-owned service is left open and dataframe mode restored
        svc.session.close.assert_not_called()
        assert svc.return_dataframe is True

    def test_bisects_full_pages(self, instruments: pd.DataFrame) -> None:
        svc = MagicMock()
        svc.fetch_open_positions.return_value = pd.DataFrame()
        full: dict[str, list[dict[str, Any]]] = {
            "activities": [{}] * ACTIVITY_PAGE_SIZE
        }
        ok = {"activities": [activity("2026-10-26T10:00:00", NG, "BUY", 1)]}
        svc.fetch_account_activity.side_effect = [full, ok, ok]
        deals, _, _ = sample_ig_positions(
            instruments,
            start_date=pd.Timestamp("2026-10-26T09:00:00Z"),
            end_date=pd.Timestamp("2026-10-26T11:00:00Z"),
            service=svc,
        )
        assert svc.fetch_account_activity.call_count == 3
        assert len(deals) == 1  # deduplicated across halves


def test_live_run_extends_to_now() -> None:
    svc = MagicMock()
    svc.fetch_open_positions.return_value = pd.DataFrame()
    svc.fetch_account_activity.return_value = {"activities": []}
    instruments = pd.DataFrame(index=pd.Index([NG], name="Symbol"))
    end = pd.Timestamp.now("UTC") - pd.Timedelta("10min")
    _, position, _ = sample_ig_positions(
        instruments, end - pd.Timedelta("15min"), end, service=svc
    )
    assert position.index[-1] > end
    _, backfill, _ = sample_ig_positions(
        instruments,
        end - pd.Timedelta("3D"),
        end - pd.Timedelta("2D"),
        service=svc,
    )
    assert backfill.index[-1] == end - pd.Timedelta("2D")


def test_query_time_is_london_local() -> None:
    # BST: UTC 17:07 -> 18:07 London
    assert _ig_query_time(pd.Timestamp("2026-10-22T17:07:00Z")) == pd.Timestamp(
        "2026-10-22T18:07:00"
    )
    # GMT
    assert _ig_query_time(pd.Timestamp("2026-10-26T17:07:00Z")) == pd.Timestamp(
        "2026-10-26T17:07:00"
    )
