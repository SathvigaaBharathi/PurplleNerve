# PROMPT: "Write pytest tests for POS loading and correlation logic.
#          Cover: CSV parsing, timestamp handling, transaction insert idempotency,
#          visitor correlation within 5-minute billing window, missing file graceful handling."
#
# CHANGES MADE:
# - Added explicit billing zone correlation test (AI only generated happy-path CSV load)
# - Added test for parse_iso_timestamp Z-suffix handling
# - Added test for missing CSV path returning 0 without crashing

import pytest
import os
import csv
import datetime
import tempfile
from sqlalchemy import insert, select

from app.db import DBEvent, DBPosTransaction, DBSessionConversion
from app.pos import (
    load_pos_transactions_from_csv,
    correlate_transactions,
    parse_iso_timestamp,
)
from tests.test_ingestion import make_dummy_event

pytestmark = pytest.mark.asyncio


# ─────────────────────────────────────────────
# parse_iso_timestamp utility
# ─────────────────────────────────────────────

async def test_parse_iso_timestamp_handles_Z_suffix():
    ts = parse_iso_timestamp("2026-03-03T14:38:12Z")
    assert ts.tzinfo is not None
    assert ts.year == 2026
    assert ts.month == 3


async def test_parse_iso_timestamp_handles_offset_format():
    ts = parse_iso_timestamp("2026-03-03T14:38:12+00:00")
    assert ts.tzinfo is not None


# ─────────────────────────────────────────────
# load_pos_transactions_from_csv
# ─────────────────────────────────────────────

async def test_load_pos_missing_file_returns_zero(db_session):
    count = await load_pos_transactions_from_csv("/nonexistent/path.csv", db_session)
    assert count == 0


async def test_load_pos_csv_inserts_transactions(db_session):
    # Write a minimal CSV to a temp file
    rows = [
        {"store_id": "STORE_BLR_002", "transaction_id": "TXN_CSV_001",
         "timestamp": "2026-03-03T14:38:12Z", "basket_value_inr": "1200.00"},
        {"store_id": "STORE_BLR_002", "transaction_id": "TXN_CSV_002",
         "timestamp": "2026-03-03T15:00:00Z", "basket_value_inr": "850.50"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["store_id", "transaction_id", "timestamp", "basket_value_inr"])
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = f.name

    try:
        count = await load_pos_transactions_from_csv(tmp_path, db_session)
        assert count == 2
    finally:
        os.unlink(tmp_path)


async def test_load_pos_csv_is_idempotent(db_session):
    """Loading the same CSV twice should not duplicate rows (ON CONFLICT DO NOTHING)."""
    rows = [
        {"store_id": "STORE_BLR_002", "transaction_id": "TXN_IDEM_001",
         "timestamp": "2026-03-03T14:38:12Z", "basket_value_inr": "500.00"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["store_id", "transaction_id", "timestamp", "basket_value_inr"])
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = f.name

    try:
        count1 = await load_pos_transactions_from_csv(tmp_path, db_session)
        count2 = await load_pos_transactions_from_csv(tmp_path, db_session)
        assert count1 == 1
        # Second load returns 1 (same CSV) but DB should still have only 1 row
        result = await db_session.execute(
            select(DBPosTransaction).where(DBPosTransaction.transaction_id == "TXN_IDEM_001")
        )
        rows_db = result.scalars().all()
        assert len(rows_db) == 1
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────
# correlate_transactions
# ─────────────────────────────────────────────

async def test_correlate_matches_billing_visitor_within_5_min_window(db_session, seed_events):
    """
    Visitor is in BILLING zone 2 minutes before a transaction → should be correlated.
    """
    t_base = datetime.datetime.now(datetime.timezone.utc)

    # Seed visitor in BILLING zone
    ev = {
        **make_dummy_event(event_type="ZONE_DWELL", zone_id="BILLING"),
        "visitor_id": "VIS_pos_corr_01",
        "timestamp": t_base,
    }
    await seed_events([ev])

    # Insert transaction 2 minutes after billing dwell
    txn_ts = t_base + datetime.timedelta(minutes=2)
    await db_session.execute(
        insert(DBPosTransaction).values([{
            "transaction_id": "TXN_CORR_001",
            "store_id": "STORE_BLR_002",
            "timestamp": txn_ts,
            "basket_value": 800.0,
        }])
    )
    await db_session.commit()

    await correlate_transactions(db_session)

    # Check session_conversions table
    result = await db_session.execute(
        select(DBSessionConversion).where(
            DBSessionConversion.visitor_id == "VIS_pos_corr_01"
        )
    )
    conversions = result.scalars().all()
    assert len(conversions) == 1


async def test_correlate_does_not_match_visitor_outside_window(db_session, seed_events):
    """
    Visitor was in BILLING zone 10 minutes before transaction (outside 5-min window).
    They should NOT be correlated.
    """
    t_base = datetime.datetime.now(datetime.timezone.utc)

    ev = {
        **make_dummy_event(event_type="ZONE_DWELL", zone_id="BILLING"),
        "visitor_id": "VIS_pos_outside_01",
        "timestamp": t_base - datetime.timedelta(minutes=10),
    }
    await seed_events([ev])

    txn_ts = t_base  # 10 min after billing dwell → outside window
    await db_session.execute(
        insert(DBPosTransaction).values([{
            "transaction_id": "TXN_OUTSIDE_001",
            "store_id": "STORE_BLR_002",
            "timestamp": txn_ts,
            "basket_value": 600.0,
        }])
    )
    await db_session.commit()

    await correlate_transactions(db_session)

    result = await db_session.execute(
        select(DBSessionConversion).where(
            DBSessionConversion.visitor_id == "VIS_pos_outside_01"
        )
    )
    conversions = result.scalars().all()
    assert len(conversions) == 0


async def test_correlate_excludes_staff_from_conversion(db_session, seed_events):
    """
    Staff events in BILLING zone should NOT be matched as converted visitors.
    """
    t_base = datetime.datetime.now(datetime.timezone.utc)

    # Staff event in billing zone
    ev = {
        **make_dummy_event(event_type="ZONE_DWELL", zone_id="BILLING", is_staff=True),
        "visitor_id": "VIS_staff_billing_01",
        "timestamp": t_base,
    }
    await seed_events([ev])

    txn_ts = t_base + datetime.timedelta(minutes=1)
    await db_session.execute(
        insert(DBPosTransaction).values([{
            "transaction_id": "TXN_STAFF_001",
            "store_id": "STORE_BLR_002",
            "timestamp": txn_ts,
            "basket_value": 300.0,
        }])
    )
    await db_session.commit()

    await correlate_transactions(db_session)

    result = await db_session.execute(
        select(DBSessionConversion).where(
            DBSessionConversion.visitor_id == "VIS_staff_billing_01"
        )
    )
    conversions = result.scalars().all()
    assert len(conversions) == 0
