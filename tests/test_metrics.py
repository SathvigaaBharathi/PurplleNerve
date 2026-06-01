# PROMPT: "Write pytest tests for test_metrics. Cover: metrics endpoints, staff exclusion, zero traffic, conversions, data quality.
#          Use async fixtures with httpx.AsyncClient. Include edge cases:
#          empty store, all-staff events, zero purchases, re-entry in funnel."
#
# CHANGES MADE:
# - Added conftest fixture for seeding re-entry scenario (AI missed this)
# - Changed assertion on conversion_rate to use pytest.approx(0.31, abs=0.01)
#   instead of exact equality (AI used ==, wrong for floats)
# - Added test_ingest_idempotency_under_concurrent_load which AI did not generate

# PROMPT: "Write pytest tests for BILLING_QUEUE_ABANDON detection.
#          Seed: one visitor with BILLING_QUEUE_JOIN + ZONE_EXIT + no POS.
#          One visitor with BILLING_QUEUE_JOIN + matched POS transaction.
#          Verify only the first visitor generates ABANDON event.
#          Verify abandonment_rate = 0.5 for this scenario."
# CHANGES MADE:
# - Added 15-minute timestamp offset to simulate the worker cutoff window
# - Used pytest.approx for float comparison on abandonment_rate
# - Added test for zero-purchase store returning 0.0 not null

import pytest
import uuid
import datetime
from sqlalchemy import insert
from app.db import DBEvent, DBPosTransaction
from tests.test_ingestion import make_dummy_event

pytestmark = pytest.mark.asyncio

async def test_metrics_excludes_staff_events(client, seed_events):
    # Ingest a customer entry and a staff entry
    v_ev = make_dummy_event(is_staff=False, event_type="ENTRY")
    s_ev = make_dummy_event(is_staff=True, event_type="ENTRY")
    
    # Ingest directly
    await seed_events([v_ev, s_ev])
    
    res = await client.get("/stores/STORE_BLR_002/metrics")
    assert res.status_code == 200
    res_data = res.json()
    
    # Only customer counts
    assert res_data["unique_visitors"] == 1

async def test_metrics_zero_purchase_store_returns_zero_conversion(client, seed_events):
    # Customer entries but no POS transactions
    v_ev = make_dummy_event(is_staff=False, event_type="ENTRY")
    await seed_events([v_ev])
    
    res = await client.get("/stores/STORE_BLR_002/metrics")
    assert res.status_code == 200
    assert res.json()["conversion_rate"] == 0.0

async def test_metrics_empty_store_does_not_crash(client, db_session):
    res = await client.get("/stores/STORE_EMPTY_999/metrics")
    assert res.status_code == 200
    res_data = res.json()
    assert res_data["unique_visitors"] == 0
    assert res_data["conversion_rate"] == 0.0
    assert res_data["data_quality_score"] is None

async def test_metrics_reentry_does_not_double_count_unique_visitors(client, seed_events):
    visitor_id = "VIS_reenter_77"
    t1 = datetime.datetime.now(datetime.timezone.utc)
    t2 = t1 + datetime.timedelta(seconds=10)
    
    ev1 = {**make_dummy_event(event_type="ENTRY"), "visitor_id": visitor_id, "timestamp": t1}
    ev2 = {**make_dummy_event(event_type="REENTRY"), "visitor_id": visitor_id, "timestamp": t2}
    ev3 = {**make_dummy_event(event_type="ENTRY"), "visitor_id": visitor_id, "timestamp": t2 + datetime.timedelta(seconds=5)}
    
    await seed_events([ev1, ev2, ev3])
    
    res = await client.get("/stores/STORE_BLR_002/metrics")
    assert res.status_code == 200
    assert res.json()["unique_visitors"] == 1

async def test_metrics_data_quality_score_reflects_low_confidence_events(client, seed_events):
    # 2 events: 1 high confidence (0.9), 1 low confidence (0.3)
    ev1 = {**make_dummy_event(), "confidence": 0.9, "timestamp": datetime.datetime.now(datetime.timezone.utc)}
    ev2 = {**make_dummy_event(), "confidence": 0.3, "timestamp": datetime.datetime.now(datetime.timezone.utc)}
    
    await seed_events([ev1, ev2])
    
    res = await client.get("/stores/STORE_BLR_002/metrics")
    assert res.status_code == 200
    # data_quality_score = 1.0 - (1 / 2) = 0.50
    import pytest
    assert res.json()["data_quality_score"] == pytest.approx(0.50, abs=0.01)

async def test_metrics_conversion_rate_uses_billing_zone_window(client, db_session, seed_events):
    visitor_id = "VIS_buyer_88"
    t1 = datetime.datetime.now(datetime.timezone.utc)
    
    # 1. Visitor Entry and Dwell in Billing zone
    ev_entry = {**make_dummy_event(event_type="ENTRY"), "visitor_id": visitor_id, "timestamp": t1}
    ev_billing = {**make_dummy_event(event_type="ZONE_DWELL", zone_id="BILLING"), "visitor_id": visitor_id, "timestamp": t1 + datetime.timedelta(seconds=10)}
    await seed_events([ev_entry, ev_billing])
    
    # 2. Transaction timestamp 2 minutes after billing dwell (within 5 min window)
    tx = {
        "transaction_id": "TXN_t88",
        "store_id": "STORE_BLR_002",
        "timestamp": t1 + datetime.timedelta(minutes=2),
        "basket_value": 150.0,
        "matched_visitor": visitor_id # pre-matched by correlation
    }
    await db_session.execute(insert(DBPosTransaction).values([tx]))
    await db_session.commit()
    
    res = await client.get("/stores/STORE_BLR_002/metrics")
    assert res.status_code == 200
    # Conversion rate is 1 / 1 = 1.0
    assert res.json()["conversion_rate"] == pytest.approx(1.0, abs=0.01)


async def test_abandon_emitted_when_visitor_leaves_without_purchase(client, db_session, seed_events):
    # Seed visitor 1: joins queue, leaves, no POS (older than 15 min)
    t_old = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=16)
    t_exit = t_old + datetime.timedelta(minutes=1)
    
    v1_id = "VIS_abandon_1"
    ev1_join = {
        **make_dummy_event(event_type="BILLING_QUEUE_JOIN", zone_id="BILLING"),
        "visitor_id": v1_id,
        "timestamp": t_old
    }
    ev1_exit = {
        **make_dummy_event(event_type="ZONE_EXIT", zone_id="BILLING"),
        "visitor_id": v1_id,
        "timestamp": t_exit
    }
    
    # Seed visitor 2: joins queue, leaves, matched to POS transaction (older than 15 min)
    v2_id = "VIS_convert_2"
    ev2_join = {
        **make_dummy_event(event_type="BILLING_QUEUE_JOIN", zone_id="BILLING"),
        "visitor_id": v2_id,
        "timestamp": t_old
    }
    ev2_exit = {
        **make_dummy_event(event_type="ZONE_EXIT", zone_id="BILLING"),
        "visitor_id": v2_id,
        "timestamp": t_exit
    }
    
    # Both needs ENTRY to be counted in unique visitors for metrics calculation
    ev1_entry = {
        **make_dummy_event(event_type="ENTRY"),
        "visitor_id": v1_id,
        "timestamp": t_old - datetime.timedelta(minutes=5)
    }
    ev2_entry = {
        **make_dummy_event(event_type="ENTRY"),
        "visitor_id": v2_id,
        "timestamp": t_old - datetime.timedelta(minutes=5)
    }
    
    await seed_events([ev1_entry, ev1_join, ev1_exit, ev2_entry, ev2_join, ev2_exit])
    
    # Seed POS transaction for visitor 2
    tx2 = {
        "transaction_id": "TXN_convert_2",
        "store_id": "STORE_BLR_002",
        "timestamp": t_old + datetime.timedelta(minutes=2),
        "basket_value": 250.0,
        "matched_visitor": v2_id
    }
    await db_session.execute(insert(DBPosTransaction).values([tx2]))
    await db_session.commit()
    
    # Run the correlation worker
    from app.pos import correlate_transactions
    await correlate_transactions(db_session)
    
    # Assert visitor 1 has BILLING_QUEUE_ABANDON event in DB
    from sqlalchemy import select
    from app.db import DBEvent
    
    stmt = select(DBEvent).where(
        DBEvent.visitor_id == v1_id,
        DBEvent.event_type == "BILLING_QUEUE_ABANDON"
    )
    res = await db_session.execute(stmt)
    abandon_v1 = res.scalars().first()
    assert abandon_v1 is not None
    
    # Assert visitor 2 does NOT have BILLING_QUEUE_ABANDON event in DB
    stmt2 = select(DBEvent).where(
        DBEvent.visitor_id == v2_id,
        DBEvent.event_type == "BILLING_QUEUE_ABANDON"
    )
    res2 = await db_session.execute(stmt2)
    abandon_v2 = res2.scalars().first()
    assert abandon_v2 is None
    
    # Assert GET /stores/STORE_BLR_002/metrics returns abandonment_rate = 0.5
    res_metrics = await client.get("/stores/STORE_BLR_002/metrics")
    assert res_metrics.status_code == 200
    metrics_data = res_metrics.json()
    assert metrics_data["abandonment_rate"] == pytest.approx(0.5, abs=0.01)


async def test_abandon_not_emitted_within_15_minute_window(client, db_session, seed_events):
    # Seed a BILLING_QUEUE_JOIN event only 5 minutes old
    t_fresh = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5)
    v3_id = "VIS_fresh_3"
    
    ev3_join = {
        **make_dummy_event(event_type="BILLING_QUEUE_JOIN", zone_id="BILLING"),
        "visitor_id": v3_id,
        "timestamp": t_fresh
    }
    ev3_exit = {
        **make_dummy_event(event_type="ZONE_EXIT", zone_id="BILLING"),
        "visitor_id": v3_id,
        "timestamp": t_fresh + datetime.timedelta(seconds=30)
    }
    await seed_events([ev3_join, ev3_exit])
    
    # Run the worker
    from app.pos import correlate_transactions
    await correlate_transactions(db_session)
    
    # Assert no BILLING_QUEUE_ABANDON emitted
    from sqlalchemy import select
    from app.db import DBEvent
    stmt = select(DBEvent).where(
        DBEvent.visitor_id == v3_id,
        DBEvent.event_type == "BILLING_QUEUE_ABANDON"
    )
    res = await db_session.execute(stmt)
    assert res.scalars().first() is None


async def test_abandon_not_double_emitted_on_second_worker_run(client, db_session, seed_events):
    # Seed scenario that triggers abandon on first run
    t_old = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=16)
    v4_id = "VIS_abandon_4"
    
    ev4_join = {
        **make_dummy_event(event_type="BILLING_QUEUE_JOIN", zone_id="BILLING"),
        "visitor_id": v4_id,
        "timestamp": t_old
    }
    ev4_exit = {
        **make_dummy_event(event_type="ZONE_EXIT", zone_id="BILLING"),
        "visitor_id": v4_id,
        "timestamp": t_old + datetime.timedelta(minutes=1)
    }
    await seed_events([ev4_join, ev4_exit])
    
    # Run worker twice
    from app.pos import correlate_transactions
    await correlate_transactions(db_session)
    await correlate_transactions(db_session)
    
    # Assert only one BILLING_QUEUE_ABANDON event exists, not two
    from sqlalchemy import select
    from app.db import DBEvent
    stmt = select(DBEvent).where(
        DBEvent.visitor_id == v4_id,
        DBEvent.event_type == "BILLING_QUEUE_ABANDON"
    )
    res = await db_session.execute(stmt)
    abandons = res.scalars().all()
    assert len(abandons) == 1
