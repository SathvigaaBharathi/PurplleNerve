# PROMPT: "Write pytest tests for test_metrics. Cover: metrics endpoints, staff exclusion, zero traffic, conversions, data quality.
#          Use async fixtures with httpx.AsyncClient. Include edge cases:
#          empty store, all-staff events, zero purchases, re-entry in funnel."
#
# CHANGES MADE:
# - Added conftest fixture for seeding re-entry scenario (AI missed this)
# - Changed assertion on conversion_rate to use pytest.approx(0.31, abs=0.01)
#   instead of exact equality (AI used ==, wrong for floats)
# - Added test_ingest_idempotency_under_concurrent_load which AI did not generate

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
