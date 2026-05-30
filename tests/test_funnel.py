# PROMPT: "Write pytest tests for test_funnel. Cover: funnel stages, session boundaries, re-entry continuations, staff exclusion.
#          Use async fixtures with httpx.AsyncClient. Include edge cases:
#          empty store, all-staff events, zero purchases, re-entry in funnel."
#
# CHANGES MADE:
# - Added conftest fixture for seeding re-entry scenario (AI missed this)
# - Changed assertion on conversion_rate to use pytest.approx(0.31, abs=0.01)
#   instead of exact equality (AI used ==, wrong for floats)
# - Added test_ingest_idempotency_under_concurrent_load which AI did not generate

import pytest
import datetime
from sqlalchemy import insert
from app.db import DBEvent, DBPosTransaction
from tests.test_ingestion import make_dummy_event

pytestmark = pytest.mark.asyncio

async def test_funnel_session_is_unit_not_raw_events(client, seed_events):
    # One visitor with multiple entry events (re-entry) should count as 1 session
    vid = "VIS_session_u1"
    t1 = datetime.datetime.now(datetime.timezone.utc)
    ev1 = {**make_dummy_event(event_type="ENTRY"), "visitor_id": vid, "timestamp": t1}
    ev2 = {**make_dummy_event(event_type="EXIT"), "visitor_id": vid, "timestamp": t1 + datetime.timedelta(seconds=20)}
    ev3 = {**make_dummy_event(event_type="ENTRY"), "visitor_id": vid, "timestamp": t1 + datetime.timedelta(seconds=40)}
    
    await seed_events([ev1, ev2, ev3])
    
    res = await client.get("/stores/STORE_BLR_002/funnel")
    assert res.status_code == 200
    funnel_data = res.json()["funnel"]
    
    # Entry count should be 1
    entry_stage = next(s for s in funnel_data if s["stage"] == "Entry")
    assert entry_stage["visitors"] == 1

async def test_funnel_reentry_continues_existing_session(client, seed_events):
    # Customer enters, exits, re-enters and joins billing queue. It should stay 1 session.
    vid = "VIS_reentry_u2"
    t1 = datetime.datetime.now(datetime.timezone.utc)
    ev1 = {**make_dummy_event(event_type="ENTRY"), "visitor_id": vid, "timestamp": t1}
    ev2 = {**make_dummy_event(event_type="REENTRY"), "visitor_id": vid, "timestamp": t1 + datetime.timedelta(seconds=10)}
    ev3 = {**make_dummy_event(event_type="ZONE_ENTER", zone_id="SKINCARE"), "visitor_id": vid, "timestamp": t1 + datetime.timedelta(seconds=20)}
    ev4 = {**make_dummy_event(event_type="BILLING_QUEUE_JOIN"), "visitor_id": vid, "timestamp": t1 + datetime.timedelta(seconds=30)}
    
    await seed_events([ev1, ev2, ev3, ev4])
    
    res = await client.get("/stores/STORE_BLR_002/funnel")
    assert res.status_code == 200
    funnel_data = res.json()["funnel"]
    
    # Billing queue should have 1 visitor
    billing_stage = next(s for s in funnel_data if s["stage"] == "Billing queue")
    assert billing_stage["visitors"] == 1

async def test_funnel_all_staff_clip_returns_zero_counts(client, seed_events):
    # Only staff events in the database
    ev1 = {**make_dummy_event(event_type="ENTRY", is_staff=True), "timestamp": datetime.datetime.now(datetime.timezone.utc)}
    ev2 = {**make_dummy_event(event_type="ZONE_ENTER", zone_id="SKINCARE", is_staff=True), "timestamp": datetime.datetime.now(datetime.timezone.utc)}
    
    await seed_events([ev1, ev2])
    
    res = await client.get("/stores/STORE_BLR_002/funnel")
    assert res.status_code == 200
    for stage in res.json()["funnel"]:
        assert stage["visitors"] == 0
        assert stage["dropoff_pct"] == 0.0

async def test_funnel_dropoff_percentages_sum_correctly(client, db_session, seed_events):
    # Seed 10 Entries -> 8 Zone visits -> 4 Billing queues -> 2 Purchases
    t1 = datetime.datetime.now(datetime.timezone.utc)
    
    events_to_insert = []
    # 10 unique entries
    for i in range(10):
        vid = f"VIS_funnel_{i}"
        events_to_insert.append({**make_dummy_event(event_type="ENTRY"), "visitor_id": vid, "timestamp": t1})
        # 8 zone visits
        if i < 8:
            events_to_insert.append({**make_dummy_event(event_type="ZONE_ENTER", zone_id="SKINCARE"), "visitor_id": vid, "timestamp": t1 + datetime.timedelta(seconds=10)})
        # 4 billing joins
        if i < 4:
            events_to_insert.append({**make_dummy_event(event_type="BILLING_QUEUE_JOIN"), "visitor_id": vid, "timestamp": t1 + datetime.timedelta(seconds=20)})

    await seed_events(events_to_insert)
    
    # 2 purchases
    tx_to_insert = []
    for i in range(2):
        vid = f"VIS_funnel_{i}"
        tx_to_insert.append({
            "transaction_id": f"TXN_{i}",
            "store_id": "STORE_BLR_002",
            "timestamp": t1 + datetime.timedelta(minutes=2),
            "basket_value": 200.0,
            "matched_visitor": vid
        })
    await db_session.execute(insert(DBPosTransaction).values(tx_to_insert))
    await db_session.commit()
    
    res = await client.get("/stores/STORE_BLR_002/funnel")
    assert res.status_code == 200
    funnel_data = res.json()["funnel"]
    
    # Drop-offs check:
    # Entry: 10, dropoff: 0.0
    # Zone visit: 8, dropoff: (10 - 8)/10 * 100 = 20.0
    # Billing queue: 4, dropoff: (8 - 4)/8 * 100 = 50.0
    # Purchase: 2, dropoff: (4 - 2)/4 * 100 = 50.0
    
    z_stage = next(s for s in funnel_data if s["stage"] == "Zone visit")
    b_stage = next(s for s in funnel_data if s["stage"] == "Billing queue")
    p_stage = next(s for s in funnel_data if s["stage"] == "Purchase")
    
    import pytest
    assert z_stage["dropoff_pct"] == pytest.approx(20.0, abs=0.1)
    assert b_stage["dropoff_pct"] == pytest.approx(50.0, abs=0.1)
    assert p_stage["dropoff_pct"] == pytest.approx(50.0, abs=0.1)
