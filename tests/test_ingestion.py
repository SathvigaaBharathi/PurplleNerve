# PROMPT: "Write pytest tests for test_ingestion. Cover: ingestion endpoints, validation, idempotency, partial success, performance.
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
import time
import asyncio
from datetime import datetime, timezone
from sqlalchemy import select
from app.db import DBEvent

pytestmark = pytest.mark.asyncio

def make_dummy_event(event_id=None, is_staff=False, event_type="ENTRY", zone_id=None):
    eid = event_id or str(uuid.uuid4())
    return {
        "event_id": eid,
        "store_id": "STORE_BLR_002",
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": "VIS_test_123",
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "zone_id": zone_id,
        "dwell_ms": 0,
        "is_staff": is_staff,
        "confidence": 0.95,
        "metadata": {
            "queue_depth": None,
            "sku_zone": None,
            "session_seq": 0
        }
    }

async def test_ingest_accepts_valid_batch(client, db_session):
    event = make_dummy_event()
    res = await client.post("/events/ingest", json=[event])
    assert res.status_code == 200
    res_data = res.json()
    assert res_data["accepted"] == 1
    assert res_data["rejected"] == 0
    assert "trace_id" in res_data

async def test_ingest_idempotent_same_event_id(client, db_session):
    event = make_dummy_event()
    # First post
    res1 = await client.post("/events/ingest", json=[event])
    assert res1.status_code == 200
    # Second post (same payload)
    res2 = await client.post("/events/ingest", json=[event])
    assert res2.status_code == 200
    
    # Confirm only one row exists in DB
    result = await db_session.execute(select(DBEvent).where(DBEvent.event_id == event["event_id"]))
    rows = result.scalars().all()
    assert len(rows) == 1

async def test_ingest_partial_failure_returns_accepted_and_rejected(client, db_session):
    valid_ev = make_dummy_event()
    # Invalid event missing store_id
    invalid_ev = make_dummy_event()
    invalid_ev.pop("store_id")
    
    res = await client.post("/events/ingest", json=[valid_ev, invalid_ev])
    # Returns 207 Multi Status since some succeeded and some failed
    assert res.status_code == 207
    res_data = res.json()
    assert res_data["accepted"] == 1
    assert res_data["rejected"] == 1
    assert len(res_data["errors"]) == 1

async def test_ingest_rejects_malformed_schema(client, db_session):
    # Malformed schema (missing visitor_id, wrong confidence data type)
    bad_ev = {
        "event_id": str(uuid.uuid4()),
        "store_id": "STORE_BLR_002",
        "camera_id": "CAM_ENTRY_01",
        "event_type": "ENTRY",
        "timestamp": "invalid-time",
        "confidence": "super_confident"
    }
    res = await client.post("/events/ingest", json=[bad_ev])
    assert res.status_code == 207 or res.status_code == 400
    res_data = res.json()
    assert res_data["accepted"] == 0
    assert res_data["rejected"] == 1

async def test_ingest_500_event_batch_under_2_seconds(client, db_session):
    batch = [make_dummy_event() for _ in range(500)]
    start = time.time()
    res = await client.post("/events/ingest", json=batch)
    duration = time.time() - start
    
    assert res.status_code == 200
    assert duration < 2.0
    assert res.json()["accepted"] == 500

async def test_ingest_returns_trace_id(client, db_session):
    event = make_dummy_event()
    res = await client.post("/events/ingest", json=[event])
    assert "trace_id" in res.json()
    assert "X-Trace-ID" in res.headers

async def test_ingest_idempotency_under_concurrent_load(client, db_session):
    event = make_dummy_event()
    # Dispatch duplicate payloads concurrently
    tasks = [client.post("/events/ingest", json=[event]) for _ in range(5)]
    results = await asyncio.gather(*tasks)
    
    for r in results:
        assert r.status_code == 200
        
    # DB check
    result = await db_session.execute(select(DBEvent).where(DBEvent.event_id == event["event_id"]))
    rows = result.scalars().all()
    assert len(rows) == 1
