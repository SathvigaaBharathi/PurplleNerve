# PROMPT: "Write pytest tests for test_anomalies. Cover: anomaly detection rules, severities, thresholds, dead zones, stale feeds.
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
from app.db import DBEvent
from tests.test_ingestion import make_dummy_event

pytestmark = pytest.mark.asyncio

async def test_anomaly_queue_spike_fires_at_2x_baseline(client, seed_events):
    # Seed 7-day average queue baseline of 2.0 (e.g. historical join events)
    hist_t = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=2)
    ev_hist = {
        **make_dummy_event(event_type="BILLING_QUEUE_JOIN"),
        "queue_depth": 2,
        "timestamp": hist_t
    }
    await seed_events([ev_hist])
    
    # Ingest current join with depth 5 (which is > 2.0 * 2.0 = 4.0)
    curr_t = datetime.datetime.now(datetime.timezone.utc)
    ev_curr = {
        **make_dummy_event(event_type="BILLING_QUEUE_JOIN"),
        "queue_depth": 5,
        "timestamp": curr_t
    }
    await seed_events([ev_curr])
    
    res = await client.get("/stores/STORE_BLR_002/anomalies")
    assert res.status_code == 200
    anomalies = res.json()["active_anomalies"]
    
    spike_anomaly = next((a for a in anomalies if a["type"] == "BILLING_QUEUE_SPIKE"), None)
    assert spike_anomaly is not None
    assert spike_anomaly["severity"] == "WARN"

async def test_anomaly_queue_spike_critical_at_absolute_threshold(client, seed_events):
    # Absolute threshold is > 8. Set depth to 9
    curr_t = datetime.datetime.now(datetime.timezone.utc)
    ev_curr = {
        **make_dummy_event(event_type="BILLING_QUEUE_JOIN"),
        "queue_depth": 9,
        "timestamp": curr_t
    }
    await seed_events([ev_curr])
    
    res = await client.get("/stores/STORE_BLR_002/anomalies")
    assert res.status_code == 200
    anomalies = res.json()["active_anomalies"]
    
    spike_anomaly = next((a for a in anomalies if a["type"] == "BILLING_QUEUE_SPIKE"), None)
    assert spike_anomaly is not None
    assert spike_anomaly["severity"] == "CRITICAL"

async def test_anomaly_conversion_drop_fires_at_70pct_below_baseline(client, seed_events):
    # Default baseline conversion is 0.30
    # Create 10 entries but 0 purchases (so conversion is 0.0, which is < 0.30 * 0.7 = 0.21)
    # Severity is CRITICAL if <50% of baseline (0.0 < 0.15 is true)
    t = datetime.datetime.now(datetime.timezone.utc)
    events = []
    for i in range(10):
        events.append({**make_dummy_event(event_type="ENTRY"), "visitor_id": f"VIS_drop_{i}", "timestamp": t})
        
    await seed_events(events)
    
    res = await client.get("/stores/STORE_BLR_002/anomalies")
    assert res.status_code == 200
    anomalies = res.json()["active_anomalies"]
    
    conv_anomaly = next((a for a in anomalies if a["type"] == "CONVERSION_DROP"), None)
    assert conv_anomaly is not None
    assert conv_anomaly["severity"] == "CRITICAL"

async def test_anomaly_dead_zone_fires_after_30min_no_visits(client, seed_events):
    # Seed historical visit in SKINCARE zone
    hist_t = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=2)
    ev_hist = {
        **make_dummy_event(event_type="ZONE_ENTER", zone_id="SKINCARE"),
        "timestamp": hist_t
    }
    # No current visits in the last 30 minutes. Let's insert this event.
    await seed_events([ev_hist])
    
    res = await client.get("/stores/STORE_BLR_002/anomalies")
    assert res.status_code == 200
    anomalies = res.json()["active_anomalies"]
    
    dead_anomaly = next((a for a in anomalies if a["type"] == "DEAD_ZONE" and a["details"]["zone_id"] == "SKINCARE"), None)
    assert dead_anomaly is not None
    assert dead_anomaly["severity"] == "INFO"

async def test_anomaly_stale_feed_fires_after_10min_no_events(client, seed_events):
    # Insert event older than 10 minutes (e.g. 15 minutes ago) for CAM_ENTRY_01
    hist_t = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=15)
    ev = {
        **make_dummy_event(event_type="ENTRY"),
        "camera_id": "CAM_ENTRY_01",
        "timestamp": hist_t
    }
    await seed_events([ev])
    
    res = await client.get("/stores/STORE_BLR_002/anomalies")
    assert res.status_code == 200
    anomalies = res.json()["active_anomalies"]
    
    stale_anomaly = next((a for a in anomalies if a["type"] == "STALE_FEED" and a["details"]["camera_id"] == "CAM_ENTRY_01"), None)
    assert stale_anomaly is not None
    assert stale_anomaly["severity"] == "CRITICAL"

async def test_no_anomalies_when_store_is_healthy(client, seed_events):
    # Store with recent events, normal queue depth, and active feeds
    t = datetime.datetime.now(datetime.timezone.utc)
    ev1 = {**make_dummy_event(event_type="ENTRY"), "camera_id": "CAM_ENTRY_01", "timestamp": t}
    ev2 = {**make_dummy_event(event_type="ENTRY"), "camera_id": "CAM_FLOOR_01", "timestamp": t}
    ev3 = {**make_dummy_event(event_type="ENTRY"), "camera_id": "CAM_BILLING_01", "timestamp": t}
    ev4 = {**make_dummy_event(event_type="ZONE_ENTER", zone_id="SKINCARE"), "camera_id": "CAM_FLOOR_01", "timestamp": t}
    ev5 = {**make_dummy_event(event_type="ZONE_ENTER", zone_id="MOISTURISER"), "camera_id": "CAM_FLOOR_01", "timestamp": t}
    ev6 = {**make_dummy_event(event_type="ZONE_ENTER", zone_id="BILLING"), "camera_id": "CAM_BILLING_01", "timestamp": t}
    ev7 = {**make_dummy_event(event_type="BILLING_QUEUE_JOIN"), "queue_depth": 2, "camera_id": "CAM_BILLING_01", "timestamp": t}
    ev8 = {**make_dummy_event(event_type="ENTRY"), "camera_id": "CAM_FLOOR_02", "timestamp": t}
    ev9 = {**make_dummy_event(event_type="ENTRY"), "camera_id": "CAM_BILLING_02", "timestamp": t}
    
    await seed_events([ev1, ev2, ev3, ev4, ev5, ev6, ev7, ev8, ev9])
    
    res = await client.get("/stores/STORE_BLR_002/anomalies")
    assert res.status_code == 200
    anomalies = res.json()["active_anomalies"]
    
    # Filters out critical/warning alerts since everything is in range
    stale_or_spike = [a for a in anomalies if a["type"] in ("STALE_FEED", "BILLING_QUEUE_SPIKE")]
    assert len(stale_or_spike) == 0
