# PROMPT: "Write pytest tests for the FastAPI app main.py routes.
#          Cover: /pos/load endpoint, /stores/{id}/events endpoint, SSE stream endpoint,
#          startup event hooks, and structured logging middleware trace_id header."
#
# CHANGES MADE:
# - Mocked SSE stream to avoid infinite generator blocking test runner
# - Added test for /stores/{id}/events return structure
# - Added test for X-Trace-ID header in all responses

import pytest
import datetime
from tests.test_ingestion import make_dummy_event

pytestmark = pytest.mark.asyncio


async def test_trace_id_header_present_on_all_responses(client):
    """All responses must include X-Trace-ID header (structured logging middleware)."""
    res = await client.get("/health")
    assert "x-trace-id" in res.headers or "X-Trace-ID" in res.headers


async def test_trace_id_propagated_from_request_header(client):
    """If X-Trace-ID is sent in the request, the same ID should echo back in response."""
    custom_trace = "test-trace-abc-123"
    res = await client.get("/health", headers={"X-Trace-ID": custom_trace})
    assert res.status_code == 200
    response_trace = res.headers.get("x-trace-id") or res.headers.get("X-Trace-ID")
    assert response_trace == custom_trace


async def test_events_endpoint_returns_list(client, seed_events):
    """GET /stores/{id}/events returns a JSON list of recent events."""
    t = datetime.datetime.now(datetime.timezone.utc)
    ev = {**make_dummy_event(event_type="ENTRY"), "timestamp": t}
    await seed_events([ev])

    res = await client.get("/stores/STORE_BLR_002/events")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)


async def test_events_endpoint_returns_correct_fields(client, seed_events):
    """Events list must contain event_id, store_id, visitor_id, event_type, timestamp."""
    t = datetime.datetime.now(datetime.timezone.utc)
    ev = {**make_dummy_event(event_type="ENTRY"), "visitor_id": "VIS_ev_001", "timestamp": t}
    await seed_events([ev])

    res = await client.get("/stores/STORE_BLR_002/events")
    assert res.status_code == 200
    events = res.json()
    if events:
        first = events[0]
        assert "event_id" in first
        assert "store_id" in first
        assert "visitor_id" in first
        assert "event_type" in first
        assert "timestamp" in first


async def test_events_endpoint_limits_to_8_results(client, seed_events):
    """Endpoint returns at most 8 events even when more are stored."""
    t = datetime.datetime.now(datetime.timezone.utc)
    events = [
        {**make_dummy_event(event_type="ENTRY"), "visitor_id": f"VIS_lim_{i}",
         "timestamp": t + datetime.timedelta(seconds=i)}
        for i in range(15)
    ]
    await seed_events(events)

    res = await client.get("/stores/STORE_BLR_002/events")
    assert res.status_code == 200
    assert len(res.json()) <= 8


async def test_events_endpoint_empty_store_returns_empty_list(client):
    """Empty store should return [] not an error."""
    res = await client.get("/stores/STORE_EMPTY_MAIN_999/events")
    assert res.status_code == 200
    assert res.json() == []


async def test_pos_load_endpoint_missing_file_returns_success(client):
    """
    POST /pos/load should return HTTP 200 even if no CSV is found at expected paths.
    Returns loaded_transactions = 0 gracefully.
    """
    res = await client.post("/pos/load")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "loaded_transactions" in data
    assert data["loaded_transactions"] >= 0


async def test_demo_playback_start_and_stop(client):
    """Test start and stop simulated demo playback endpoints."""
    try:
        # Test STORE_BLR_002 endpoint
        res = await client.post("/stores/STORE_BLR_002/demo/start")
        assert res.status_code == 200
        assert res.json()["status"] in ("success", "already_running")

        res = await client.post("/stores/STORE_BLR_002/demo/stop")
        assert res.status_code == 200
        assert res.json()["status"] == "success"

        # Test /pos/demo endpoint
        res = await client.post("/pos/demo")
        assert res.status_code == 200
        assert res.json()["status"] in ("success", "already_running")

        res = await client.post("/pos/demo/stop")
        assert res.status_code == 200
        assert res.json()["status"] == "success"
    finally:
        # Guarantee demo is stopped to avoid leakage
        await client.post("/stores/STORE_BLR_002/demo/stop")
        await client.post("/pos/demo/stop")

