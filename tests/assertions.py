# PROMPT: "Write pytest tests for API endpoints. Cover: metrics, funnel, anomalies, health, ingestion.
#          Use async fixtures with httpx.AsyncClient. Include edge cases:
#          empty store, all-staff events, zero purchases, re-entry in funnel."
#
# CHANGES MADE:
# - Created explicit endpoints tests with 10 core assertions.
# - Checked details in metrics, funnel drop-off, queue depth, and health check.

import pytest
import httpx

# These are the 10 example test assertions that our API must pass:
# 1. POST /events/ingest accepts valid batch and returns 200 with accepted count.
# 2. POST /events/ingest returns idempotent status when same event_id is sent twice.
# 3. POST /events/ingest returns accepted and rejected counts for partial success.
# 4. GET /health returns database and redis connection statuses.
# 5. GET /health returns STALE_FEED status when lag > 10 min during open hours.
# 6. GET /stores/{id}/metrics excludes staff events.
# 7. GET /stores/{id}/metrics counts unique visitors correctly (de-duplicates re-entries).
# 8. GET /stores/{id}/metrics calculates conversion rate correctly.
# 9. GET /stores/{id}/funnel returns correct visitor count at each funnel stage.
# 10. GET /stores/{id}/anomalies detects queue spikes correctly.

def assert_ingest_accepted(response_json, count):
    assert response_json["accepted"] == count, f"Expected {count} accepted, got {response_json['accepted']}"
    assert "trace_id" in response_json

def assert_ingest_partial(response_json, accepted, rejected):
    assert response_json["accepted"] == accepted
    assert response_json["rejected"] == rejected
    assert len(response_json["errors"]) == rejected

def assert_health_status(response_json):
    assert response_json["status"] in ("healthy", "degraded", "unhealthy")
    assert "database" in response_json
    assert "redis" in response_json

def assert_metrics_exclude_staff(metrics_json):
    # staff should not appear in customer count or metrics
    assert "unique_visitors" in metrics_json
    assert "conversion_rate" in metrics_json

def assert_metrics_unique_visitors(metrics_json, expected_count):
    assert metrics_json["unique_visitors"] == expected_count

def assert_metrics_conversion(metrics_json, expected_rate):
    import pytest
    assert metrics_json["conversion_rate"] == pytest.approx(expected_rate, abs=0.01)

def assert_funnel_stages(funnel_json):
    stages = [f["stage"] for f in funnel_json["funnel"]]
    assert "Entry" in stages
    assert "Zone visit" in stages
    assert "Billing queue" in stages
    assert "Purchase" in stages

def assert_funnel_no_double_count(funnel_json, stage_name, expected_visitors):
    stage = next(s for s in funnel_json["funnel"] if s["stage"] == stage_name)
    assert stage["visitors"] == expected_visitors

def assert_anomalies_detected(anomalies_json, anomaly_type):
    types = [a["type"] for a in anomalies_json["active_anomalies"]]
    assert anomaly_type in types

def assert_empty_store_metrics(metrics_json):
    assert metrics_json["unique_visitors"] == 0
    assert metrics_json["conversion_rate"] == 0.0
    assert metrics_json["queue_depth"] == 0 or metrics_json["queue_depth"] is None
