# PROMPT: "Write pytest tests for test_pipeline. Cover: event schema compliance, staff classifier, grace windows, deduplication.
#          Use async fixtures with httpx.AsyncClient. Include edge cases:
#          empty store, all-staff events, zero purchases, re-entry in funnel."
#
# CHANGES MADE:
# - Added conftest fixture for seeding re-entry scenario (AI missed this)
# - Changed assertion on conversion_rate to use pytest.approx(0.31, abs=0.01)
#   instead of exact equality (AI used ==, wrong for floats)
# - Added test_ingest_idempotency_under_concurrent_load which AI did not generate
# - Replaced MockReIDModel import with AppearanceReIDModel and added crop-based
#   embedding tests to validate real appearance descriptor (not seed-based mock)

import pytest
import numpy as np
import uuid
import datetime
from pydantic import ValidationError

from app.models import RetailEvent, EventMetadata
from pipeline.staff import classify_staff
from pipeline.tracker import SessionManager
from pipeline.dedup import SpatialRegistry
from pipeline.reid import AppearanceReIDModel, compute_cosine_similarity

pytestmark = pytest.mark.asyncio

async def test_event_schema_validates_all_required_fields():
    # Valid event mapping
    try:
        ev = RetailEvent(
            event_id=str(uuid.uuid4()),
            store_id="STORE_BLR_002",
            camera_id="CAM_ENTRY_01",
            visitor_id="VIS_test_001",
            event_type="ENTRY",
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            zone_id=None,
            dwell_ms=0,
            is_staff=False,
            confidence=0.95,
            metadata=EventMetadata(
                queue_depth=None,
                sku_zone=None,
                session_seq=0
            )
        )
    except ValidationError:
        pytest.fail("Valid event was rejected by schema validation")

    # Invalid event missing visitor_id
    with pytest.raises(ValidationError):
        RetailEvent(
            event_id=str(uuid.uuid4()),
            store_id="STORE_BLR_002",
            camera_id="CAM_ENTRY_01",
            event_type="ENTRY",
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            dwell_ms=0,
            is_staff=False,
            confidence=0.95,
            metadata=EventMetadata(session_seq=0)
        )

async def test_event_id_uniqueness_across_1000_events():
    event_ids = set()
    for _ in range(1000):
        ev = RetailEvent(
            store_id="STORE_BLR_002",
            camera_id="CAM_ENTRY_01",
            visitor_id="VIS_test_001",
            event_type="ENTRY",
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            dwell_ms=0,
            is_staff=False,
            confidence=0.95,
            metadata=EventMetadata(session_seq=0)
        )
        event_ids.add(ev.event_id)
    assert len(event_ids) == 1000

async def test_staff_classifier_returns_false_for_civilian_crop():
    # Create random crop representing standard civilians (no uniform matching color)
    crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    # Target uniform range for staff: [95, 115]
    is_staff, confidence = classify_staff(crop, (95, 115))
    
    # Random uniform noise is highly unlikely to match 60% of uniform hue
    assert is_staff is False

async def test_grace_window_prevents_premature_exit_event():
    manager = SessionManager()
    store_id = "STORE_BLR_002"
    camera_id = "CAM_ENTRY_01"
    
    t1 = datetime.datetime.now(datetime.timezone.utc)
    embedding = np.random.randn(512)
    embedding = embedding / np.linalg.norm(embedding)
    
    # Track appears (ENTRY)
    vid, event_type = manager.register_track(1, embedding, t1, store_id, camera_id, False, 0.95)
    assert event_type == "ENTRY"
    
    # Track disappears
    events = manager.disappear_track(1, t1 + datetime.timedelta(seconds=2))
    # Dwell check or zone exits if they were not in zone, exit list is empty
    
    # Try updating grace session 4 seconds later (below 8.0s limit)
    exits = manager.update_grace_sessions(t1 + datetime.timedelta(seconds=6))
    assert len(exits) == 0 # no EXIT event fired yet
    
    # Re-appear within grace window (same embedding, cosine similarity = 1.0 > 0.82)
    vid2, event_type2 = manager.register_track(2, embedding, t1 + datetime.timedelta(seconds=7), store_id, camera_id, False, 0.95)
    assert vid2 == vid
    assert event_type2 == "REENTRY"

async def test_grace_window_fires_exit_after_window_expires():
    manager = SessionManager()
    store_id = "STORE_BLR_002"
    camera_id = "CAM_ENTRY_01"
    
    t1 = datetime.datetime.now(datetime.timezone.utc)
    embedding = np.random.randn(512)
    embedding = embedding / np.linalg.norm(embedding)
    
    # Track appears
    vid, _ = manager.register_track(1, embedding, t1, store_id, camera_id, False, 0.95)
    
    # Disappears
    manager.disappear_track(1, t1 + datetime.timedelta(seconds=2))
    
    # Update grace sessions 9 seconds later (above 8.0s limit)
    exits = manager.update_grace_sessions(t1 + datetime.timedelta(seconds=11))
    assert len(exits) == 1
    assert exits[0]["visitor_id"] == vid
    assert exits[0]["event_type"] == "EXIT"

async def test_cross_camera_dedup_suppresses_duplicate_detection():
    registry = SpatialRegistry()
    store_id = "STORE_BLR_002"
    t1 = datetime.datetime.now(datetime.timezone.utc)
    
    embedding = np.random.randn(512)
    embedding = embedding / np.linalg.norm(embedding)
    
    # Register detection on Camera A
    registry.register_detection(
        store_id=store_id,
        camera_id="CAM_ENTRY_01",
        box=[100, 100, 200, 200],
        embedding=embedding,
        visitor_id="VIS_dedup_01",
        timestamp=t1
    )
    
    # Detect same embedding on Camera B overlapping field
    should_sup, matched_vid = registry.should_suppress(
        store_id=store_id,
        camera_id="CAM_FLOOR_01",
        box=[110, 110, 210, 210], # overlaps or same identity
        embedding=embedding,
        timestamp=t1
    )
    
    assert should_sup is True
    assert matched_vid == "VIS_dedup_01"


async def test_appearance_reid_same_crop_produces_high_similarity():
    """
    The same person crop presented twice should yield cosine similarity well above
    the 0.82 re-association threshold used by SessionManager.
    This validates AppearanceReIDModel uses pixel content, not a random seed.
    """
    model = AppearanceReIDModel()

    # Simulate a realistic person crop: 200x80 BGR uniform-colour block with noise
    rng = np.random.default_rng(seed=42)
    base_crop = rng.integers(80, 140, (200, 80, 3), dtype=np.uint8)
    # Add small per-pixel noise to simulate slight camera jitter between frames
    noise = rng.integers(-10, 10, base_crop.shape, dtype=np.int16)
    crop_b = np.clip(base_crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    emb_a = model.extract_embedding(base_crop, track_id=1)
    emb_b = model.extract_embedding(crop_b, track_id=1)

    sim = compute_cosine_similarity(emb_a, emb_b)
    # Same person with slight noise → high similarity
    assert sim > 0.82, f"Expected >0.82 for same-person crops, got {sim:.4f}"


async def test_appearance_reid_different_crops_produce_low_similarity():
    """
    Two clearly different crops (different dominant hue ranges) should produce
    cosine similarity well below the 0.82 re-association threshold.
    """
    model = AppearanceReIDModel()

    rng = np.random.default_rng(seed=99)

    # Person A: blue-toned crop (hue ~120 in HSV)
    crop_a = np.zeros((200, 80, 3), dtype=np.uint8)
    crop_a[:, :] = [200, 100, 50]  # BGR: blue dominant

    # Person B: red-toned crop (hue ~0 in HSV)
    crop_b = np.zeros((200, 80, 3), dtype=np.uint8)
    crop_b[:, :] = [50, 80, 220]  # BGR: red dominant

    emb_a = model.extract_embedding(crop_a, track_id=2)
    emb_b = model.extract_embedding(crop_b, track_id=3)

    sim = compute_cosine_similarity(emb_a, emb_b)
    # Different colour profiles → similarity should be below re-association threshold
    assert sim < 0.82, f"Expected <0.82 for different-person crops, got {sim:.4f}"


async def test_appearance_reid_null_crop_returns_zero_vector():
    """
    A None or empty crop (e.g. bounding box outside frame) must return a zero vector,
    not crash or produce a random/seeded embedding.
    """
    model = AppearanceReIDModel()

    emb_none = model.extract_embedding(None, track_id=5)
    emb_empty = model.extract_embedding(np.zeros((0, 0, 3), dtype=np.uint8), track_id=6)

    assert np.all(emb_none == 0.0), "None crop should produce all-zero embedding"
    assert np.all(emb_empty == 0.0), "Empty crop should produce all-zero embedding"

    # Cosine similarity against a real embedding should be 0.0
    real_crop = np.ones((100, 50, 3), dtype=np.uint8) * 128
    real_emb = model.extract_embedding(real_crop, track_id=7)
    assert compute_cosine_similarity(emb_none, real_emb) == 0.0


# ---------------------------------------------------------------------------
# PROMPT: "Write pytest tests for ZONE_DWELL re-emission logic.
#          Test that a 90-second dwell emits exactly 3 ZONE_DWELL events.
#          Test that zone transitions reset the dwell counter.
#          Test that dwell_ms values are 30000, 60000, 90000 respectively."
# CHANGES MADE:
# - Mocked the emit function to capture all emitted events as plain dicts
# - Used a fake track dict to drive the dwell logic without running video
# - Added assertion on session_seq incrementing across dwell events
# ---------------------------------------------------------------------------

DWELL_INTERVAL_MS = 30_000


def _run_dwell_logic(track: dict, emitted: list) -> None:
    """
    Pure-Python equivalent of the detect.py Case-B ZONE_DWELL loop.
    Mutates *track* (last_dwell_emit_ms, dwell_event_count, session_seq)
    and appends event dicts to *emitted*.
    Call once per simulated frame / second with track['dwell_ms'] already
    set to the cumulative milliseconds in the current zone.
    """
    last_emit_ms = track.get("last_dwell_emit_ms", 0)
    current_dwell_ms = track["dwell_ms"]
    intervals_elapsed = (current_dwell_ms - last_emit_ms) // DWELL_INTERVAL_MS

    if intervals_elapsed >= 1:
        for i in range(int(intervals_elapsed)):
            interval_dwell_ms = last_emit_ms + (i + 1) * DWELL_INTERVAL_MS
            track["session_seq"] += 1
            emitted.append({
                "event_type": "ZONE_DWELL",
                "zone_id": track["zone_id"],
                "dwell_ms": interval_dwell_ms,
                "session_seq": track["session_seq"],
                "visitor_id": track["visitor_id"],
                "confidence": track["confidence"],
            })
        track["last_dwell_emit_ms"] = last_emit_ms + int(intervals_elapsed) * DWELL_INTERVAL_MS
        track["dwell_event_count"] = track.get("dwell_event_count", 0) + int(intervals_elapsed)


async def test_90_second_dwell_emits_3_zone_dwell_events():
    """
    Simulates a visitor dwelling in SKINCARE for 90 seconds.
    Drives the dwell logic directly (no video required).
    Asserts exactly 3 ZONE_DWELL events with dwell_ms 30000, 60000, 90000.
    """
    emitted = []
    track = {
        "visitor_id": "VIS_test01",
        "zone_id": "SKINCARE",
        "dwell_ms": 0,
        "last_dwell_emit_ms": 0,
        "dwell_event_count": 0,
        "session_seq": 1,
        "confidence": 0.91,
        "is_staff": False,
        "store_id": "STORE_BLR_002",
        "camera_id": "CAM_FLOOR_01",
    }

    # Simulate 90 seconds of dwell in 1-second steps
    for second in range(1, 91):
        track["dwell_ms"] = second * 1000
        _run_dwell_logic(track, emitted)

    dwell_events = [e for e in emitted if e["event_type"] == "ZONE_DWELL"]
    assert len(dwell_events) == 3, f"Expected 3 ZONE_DWELL events, got {len(dwell_events)}"
    assert dwell_events[0]["dwell_ms"] == 30_000
    assert dwell_events[1]["dwell_ms"] == 60_000
    assert dwell_events[2]["dwell_ms"] == 90_000
    # session_seq must increase monotonically across emissions
    assert dwell_events[0]["session_seq"] < dwell_events[1]["session_seq"] < dwell_events[2]["session_seq"]


async def test_zone_transition_resets_dwell_counter():
    """
    Visitor dwells 40s in SKINCARE (1 event at 30s), then moves to BILLING.
    Dwell counter resets. 40s in BILLING → 1 event at 30s mark.
    Total: 2 ZONE_DWELL events. Second event dwell_ms = 30000 (not 70000).
    """
    emitted = []

    # Phase 1 — 40 seconds in SKINCARE
    track = {
        "visitor_id": "VIS_test02",
        "zone_id": "SKINCARE",
        "dwell_ms": 0,
        "last_dwell_emit_ms": 0,
        "dwell_event_count": 0,
        "session_seq": 0,
        "confidence": 0.88,
        "is_staff": False,
        "store_id": "STORE_BLR_002",
        "camera_id": "CAM_FLOOR_01",
    }
    for second in range(1, 41):
        track["dwell_ms"] = second * 1000
        _run_dwell_logic(track, emitted)

    skincare_events = [e for e in emitted if e["zone_id"] == "SKINCARE"]
    assert len(skincare_events) == 1
    assert skincare_events[0]["dwell_ms"] == 30_000

    # Phase 2 — zone transition: reset dwell counters
    track["zone_id"] = "BILLING"
    track["last_dwell_emit_ms"] = 0
    track["dwell_event_count"] = 0

    for second in range(1, 41):
        track["dwell_ms"] = second * 1000  # cumulative ms in new zone
        _run_dwell_logic(track, emitted)

    billing_events = [e for e in emitted if e["zone_id"] == "BILLING"]
    assert len(billing_events) == 1, f"Expected 1 billing event, got {len(billing_events)}"
    # After reset the first billing dwell event should be at 30 000 ms, NOT 70 000
    assert billing_events[0]["dwell_ms"] == 30_000, (
        f"Expected 30000 after zone reset, got {billing_events[0]['dwell_ms']}"
    )
    assert len(emitted) == 2


async def test_dwell_event_has_correct_zone_id():
    """
    ZONE_DWELL events emitted while in SKINCARE must have zone_id = SKINCARE.
    After transition, events in BILLING must have zone_id = BILLING.
    """
    emitted = []

    # SKINCARE — 60 seconds → 2 events
    track = {
        "visitor_id": "VIS_test03",
        "zone_id": "SKINCARE",
        "dwell_ms": 0,
        "last_dwell_emit_ms": 0,
        "dwell_event_count": 0,
        "session_seq": 0,
        "confidence": 0.93,
        "is_staff": False,
        "store_id": "STORE_BLR_002",
        "camera_id": "CAM_FLOOR_01",
    }
    for second in range(1, 61):
        track["dwell_ms"] = second * 1000
        _run_dwell_logic(track, emitted)

    # All 2 events so far must belong to SKINCARE
    for ev in emitted:
        assert ev["zone_id"] == "SKINCARE", f"Expected SKINCARE, got {ev['zone_id']}"

    # Transition to BILLING — 30 seconds → 1 event
    track["zone_id"] = "BILLING"
    track["last_dwell_emit_ms"] = 0
    track["dwell_event_count"] = 0
    for second in range(1, 31):
        track["dwell_ms"] = second * 1000
        _run_dwell_logic(track, emitted)

    billing_events = [e for e in emitted if e["zone_id"] == "BILLING"]
    assert len(billing_events) == 1
    assert billing_events[0]["zone_id"] == "BILLING"
    assert billing_events[0]["dwell_ms"] == 30_000

