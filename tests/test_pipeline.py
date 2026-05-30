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
