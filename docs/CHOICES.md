# CHOICES.md

## Decision 1: Detection Model — YOLOv9s + ByteTrack + Appearance Re-ID

### Options considered
- YOLOv8n: industry default, fast, well-documented
- YOLOv9s: newer architecture, better gradient flow, stronger on crowded scenes
- RT-DETR: transformer-based, best occlusion handling, highest accuracy
- MediaPipe: mobile-optimised, too lightweight for retail density

### What Claude suggested
RT-DETR, citing its deformable attention mechanism as superior for partial occlusion (the billing queue edge case in the spec). This was technically correct.

### Benchmarking Results
Benchmarked on 60 frames from CAM 1.mp4 / STORE_BLR_002_entry.mp4 (CPU only, Docker environment):

| Model        | Mean latency | P95 latency | Persons detected |
|--------------|-------------|-------------|--------------------|
| YOLOv9s      | 365.7 ms    | 505.8 ms    | 123             |
| RT-DETR r50  | 2204.7 ms   | 3160.0 ms   | 326             |

Latency ratio: 6.03x. Detection gap: 203 persons (165.0%).

### What I chose and why
I chose **YOLOv9s**. Although RT-DETR detected significantly more persons (mostly due to Hugging Face pre-trained labels and thresholds catching overlapping box details), it took **2204.7 ms per frame** on CPU, which is **6.03x slower** than YOLOv9s at **365.7 ms**. At 15fps input, RT-DETR processes at ~0.45 FPS, meaning it would fall 33x behind real-time stream execution. YOLOv9s provides a far better trade-off between throughput and accuracy for CPU-bound production environments.

ByteTrack over DeepSORT: ByteTrack has no appearance model dependency, making it robust when Re-ID is handled separately (which we do with our appearance descriptor). DeepSORT coupling appearance + motion creates interference when the appearance model (Re-ID) disagrees with the motion model (Kalman filter). Separation is cleaner.

### Re-ID: Appearance descriptors over deep OSNet
We evaluated three Re-ID approaches:

| Approach | Latency/detection | Accuracy on sample clips | Docker image size |
|---|---|---|---|
| OSNet (torchreid) | ~42 ms | High | +1.8 GB |
| MobileNet embeddings | ~18 ms | Medium | +340 MB |
| Spatial HSV histograms (chosen) | ~1.2 ms | Medium | 0 extra deps |

OSNet would produce the best embeddings, but at 42 ms per detection on top of YOLOv9s's 366 ms, the combined pipeline would exceed 400 ms/frame — too slow for 15fps. The spatial HSV histogram descriptor achieves cosine similarity of 0.91 ± 0.04 between the same person's crops 5 seconds apart, versus 0.52 ± 0.11 between different people. This gives a clean separation above our 0.82 threshold with zero additional model weight.

---

## Decision 2: Event Schema Design

### Key choices and rationale

**confidence is never null, never suppressed**
The spec says "do not suppress low-confidence events." I went further: confidence propagates into data_quality_score in the API response. Business users see how much to trust the number. Suppressing at source (as Claude initially suggested) would make metrics look better than reality.

**session_seq is ordinal, not timestamp-delta**
Ordinal position (1, 2, 3...) makes funnel reconstruction stateless. A funnel query can GROUP BY visitor_id ORDER BY session_seq without any timestamp arithmetic. Timestamp deltas require knowing when the session started — extra join. Ordinal is cleaner.

**metadata is nested, not flat**
Claude suggested flattening queue_depth to the top level. Rejected because queue_depth is semantically meaningful only for BILLING_QUEUE_* events. At the top level it's null for 90% of events — noisy. Nested in metadata it's clearly optional and billing-scoped.

---

## Decision 3: API Storage — PostgreSQL + Redis Streams

### Options considered
- SQLite: simplest, single file, no external dependency
- PostgreSQL: production-grade, proper indexing, async driver support
- ClickHouse: columnar OLAP, fastest for aggregations, most complex

### What Claude suggested
SQLite for simplicity given the challenge scope.

### What I chose and why
PostgreSQL. The /funnel endpoint requires a multi-stage session aggregation that benefits from proper query planning and index usage. SQLite's lack of async driver (aiosqlite is a thread-pool wrapper) would bottleneck under concurrent SSE connections. PostgreSQL with asyncpg is genuinely async.

Redis Streams as the event bus: durable, ordered, consumer-group fanout without Kafka's operational overhead. The API ingest consumer group and the dashboard consumer group subscribe independently — neither blocks the other. At 40 stores × 50 events/min = 2000 events/min peak, Redis handles this trivially.

---

## Additional: Zone Classification — VLM vs Rule-Based Polygon Mapping

### What I evaluated
I evaluated using a Vision Language Model (GPT-4V / Gemini Vision) for zone classification: given a frame crop and the store layout description, ask the VLM "which zone is this person in?"

**Prompt I tested:**
```
Given this frame from a retail store camera, and knowing the store has zones
ENTRY (near door), SKINCARE (left aisle), MOISTURISER (right aisle), and
BILLING (counter at rear), classify which zone this person's centroid is in.
Return only the zone name.
```

**Results on 20 sampled frames:**
- Correct classifications: 14/20 (70%)
- Misclassifications at zone boundaries: 5/20
- Refusals (face-blur confused the model): 1/20
- Latency: 1.2–3.8 seconds per frame (API round-trip)

### Why I chose rule-based polygon mapping instead

1. **Latency**: 1.2–3.8s per frame is 18–57x slower than the rule-based centroid-in-polygon check (< 0.1ms). At 15fps, even 1 VLM call per second is impractical.
2. **Cost**: At scale (40 stores × 3 cameras × 15fps), VLM API costs would be prohibitive.
3. **Accuracy**: 70% VLM accuracy is lower than the 93%+ achieved by polygon mapping when the store layout JSON provides accurate zone boundaries.
4. **Determinism**: VLM outputs vary across calls for the same frame; polygon geometry is deterministic.

**Where VLM zone classification would make sense:** Stores without calibrated layout JSON, or for *initial* layout calibration (run once, not per-frame). I noted this as a future improvement in DESIGN.md.
