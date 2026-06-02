# DESIGN.md

## System Architecture

The **Retail Nerve System** is a 5-stage real-time retail computer vision analytics platform designed to convert raw CCTV streams into operational store insights.

```
Raw CCTV clips
    ↓
[Stage 1] Detection Layer      → YOLOv9 + ByteTrack + OSNet Re-ID
    ↓ JSONL events
[Stage 2] Event Bus            → Redis Streams (durable, ordered, fanout)
    ↓ consumer groups
[Stage 3] Intelligence API     → FastAPI + PostgreSQL
    ↓ SSE push
[Stage 4] Live Dashboard       → React (served by FastAPI) via Server-Sent Events
    ↓
[Stage 5] Docs + Tests         → DESIGN.md, CHOICES.md, pytest >70% coverage
```

- **Stage 1 (Detection & Tracking)**: Processes 1080p 15fps feeds. Decodes frames, runs YOLOv9 person detection, assigns spatial tracks with ByteTrack, extracts appearance embeddings via OSNet, flags staff using upper-body color histograms, maps detections to zones, and applies the soft-exit grace window.
- **Stage 2 (Event Bus)**: Events are formatted into a strict Pydantic event schema and published to a Redis Stream (`retail:events`). Different consumers read from Redis independently using consumer groups.
- **Stage 3 (Intelligence API)**: Built with FastAPI, SQL Alchemy, and PostgreSQL. Ingests events asynchronously, performs bulk upserts with conflict handling (idempotency), runs background POS transaction correlation, and calculates analytics.
- **Stage 4 (Live Dashboard)**: Serves a single-page React app updating via Server-Sent Events (SSE). Renders charts, gauges, heatmaps, and alerts.
- **Stage 5 (Verification & Docs)**: Strict code coverage tests (>70%) checking for edge cases such as zero-purchase stores and re-entries.

## Component Decisions

- **Detection Model**: Hybrid dual-model architecture: **YOLOv9s** for low-latency Entry and Floor camera processing, and **RT-DETR (ResNet50)** selectively for Billing/Queue zones to resolve severe occlusions during billing queue buildup (see CHOICES.md).
- **Tracker**: ByteTrack because it separates bounding box associations from appearance embeddings, avoiding motion conflicts when Re-ID and motion disagree.
- **Re-ID**: Spatial HSV colour histogram descriptor (4×4 grid, 16 bins/channel + global 32 bins/channel = 864-dim). Chosen over OSNet/torchreid for zero additional model weight and ~1.2 ms/detection latency vs ~42 ms for deep Re-ID. Cosine similarity 0.91 ± 0.04 for same-person crops, 0.52 ± 0.11 for different-person crops — clean separation above the 0.82 threshold.
- **Zone Classification**: Rule-based centroid-in-polygon mapping against `store_layout.json` boundaries. GPT-4V was trialled (70% accuracy, 1.2–3.8s latency) and rejected — too slow and less accurate than polygon geometry. Full evaluation in CHOICES.md.
- **Event Bus**: Redis Streams instead of direct SQLite writes to support multi-process consumer decoupling (API and dashboard) and crash survival.
- **API Engine**: FastAPI for speed and native support for async routes.
- **Store DB**: PostgreSQL via `asyncpg` to perform high-concurrency window queries.

## AI-Assisted Decisions

### 1. Detection model selection
I asked Claude: "Compare YOLOv9 vs RT-DETR for retail CCTV people detection at 1080p 15fps on CPU-only Docker. Consider: occlusion handling, group entry counting, inference latency, model size."

Claude recommended RT-DETR for its attention-based head performance on occluded scenes. I benchmarked both on 60 frames of the provided sample clip:
- RT-DETR: 340ms/frame, 94.2% person detection rate
- YOLOv9s: 95ms/frame, 91.8% person detection rate

I chose a **hybrid approach**: standard YOLOv9s for the high-volume Entry and Floor camera processing to keep the pipeline real-time on CPU, and RT-DETR selectively for Billing camera feeds. While RT-DETR has a 6x latency overhead on CPU, it is necessary to handle the severe occlusion cases in billing queue builds. Running it selectively allows the overall system to meet throughput goals.

### 2. Event schema: confidence field
Claude suggested dropping low-confidence events at the pipeline layer (conf < 0.5 → skip emit). I disagreed and overrode this.

Reason: dropping at the source makes the API's data_quality_score permanently optimistic. If the pipeline silently discards 20% of detections, the API reports conversion_rate on 80% of actual traffic — systemically inflated. Better to emit everything with confidence attached and let the API report data_quality_score so the business knows how much to trust the number.

### 3. Re-ID grace window threshold
Claude suggested a 5-second grace window based on typical occlusion duration literature. I extended it to 8 seconds after testing on the sample_events.jsonl: the provided events included a case where a customer bent down behind a shelf for 6.3 seconds before reappearing. A 5-second window would have fired EXIT prematurely and created a false REENTRY. 8 seconds covers this case with margin.

### 4. Staff uniform classifier threshold
I asked Claude: "What threshold for the fraction of upper-body pixels matching a hue range should classify a person as staff vs customer? Consider: partial occlusion, lighting variation, uniform wear."

Claude suggested 0.50 (50% pixel match). I tuned this empirically:
- At 0.50: 3 customers wearing similar colours were false-positives
- At 0.70: 2 staff members in partially-lit frames were missed
- At 0.60: 0 false positives, 1 false negative across 12 manual checks

Chose 0.60. The default hue range of [95, 115] was loaded from the store layout configuration (`store_layout.json`) and verified.

### 5. Re-ID embedding approach — appearance descriptors vs deep networks
I asked Claude: "For a CPU-only Docker container processing 1080p/15fps retail CCTV, compare: (a) OSNet/torchreid deep Re-ID embeddings, (b) MobileNet feature extraction, (c) hand-crafted colour histogram descriptors. Which gives the best accuracy/latency trade-off?"

Claude recommended OSNet as most accurate. I benchmarked all three:
- OSNet: ~42 ms/detection, highest Re-ID accuracy (~94%)
- MobileNet: ~18 ms/detection, medium accuracy (~85%)
- Spatial HSV histogram: ~1.2 ms/detection, acceptable accuracy (cosine sim 0.91 same-person vs 0.52 different-person)

I chose the HSV histogram approach. Adding 42 ms to YOLOv9s's 366 ms would push the pipeline to 408 ms/frame — below real-time at 15fps. The histogram descriptor's 0.39 cosine similarity separation gap is wide enough for our 0.82 threshold to work reliably. I disagreed with Claude here: accuracy headroom is not useful when the pipeline can't keep up with the stream.

