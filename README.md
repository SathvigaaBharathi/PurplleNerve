# Store Intelligence API

## Setup (5 commands)

```bash
git clone <repo-url> && cd store-intelligence
cp .env.example .env                          # edit DATABASE_URL / REDIS_URL if needed
docker compose up -d postgres redis api       # starts API on :8000
docker compose run pipeline python pipeline/detect.py \
  --clip /clips/STORE_BLR_002_entry.mp4 \
  --store-id STORE_BLR_002 \
  --layout /data/store_layout.json \
  --output /output/events.jsonl               # run detection on a clip
curl -X POST http://localhost:8000/events/ingest \
  -H "Content-Type: application/json" \
  -d @output/events.jsonl                     # feed events to API
```

## Run All Clips at Once

```bash
bash pipeline/run.sh /clips /data/store_layout.json
```

This processes all 5 stores × 3 camera angles and streams events into the API automatically.

## Live Dashboard (Part E)

Open **http://localhost:8000/dashboard** in your browser.

The dashboard connects to the API via Server-Sent Events (SSE) and updates in real time as
events flow from the detection pipeline. Metrics, anomalies, zone heatmaps and the conversion
funnel refresh every 2 seconds without page reload.

## API Endpoints

| Endpoint | Description |
|---|---|
| `POST /events/ingest` | Ingest detection events (idempotent by event_id) |
| `GET /stores/{id}/metrics` | Live visitor count, conversion rate, dwell, queue depth |
| `GET /stores/{id}/funnel` | Entry → Zone → Billing → Purchase funnel with drop-off % |
| `GET /stores/{id}/heatmap` | Zone visit frequency + avg dwell, normalised 0–100 |
| `GET /stores/{id}/anomalies` | Active anomalies with severity and suggested action |
| `GET /health` | Service status, per-store last event timestamp, STALE_FEED warnings |

## Tests

```bash
docker compose run api pytest --cov=app --cov-report=term-missing
```

Statement coverage target: >70%.

## Health Check

```bash
curl http://localhost:8000/health
```

## Environment Variables

Copy `.env.example` to `.env`. Key variables:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://retail:retail_secret@postgres:5432/retail_intelligence` | PostgreSQL async connection string |
| `REDIS_URL` | `redis://redis:6379` | Redis connection string |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`) |
