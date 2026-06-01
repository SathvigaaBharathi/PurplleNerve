import time
import json
import uuid
import os
import logging
import asyncio
from datetime import datetime, timezone
from fastapi import FastAPI, Depends, Request, status, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from redis.asyncio import Redis

# Import routers
from app.health import router as health_router
from app.ingestion import router as ingestion_router
from app.metrics import router as metrics_router, compute_metrics_data
from app.funnel import router as funnel_router, compute_funnel_data
from app.heatmap import router as heatmap_router
from app.anomalies import router as anomalies_router, get_active_anomalies_data
from app.db import init_db, AsyncSessionLocal, get_db
from app.redis_client import get_redis, init_redis
from app.pos import load_pos_transactions_from_csv, correlate_transactions

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Retail Nerve System API",
    description="Analytics system for physical retail stores",
    version="1.0.0"
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Structured JSON Logging Middleware
@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):
    # 1. Generate trace_id on request ingress
    trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
    request.state.trace_id = trace_id
    
    # Initialise default states for logs
    request.state.store_ids = []
    request.state.event_count = None
    request.state.latency_ms = 0
    
    start_time = time.time()
    
    # Try parsing store_id from path
    store_id = None
    parts = request.url.path.split("/")
    if "stores" in parts:
        try:
            idx = parts.index("stores")
            if idx + 1 < len(parts):
                store_id = parts[idx + 1]
        except ValueError:
            pass
            
    # 2. Run route handler
    try:
        response = await call_next(request)
    except Exception as e:
        # Wrap database/connection errors in structured responses to avoid leaking raw traces
        logger.exception("Unhandled error in pipeline request")
        latency_ms = int((time.time() - start_time) * 1000)
        log_line = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", ".000Z"),
            "level": "ERROR",
            "trace_id": trace_id,
            "store_id": store_id,
            "endpoint": f"{request.method} {request.url.path}",
            "latency_ms": latency_ms,
            "event_count": None,
            "status_code": 500
        }
        print(json.dumps(log_line))
        
        return StreamingResponse(
            iter([json.dumps({
                "error": "INTERNAL_SERVER_ERROR", 
                "message": "An unexpected error occurred in the service", 
                "trace_id": trace_id
            }).encode()]),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json"
        )
        
    # 3. Log on egress
    latency_ms = int((time.time() - start_time) * 1000)
    
    # Overwrite store_id if ingestion middleware populated it
    if not store_id and len(getattr(request.state, "store_ids", [])) > 0:
        store_id = request.state.store_ids[0]
        
    log_line = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", ".000Z"),
        "level": "INFO",
        "trace_id": trace_id,
        "store_id": store_id,
        "endpoint": f"{request.method} {request.url.path}",
        "latency_ms": latency_ms,
        "event_count": getattr(request.state, "event_count", None),
        "status_code": response.status_code
    }
    
    print(json.dumps(log_line))
    
    response.headers["X-Trace-ID"] = trace_id
    return response

# Server-Sent Events stream endpoint
async def build_stream_payload(store_id: str, db, redis=None) -> dict:
    from app.metrics import compute_metrics_data
    from app.anomalies import get_active_anomalies_data
    from app.funnel import compute_funnel_data
    from app.health import get_active_cameras
    from datetime import datetime
    
    metrics = await compute_metrics_data(store_id, db)
    anomalies = await get_active_anomalies_data(store_id, db)
    funnel = await compute_funnel_data(store_id, db)
    active_cameras = await get_active_cameras(store_id, db)
    
    return {
        "metrics": metrics,
        "anomalies": anomalies,
        "funnel": funnel,
        "active_cameras": active_cameras,
        "camera_count": len(active_cameras),
        "server_ts": datetime.utcnow().isoformat() + "Z",
        "ping": True
    }

@app.get("/stores/{store_id}/stream")
async def event_stream(store_id: str, redis: Redis = Depends(get_redis)):
    """
    Server-Sent Events endpoint.
    Pushes metric updates every 2 seconds.
    Dashboard consumer group reads from Redis Streams.
    Each SSE message is a full metrics snapshot (not a diff).
    
    Connection drops gracefully on client disconnect.
    """
    # Check redis connection health first.
    # Redis unavailable -> Dashboard SSE endpoint returns 503 with retry header
    try:
        await redis.ping()
    except Exception as e:
        logger.error(f"Redis not available for stream: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Retry-After": "5"},
            detail="Redis Stream connection unavailable"
        )

    async def generate():
        while True:
            # We use local sessions to prevent connection leaks across streaming cycles
            async with AsyncSessionLocal() as db:
                try:
                    payload = await build_stream_payload(store_id, db, redis)
                    yield f"data: {json.dumps(payload)}\n\n"
                except Exception as e:
                    logger.error(f"Error generating SSE data: {e}")
                    yield f"data: {json.dumps({'error': 'DATA_RETRIEVAL_ERROR', 'message': str(e)})}\n\n"
                    
            await asyncio.sleep(2)
            
    return StreamingResponse(generate(), media_type="text/event-stream")

# POST /pos/load - endpoint to manually load POS data
@app.post("/pos/load")
async def load_pos(request: Request, db = Depends(get_db)):
    # Try finding the transactions CSV
    csv_paths = [
        "data/pos_transactions.csv",
        "/app/data/pos_transactions.csv",
        "../data/pos_transactions.csv",
        "D:/purplle/store-intelligence/data/pos_transactions.csv"
    ]
    loaded = 0
    for p in csv_paths:
        if os.path.exists(p):
            loaded = await load_pos_transactions_from_csv(p, db)
            await correlate_transactions(db)
            break
            
    return {"status": "success", "loaded_transactions": loaded}

# Background worker for POS correlation
async def pos_correlation_loop():
    logger.info("Starting background POS correlation loop...")
    while True:
        await asyncio.sleep(60)
        try:
            from app.redis_client import get_redis
            redis = await get_redis()
            async with AsyncSessionLocal() as db:
                await correlate_transactions(db, redis)
        except Exception as e:
            logger.error(f"Error in background POS correlation loop: {e}")

# GET /stores/{store_id}/events - Endpoint for fetching recent events
@app.get("/stores/{store_id}/events")
async def get_recent_events(store_id: str, db = Depends(get_db)):
    from sqlalchemy import select
    from app.db import DBEvent
    stmt = select(DBEvent).where(DBEvent.store_id == store_id).order_by(DBEvent.timestamp.desc()).limit(8)
    res = await db.execute(stmt)
    db_events = res.scalars().all()
    events_list = []
    for ev in db_events:
        events_list.append({
            "event_id": ev.event_id,
            "store_id": ev.store_id,
            "camera_id": ev.camera_id,
            "visitor_id": ev.visitor_id,
            "event_type": ev.event_type,
            "timestamp": ev.timestamp.isoformat(),
            "zone_id": ev.zone_id,
            "dwell_ms": ev.dwell_ms,
            "is_staff": ev.is_staff,
            "confidence": ev.confidence,
            "queue_depth": ev.queue_depth,
            "sku_zone": ev.sku_zone,
            "session_seq": ev.session_seq
        })
    return events_list

# Include Routers
app.include_router(health_router)
app.include_router(ingestion_router)
app.include_router(metrics_router)
app.include_router(funnel_router)
app.include_router(heatmap_router)
app.include_router(anomalies_router)

# Mount Dashboard static page
os.makedirs("dashboard", exist_ok=True)
app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")

@app.on_event("startup")
async def startup_event():
    # 1. Initialise DB tables & views
    await init_db()
    
    # 2. Initialise Redis Streams and Consumer Groups
    try:
        redis = await get_redis()
        await init_redis(redis)
        await redis.close()
    except Exception as e:
        logger.error(f"Failed to initialise Redis stream groups on startup: {e}")
        
    # 3. Load POS transactions from CSV if present
    async with AsyncSessionLocal() as db:
        csv_paths = [
            "data/pos_transactions.csv",
            "/app/data/pos_transactions.csv",
            "../data/pos_transactions.csv",
            "D:/purplle/store-intelligence/data/pos_transactions.csv"
        ]
        for p in csv_paths:
            if os.path.exists(p):
                await load_pos_transactions_from_csv(p, db)
                await correlate_transactions(db)
                break
                
    # 4. Fire background correlation task
    asyncio.create_task(pos_correlation_loop())


    # 6. Load store layout cameras for health check camera list
    try:
        from app.health import ALL_KNOWN_CAMERAS
        paths = [
            "data/store_layout.json",
            "/app/data/store_layout.json",
            "../data/store_layout.json",
            "store-intelligence/data/store_layout.json",
            "D:/purplle/store-intelligence/data/store_layout.json"
        ]
        layout = None
        for p in paths:
            if os.path.exists(p):
                with open(p, "r") as f:
                    layout = json.load(f)
                break
        if layout:
            for s_id, store_data in layout.items():
                if isinstance(store_data, dict) and "cameras" in store_data:
                    cameras_dict = store_data.get("cameras", {})
                    if isinstance(cameras_dict, dict):
                        ALL_KNOWN_CAMERAS[s_id] = list(cameras_dict.keys())
    except Exception as e:
        logger.error(f"Failed to load store layout cameras on startup: {e}")
