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
                    metrics = await compute_metrics_data(store_id, db)
                    anomalies = await get_active_anomalies_data(store_id, db)
                    funnel = await compute_funnel_data(store_id, db)
                    
                    payload = {
                        "metrics": metrics,
                        "anomalies": anomalies,
                        "funnel": funnel
                    }
                    
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
            async with AsyncSessionLocal() as db:
                await correlate_transactions(db)
        except Exception as e:
            logger.error(f"Error in background POS correlation loop: {e}")

# Background worker for mock real-time events for Delhi and Mumbai stores
async def mock_realtime_generator_loop():
    import random
    import string
    
    if "PYTEST_CURRENT_TEST" in os.environ:
        return
        
    logger.info("Starting background mock event generator loop for Delhi/Mumbai stores...")
    
    active_visitors = {
        "STORE_MUM_001": [],
        "STORE_DEL_003": []
    }
    
    # Pre-populate some visitors
    for store in active_visitors:
        for _ in range(5):
            vid = f"VIS_{store.split('_')[1]}_{''.join(random.choices(string.ascii_lowercase + string.digits, k=6))}"
            active_visitors[store].append(vid)
            
    while True:
        await asyncio.sleep(4)
        try:
            # Pick a store
            store_id = random.choice(["STORE_MUM_001", "STORE_DEL_003"])
            
            # Select camera & event
            event_type = random.choice([
                "ENTRY", "ZONE_ENTER", "ZONE_DWELL", "BILLING_QUEUE_JOIN", 
                "BILLING_QUEUE_LEAVE", "ZONE_EXIT", "EXIT"
            ])
            
            visitors = active_visitors[store_id]
            
            if event_type == "ENTRY" or not visitors:
                visitor_id = f"VIS_{store_id.split('_')[1]}_{''.join(random.choices(string.ascii_lowercase + string.digits, k=6))}"
                visitors.append(visitor_id)
                camera_id = "CAM_ENTRY_01"
                zone_id = None
                dwell_ms = 0
                queue_depth = None
            else:
                visitor_id = random.choice(visitors)
                if event_type == "EXIT":
                    if visitor_id in visitors:
                        visitors.remove(visitor_id)
                    camera_id = "CAM_ENTRY_01"
                    zone_id = None
                    dwell_ms = random.randint(10000, 180000)
                    queue_depth = None
                elif event_type == "BILLING_QUEUE_JOIN":
                    camera_id = "CAM_BILLING_01"
                    zone_id = "BILLING"
                    dwell_ms = 0
                    queue_depth = random.randint(1, 5)
                elif event_type == "ZONE_ENTER":
                    camera_id = "CAM_FLOOR_01"
                    zone_id = random.choice(["SKINCARE", "MOISTURISER"])
                    dwell_ms = 0
                    queue_depth = None
                elif event_type == "ZONE_DWELL":
                    camera_id = "CAM_FLOOR_01"
                    zone_id = random.choice(["SKINCARE", "MOISTURISER"])
                    dwell_ms = random.randint(15000, 60000)
                    queue_depth = None
                else: # ZONE_EXIT / LEAVE
                    camera_id = "CAM_FLOOR_01" if event_type == "ZONE_EXIT" else "CAM_BILLING_01"
                    zone_id = "BILLING" if camera_id == "CAM_BILLING_01" else random.choice(["SKINCARE", "MOISTURISER"])
                    dwell_ms = random.randint(20000, 120000)
                    queue_depth = None
            
            # Insert event
            async with AsyncSessionLocal() as db:
                from app.db import DBEvent
                ev = DBEvent(
                    event_id=str(uuid.uuid4()),
                    store_id=store_id,
                    camera_id=camera_id,
                    visitor_id=visitor_id,
                    event_type=event_type,
                    timestamp=datetime.now(timezone.utc),
                    zone_id=zone_id,
                    dwell_ms=dwell_ms,
                    is_staff=random.random() < 0.08,
                    confidence=round(random.uniform(0.6, 0.98), 2),
                    queue_depth=queue_depth,
                    session_seq=random.randint(0, 4)
                )
                db.add(ev)
                await db.commit()
                
                # Refresh aggregated view if zone_dwell was added
                if event_type == "ZONE_DWELL":
                    await db.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY zone_dwell_agg;"))
                    await db.commit()
                    
        except Exception as e:
            logger.error(f"Error in mock realtime event generator: {e}")

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
    
    # 5. Fire background mock event generator loop for Delhi/Mumbai
    asyncio.create_task(mock_realtime_generator_loop())
