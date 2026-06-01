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

# Global cache for latest camera frames for MJPEG video streaming
latest_frames = {}
latest_frame_times = {}

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
    
    # Merge with cameras actively uploading frames in the last 15 seconds
    now = time.time()
    for key, upload_time in list(latest_frame_times.items()):
        if key.startswith(f"{store_id}_") and (now - upload_time) < 15.0:
            cam_id = key.split(f"{store_id}_", 1)[1]
            if cam_id not in active_cameras:
                active_cameras.append(cam_id)
    
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

# Shared Re-ID Registry for cross-camera synchronization
global_reid_sessions = {}

@app.post("/stores/{store_id}/reid/sync")
async def sync_reid_tracks(store_id: str, request: Request):
    """
    Syncs local camera tracks with the global Re-ID registry.
    Consolidates visitor_ids and is_staff statuses across all cameras.
    """
    import numpy as np
    
    payload = await request.json()
    current_time = time.time()
    
    # Prune expired sessions (not seen for > 30 seconds)
    expired_keys = [
        vid for vid, data in global_reid_sessions.items()
        if current_time - data["last_seen"] > 30.0
    ]
    for vid in expired_keys:
        global_reid_sessions.pop(vid, None)
        
    synced_tracks = []
    incoming_tracks = payload.get("tracks", [])
    
    for track in incoming_tracks:
        t_id = track.get("track_id")
        v_id = track.get("visitor_id")
        emb = track.get("embedding")
        is_staff = track.get("is_staff", False)
        
        resolved_vid = None
        resolved_is_staff = is_staff
        
        # 1. If visitor_id is provided and exists globally, use it
        if v_id and v_id in global_reid_sessions:
            resolved_vid = v_id
            global_reid_sessions[v_id]["last_seen"] = current_time
            if is_staff:
                global_reid_sessions[v_id]["is_staff"] = True
            resolved_is_staff = global_reid_sessions[v_id]["is_staff"]
            
        # 2. Otherwise, check for match in global sessions
        if not resolved_vid and emb:
            best_match_vid = None
            best_sim = -1.0
            emb_np = np.array(emb)
            
            for vid, session_data in global_reid_sessions.items():
                s_emb = np.array(session_data["embedding"])
                # Compute cosine similarity
                dot_prod = np.dot(emb_np, s_emb)
                norm_emb = np.linalg.norm(emb_np)
                norm_s = np.linalg.norm(s_emb)
                if norm_emb > 0 and norm_s > 0:
                    sim = float(dot_prod / (norm_emb * norm_s))
                else:
                    sim = 0.0
                    
                if sim > best_sim:
                    best_sim = sim
                    best_match_vid = vid
                    
            if best_match_vid and best_sim > 0.82:
                resolved_vid = best_match_vid
                global_reid_sessions[resolved_vid]["last_seen"] = current_time
                if is_staff:
                    global_reid_sessions[resolved_vid]["is_staff"] = True
                resolved_is_staff = global_reid_sessions[resolved_vid]["is_staff"]
                
        # 3. If still not resolved, register as a new global session
        if not resolved_vid:
            resolved_vid = v_id or f"VIS_{str(uuid.uuid4())[:8]}"
            global_reid_sessions[resolved_vid] = {
                "visitor_id": resolved_vid,
                "embedding": emb,
                "is_staff": is_staff,
                "last_seen": current_time
            }
            resolved_is_staff = is_staff
            
        synced_tracks.append({
            "track_id": t_id,
            "visitor_id": resolved_vid,
            "is_staff": resolved_is_staff
        })
        
    return {"synced_tracks": synced_tracks}

# Video streaming endpoints
@app.post("/stores/{store_id}/cameras/{camera_id}/frame")
async def upload_frame(store_id: str, camera_id: str, request: Request):
    """Pipeline camera nodes post JPEG frames here for real-time visualization."""
    frame_bytes = await request.body()
    latest_frames[f"{store_id}_{camera_id}"] = frame_bytes
    latest_frame_times[f"{store_id}_{camera_id}"] = time.time()
    return {"status": "success"}

@app.get("/stores/{store_id}/cameras/{camera_id}/video_feed")
async def video_feed(store_id: str, camera_id: str):
    """Returns the latest JPEG frame for this camera.
    
    The dashboard polls this endpoint every ~80ms with a cache-busting query
    parameter and draws each frame onto a <canvas> element.  This is far more
    reliable than MJPEG multipart streaming in modern Chromium browsers.
    """
    from fastapi.responses import Response
    frame = latest_frames.get(f"{store_id}_{camera_id}")
    if not frame:
        # Return a 1x1 black JPEG placeholder so the client knows the camera
        # is not yet sending frames (avoids broken-image icons)
        import base64
        # Minimal 1×1 black JPEG (valid, 631 bytes)
        placeholder = bytes([
            0xFF,0xD8,0xFF,0xE0,0x00,0x10,0x4A,0x46,0x49,0x46,0x00,0x01,
            0x01,0x00,0x00,0x01,0x00,0x01,0x00,0x00,0xFF,0xDB,0x00,0x43,
            0x00,0x08,0x06,0x06,0x07,0x06,0x05,0x08,0x07,0x07,0x07,0x09,
            0x09,0x08,0x0A,0x0C,0x14,0x0D,0x0C,0x0B,0x0B,0x0C,0x19,0x12,
            0x13,0x0F,0x14,0x1D,0x1A,0x1F,0x1E,0x1D,0x1A,0x1C,0x1C,0x20,
            0x24,0x2E,0x27,0x20,0x22,0x2C,0x23,0x1C,0x1C,0x28,0x37,0x29,
            0x2C,0x30,0x31,0x34,0x34,0x34,0x1F,0x27,0x39,0x3D,0x38,0x32,
            0x3C,0x2E,0x33,0x34,0x32,0xFF,0xC0,0x00,0x0B,0x08,0x00,0x01,
            0x00,0x01,0x01,0x01,0x11,0x00,0xFF,0xC4,0x00,0x1F,0x00,0x00,
            0x01,0x05,0x01,0x01,0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
            0x09,0x0A,0x0B,0xFF,0xC4,0x00,0xB5,0x10,0x00,0x02,0x01,0x03,
            0x03,0x02,0x04,0x03,0x05,0x05,0x04,0x04,0x00,0x00,0x01,0x7D,
            0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,
            0x13,0x51,0x61,0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xA1,0x08,
            0x23,0x42,0xB1,0xC1,0x15,0x52,0xD1,0xF0,0x24,0x33,0x62,0x72,
            0x82,0x09,0x0A,0x16,0x17,0x18,0x19,0x1A,0x25,0x26,0x27,0x28,
            0x29,0x2A,0x34,0x35,0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,
            0x46,0x47,0x48,0x49,0x4A,0x53,0x54,0x55,0x56,0x57,0x58,0x59,
            0x5A,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x73,0x74,0x75,
            0x76,0x77,0x78,0x79,0x7A,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
            0x8A,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9A,0xA2,0xA3,0xA4,
            0xA5,0xA6,0xA7,0xA8,0xA9,0xAA,0xB2,0xB3,0xB4,0xB5,0xB6,0xB7,
            0xB8,0xB9,0xBA,0xC2,0xC3,0xC4,0xC5,0xC6,0xC7,0xC8,0xC9,0xCA,
            0xD2,0xD3,0xD4,0xD5,0xD6,0xD7,0xD8,0xD9,0xDA,0xE1,0xE2,0xE3,
            0xE4,0xE5,0xE6,0xE7,0xE8,0xE9,0xEA,0xF1,0xF2,0xF3,0xF4,0xF5,
            0xF6,0xF7,0xF8,0xF9,0xFA,0xFF,0xDA,0x00,0x08,0x01,0x01,0x00,
            0x00,0x3F,0x00,0xFB,0xD6,0xFF,0xD9
        ])
        return Response(content=placeholder, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})
    return Response(content=frame, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store, no-cache, must-revalidate"})

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
