import time
import json
import os
import logging
from datetime import datetime, timezone, time as datetime_time
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy import text, select, func
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from app.db import get_db, DBEvent
from app.redis_client import get_redis

router = APIRouter()
logger = logging.getLogger(__name__)

# Global start time for uptime calculation
START_TIME = time.time()

ALL_KNOWN_CAMERAS = {}

def ensure_all_known_cameras():
    if not ALL_KNOWN_CAMERAS:
        try:
            layout = load_store_layout()
            for store_id, store_data in layout.items():
                if isinstance(store_data, dict) and "cameras" in store_data:
                    cameras_dict = store_data.get("cameras", {})
                    if isinstance(cameras_dict, dict):
                        ALL_KNOWN_CAMERAS[store_id] = list(cameras_dict.keys())
        except Exception as e:
            logger.error(f"Error ensuring ALL_KNOWN_CAMERAS: {e}")

async def get_active_cameras(store_id: str, db: AsyncSession) -> list[str]:
    """
    Returns list of camera_ids that have sent at least one event
    in the last 10 minutes for this store.
    Uses the same STALE_FEED threshold as GET /health.
    Never hardcodes camera IDs — always derived from events table.
    """
    from datetime import datetime, timezone, timedelta
    from sqlalchemy import select, distinct
    from app.db import DBEvent
    
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)
    result = await db.execute(
        select(distinct(DBEvent.camera_id))
        .where(DBEvent.store_id == store_id)
        .where(DBEvent.is_staff == False)
        .where(DBEvent.ingested_at >= cutoff)
        .order_by(DBEvent.camera_id)
    )
    return [row[0] for row in result.fetchall()]

def load_store_layout():
    paths = [
        "data/store_layout.json",
        "/app/data/store_layout.json",
        "../data/store_layout.json",
        "store-intelligence/data/store_layout.json",
        "D:/purplle/store-intelligence/data/store_layout.json"
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading layout at {p}: {e}")
    # Return a basic default config if file is missing (to prevent crashes during test runs)
    return {
        "STORE_BLR_002": {
            "store_name": "Apex Retail Bengaluru",
            "open_hours": {"start": "10:00", "end": "22:00"},
            "staff_uniform_hue_range": [95, 115]
        }
    }

def is_within_open_hours(store_layout, store_id, dt: datetime) -> bool:
    """Check if the given datetime is within store open hours."""
    store_conf = store_layout.get(store_id)
    if not store_conf or "open_hours" not in store_conf:
        return True
    
    open_h = store_conf["open_hours"]
    start_str = open_h.get("start", "10:00")
    end_str = open_h.get("end", "22:00")
    
    try:
        sh, sm = map(int, start_str.split(":"))
        eh, em = map(int, end_str.split(":"))
        
        # Local time of the store (ignoring timezone offset for time of day comparison)
        t = dt.time()
        start_time = datetime_time(sh, sm)
        end_time = datetime_time(eh, em)
        
        return start_time <= t <= end_time
    except Exception as e:
        logger.error(f"Error checking open hours: {e}")
        return True

@router.get("/health")
async def health_check(
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis)
):
    uptime = int(time.time() - START_TIME)
    
    # 1. Database Health Check
    db_status = "connected"
    try:
        await db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = "disconnected"
        logger.error(f"Database health check failed: {e}")
        # Database unavailable -> status: unhealthy, HTTP 503, no stack trace
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "SERVICE_UNAVAILABLE",
                "message": "Database temporarily unavailable",
                "trace_id": "health-check-error"
            }
        )

    # 2. Redis Health Check
    redis_status = "connected"
    try:
        await redis.ping()
    except Exception as e:
        redis_status = "disconnected"
        logger.error(f"Redis health check failed: {e}")
        # Redis unavailable -> status: degraded, HTTP 200 (API still serves cached metrics)
        
    overall_status = "healthy"
    if redis_status == "disconnected":
        overall_status = "degraded"

    # 3. Store Feed Status
    stores_health = {}
    try:
        layout = load_store_layout()
        # Query database for recent events grouped by store
        # Get last event time and count for each store
        # Wait, if database is connected, query events table
        now_dt = datetime.now(timezone.utc)
        
        for store_id in layout.keys():
            # Query last event
            q_last = select(DBEvent.timestamp).where(
                DBEvent.store_id == store_id
            ).order_by(DBEvent.timestamp.desc()).limit(1)
            
            res_last = await db.execute(q_last)
            last_ts = res_last.scalar_one_or_none()
            
            # Query count last hour
            # Let's say last hour from now
            one_hour_ago = now_dt - select(text("INTERVAL '1 hour'"))
            # Alternatively, compute time in Python
            import datetime as dt_module
            one_hour_ago = now_dt - dt_module.timedelta(hours=1)
            
            q_count = select(func.count(DBEvent.event_id)).where(
                DBEvent.store_id == store_id,
                DBEvent.timestamp >= one_hour_ago
            )
            res_count = await db.execute(q_count)
            count_last_hour = res_count.scalar_one() or 0
            
            ensure_all_known_cameras()
            active_cameras = await get_active_cameras(store_id, db)
            stale_cameras = [
                cam for cam in ALL_KNOWN_CAMERAS.get(store_id, [])
                if cam not in active_cameras
            ]
            
            in_open_hours = is_within_open_hours(layout, store_id, now_dt)
            
            feed_status = "LIVE"
            if in_open_hours and stale_cameras:
                feed_status = "STALE"
            elif not ALL_KNOWN_CAMERAS.get(store_id, []):
                # Fallback if layout not loaded / no cameras mapped
                if last_ts:
                    if last_ts.tzinfo is None:
                        last_ts = last_ts.replace(tzinfo=timezone.utc)
                    lag_minutes = (now_dt - last_ts).total_seconds() / 60.0
                    if lag_minutes > 10.0 and in_open_hours:
                        feed_status = "STALE"
                else:
                    feed_status = "STALE"
            
            last_event_str = last_ts.isoformat() if last_ts else None
                
            stores_health[store_id] = {
                "last_event_at": last_event_str,
                "feed_status": feed_status,
                "events_last_hour": count_last_hour,
                "active_cameras": active_cameras,
                "stale_cameras": stale_cameras
            }
    except Exception as e:
        logger.error(f"Error calculating store feeds health: {e}")
        # Don't fail the whole health check if querying tables has an edge-case error
        pass

    return {
        "status": overall_status,
        "database": db_status,
        "redis": redis_status,
        "stores": stores_health,
        "uptime_seconds": uptime,
        "version": "1.0.0"
    }
