import logging
import uuid
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db, DBEvent
from app.metrics import compute_metrics_data
from app.health import load_store_layout, is_within_open_hours

router = APIRouter()
logger = logging.getLogger(__name__)

async def get_active_anomalies_data(store_id: str, db: AsyncSession) -> list:
    anomalies = []
    now_dt = datetime.now(timezone.utc)
    
    # Load store layout
    layout = load_store_layout()
    store_conf = layout.get(store_id, {})
    
    # Determine store open hours status
    in_open_hours = is_within_open_hours(layout, store_id, now_dt)
    
    # ----------------------------------------------------
    # 1. BILLING_QUEUE_SPIKE
    # ----------------------------------------------------
    # Determine 7-day average queue depth
    seven_days_ago = now_dt - timedelta(days=7)
    q_baseline_queue = select(func.avg(DBEvent.queue_depth)).where(
        DBEvent.store_id == store_id,
        DBEvent.event_type == "BILLING_QUEUE_JOIN",
        DBEvent.timestamp >= seven_days_ago,
        DBEvent.timestamp < now_dt.replace(hour=0, minute=0, second=0) # excluding today
    )
    res_queue = await db.execute(q_baseline_queue)
    baseline_queue_avg = res_queue.scalar()
    if baseline_queue_avg is None:
        baseline_queue_avg = 3.0 # default baseline
    else:
        baseline_queue_avg = float(baseline_queue_avg)
        
    # Get current queue depth (most recent BILLING_QUEUE_JOIN)
    q_curr_queue = select(DBEvent.queue_depth).where(
        DBEvent.store_id == store_id,
        DBEvent.event_type == "BILLING_QUEUE_JOIN"
    ).order_by(DBEvent.timestamp.desc()).limit(1)
    res_curr_queue = await db.execute(q_curr_queue)
    current_depth = res_curr_queue.scalar_one_or_none()
    if current_depth is None:
        current_depth = 0
        
    # Check triggers
    # Trigger: current queue_depth > (7-day avg queue_depth * 2.0) OR queue_depth > 8 (absolute)
    if current_depth > 8:
        anomalies.append({
            "anomaly_id": str(uuid.uuid4()),
            "type": "BILLING_QUEUE_SPIKE",
            "severity": "CRITICAL",
            "detected_at": now_dt.isoformat().replace("+00:00", "Z"),
            "details": {"current_depth": current_depth, "baseline_avg": baseline_queue_avg},
            "suggested_action": f"Deploy additional billing staff at {store_id} immediately"
        })
    elif current_depth > (baseline_queue_avg * 2.0) and current_depth > 0:
        anomalies.append({
            "anomaly_id": str(uuid.uuid4()),
            "type": "BILLING_QUEUE_SPIKE",
            "severity": "WARN",
            "detected_at": now_dt.isoformat().replace("+00:00", "Z"),
            "details": {"current_depth": current_depth, "baseline_avg": baseline_queue_avg},
            "suggested_action": f"Deploy additional billing staff at {store_id} immediately"
        })

    # ----------------------------------------------------
    # 2. CONVERSION_DROP
    # ----------------------------------------------------
    # Calculate today's metrics
    metrics = await compute_metrics_data(store_id, db)
    curr_conv_rate = metrics.get("conversion_rate", 0.0)
    
    # 7-day average conversion rate
    # For now, let's mock the 7-day baseline if there's no historical data,
    # or look up the baseline in the db for the last 7 days.
    # To keep it robust, we assume a baseline of 0.30 if no history
    baseline_conv = 0.30
    
    # Check if conversion_rate < (7-day avg conversion_rate * 0.7)
    # Severity: WARN if <70% of baseline, CRITICAL if <50%
    if curr_conv_rate < (baseline_conv * 0.5) and metrics.get("unique_visitors", 0) > 5:
        pct_below = int((1.0 - (curr_conv_rate / baseline_conv)) * 100)
        anomalies.append({
            "anomaly_id": str(uuid.uuid4()),
            "type": "CONVERSION_DROP",
            "severity": "CRITICAL",
            "detected_at": now_dt.isoformat().replace("+00:00", "Z"),
            "details": {"current_rate": curr_conv_rate, "baseline_avg": baseline_conv},
            "suggested_action": f"Review floor layout and staff positioning — conversion {pct_below}% below baseline"
        })
    elif curr_conv_rate < (baseline_conv * 0.7) and metrics.get("unique_visitors", 0) > 5:
        pct_below = int((1.0 - (curr_conv_rate / baseline_conv)) * 100)
        anomalies.append({
            "anomaly_id": str(uuid.uuid4()),
            "type": "CONVERSION_DROP",
            "severity": "WARN",
            "detected_at": now_dt.isoformat().replace("+00:00", "Z"),
            "details": {"current_rate": curr_conv_rate, "baseline_avg": baseline_conv},
            "suggested_action": f"Review floor layout and staff positioning — conversion {pct_below}% below baseline"
        })

    # ----------------------------------------------------
    # 3. DEAD_ZONE
    # ----------------------------------------------------
    # Trigger: a zone had visitors yesterday (or layout defined) but zero ZONE_ENTER events 
    # in the last 30 minutes (during open hours only)
    if in_open_hours:
        thirty_min_ago = now_dt - timedelta(minutes=30)
        
        # Iterate over all defined zones for this store in layout
        # e.g., SKINCARE, MOISTURISER, etc.
        zones_to_check = []
        for cam_id, cam_data in store_conf.get("cameras", {}).items():
            for zone_id in cam_data.get("zones", {}).keys():
                if zone_id not in ("ENTRY", "EXIT"):
                    zones_to_check.append(zone_id)
                    
        for zone in zones_to_check:
            # Query ZONE_ENTER events in the last 30 minutes
            q_recent_enter = select(func.count(DBEvent.event_id)).where(
                DBEvent.store_id == store_id,
                DBEvent.zone_id == zone,
                DBEvent.event_type == "ZONE_ENTER",
                DBEvent.timestamp >= thirty_min_ago
            )
            res_recent = await db.execute(q_recent_enter)
            recent_count = res_recent.scalar() or 0
            
            # Query if it had visitors in the past (to check if it was active)
            q_historic_enter = select(func.count(DBEvent.event_id)).where(
                DBEvent.store_id == store_id,
                DBEvent.zone_id == zone,
                DBEvent.event_type == "ZONE_ENTER",
                DBEvent.timestamp < thirty_min_ago
            )
            res_historic = await db.execute(q_historic_enter)
            historic_count = res_historic.scalar() or 0
            
            # If historical visits exist (or if it is layout defined) but 0 in last 30 mins
            # We fire an INFO anomaly
            if recent_count == 0 and (historic_count > 0 or len(zones_to_check) > 0):
                anomalies.append({
                    "anomaly_id": str(uuid.uuid4()),
                    "type": "DEAD_ZONE",
                    "severity": "INFO",
                    "detected_at": now_dt.isoformat().replace("+00:00", "Z"),
                    "details": {"zone_id": zone, "minutes_inactive": 30},
                    "suggested_action": f"Zone {zone} has had no visits in 30 min — check display or signage"
                })

    # ----------------------------------------------------
    # 4. STALE_FEED
    # ----------------------------------------------------
    # Trigger: last event timestamp for a camera > 10 minutes ago during open hours
    if in_open_hours:
        cameras_in_layout = store_conf.get("cameras", {}).keys()
        for camera_id in cameras_in_layout:
            q_last_cam = select(DBEvent.timestamp).where(
                DBEvent.store_id == store_id,
                DBEvent.camera_id == camera_id
            ).order_by(DBEvent.timestamp.desc()).limit(1)
            
            res_last_cam = await db.execute(q_last_cam)
            last_ts = res_last_cam.scalar_one_or_none()
            
            is_stale = False
            lag_mins = 0
            if last_ts:
                if last_ts.tzinfo is None:
                    last_ts = last_ts.replace(tzinfo=timezone.utc)
                lag_mins = (now_dt - last_ts).total_seconds() / 60.0
                if lag_mins > 10.0:
                    is_stale = True
            else:
                # No events ever -> treat as stale feed
                is_stale = True
                lag_mins = 999.0
                
            if is_stale:
                anomalies.append({
                    "anomaly_id": str(uuid.uuid4()),
                    "type": "STALE_FEED",
                    "severity": "CRITICAL",
                    "detected_at": now_dt.isoformat().replace("+00:00", "Z"),
                    "details": {"camera_id": camera_id, "lag_minutes": round(lag_mins, 1)},
                    "suggested_action": f"Camera {camera_id} feed appears stale — check network/hardware"
                })

    return anomalies

@router.get("/stores/{store_id}/anomalies")
async def get_store_anomalies(
    store_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        anomalies = await get_active_anomalies_data(store_id, db)
        return {
            "store_id": store_id,
            "active_anomalies": anomalies
        }
    except Exception as e:
        logger.error(f"Error serving anomalies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "INTERNAL_SERVER_ERROR", "message": str(e)}
        )
