import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db, DBEvent

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/stores/{store_id}/heatmap")
async def get_store_heatmap(
    store_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        # 1. Query metrics from materialized view
        # Columns in view: store_id, zone_id, visitor_count, avg_dwell_ms, last_seen
        # Use raw text or select. Since it's a materialized view we can query it using standard select
        query = text("""
            SELECT zone_id, visitor_count, avg_dwell_ms, last_seen 
            FROM zone_dwell_agg 
            WHERE store_id = :store_id
        """)
        res = await db.execute(query, {"store_id": store_id})
        rows = res.all()

        # 2. Check total unique visitor sessions to compute data_confidence flag (threshold >= 20)
        # Sessions are unique non-staff visitors who have an ENTRY today
        # Find latest event date first
        q_latest = select(func.max(DBEvent.timestamp)).where(DBEvent.store_id == store_id)
        res_latest = await db.execute(q_latest)
        latest_ts = res_latest.scalar_one_or_none()
        
        if latest_ts:
            today_date = latest_ts.date()
        else:
            today_date = datetime.utcnow().date()
            
        import datetime as dt_module
        start_of_day = datetime.combine(today_date, datetime.min.time())
        end_of_day = datetime.combine(today_date, datetime.max.time())
        
        q_sessions = select(func.count(func.distinct(DBEvent.visitor_id))).where(
            DBEvent.store_id == store_id,
            DBEvent.event_type == "ENTRY",
            DBEvent.is_staff == False,
            DBEvent.timestamp >= start_of_day,
            DBEvent.timestamp <= end_of_day
        )
        res_sessions = await db.execute(q_sessions)
        session_count = res_sessions.scalar_one() or 0
        data_confidence = (session_count >= 20)

        # 3. Process records and normalize
        zones_data = []
        max_visitors = 0
        max_dwell = 0.0
        
        for r in rows:
            zone_id, visitor_count, avg_dwell_ms, last_seen = r
            visitor_count = int(visitor_count or 0)
            avg_dwell_ms = float(avg_dwell_ms or 0.0)
            
            if visitor_count > max_visitors:
                max_visitors = visitor_count
            if avg_dwell_ms > max_dwell:
                max_dwell = avg_dwell_ms
                
            zones_data.append({
                "zone_id": zone_id,
                "visitor_count": visitor_count,
                "avg_dwell_ms": avg_dwell_ms,
                "last_seen": last_seen.isoformat() if last_seen else None
            })

        # Normalize score between 0 and 100
        for zone in zones_data:
            vc = zone["visitor_count"]
            ad = zone["avg_dwell_ms"]
            
            norm_frequency = round((vc / max_visitors) * 100, 1) if max_visitors > 0 else 0.0
            norm_dwell = round((ad / max_dwell) * 100, 1) if max_dwell > 0.0 else 0.0
            
            # Combined heatmap score
            zone["normalized_frequency"] = norm_frequency
            zone["normalized_dwell"] = norm_dwell
            zone["normalized_score"] = round((norm_frequency + norm_dwell) / 2.0, 1)

        return {
            "store_id": store_id,
            "data_confidence": data_confidence,
            "session_count": session_count,
            "zones": zones_data,
            "computed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        }
    except Exception as e:
        logger.error(f"Error serving heatmap: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "INTERNAL_SERVER_ERROR", "message": str(e)}
        )
