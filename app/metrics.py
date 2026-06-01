import logging
from datetime import datetime, time, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db, DBEvent, DBPosTransaction

router = APIRouter()
logger = logging.getLogger(__name__)

async def compute_metrics_data(store_id: str, db: AsyncSession) -> dict:
    # 1. Determine "today" based on the latest event timestamp for this store
    q_latest = select(func.max(DBEvent.timestamp)).where(DBEvent.store_id == store_id)
    res_latest = await db.execute(q_latest)
    latest_ts = res_latest.scalar_one_or_none()
    
    if latest_ts:
        today_date = latest_ts.date()
    else:
        # Fallback to current calendar date if no events
        today_date = datetime.utcnow().date()
        
    start_of_day = datetime.combine(today_date, time.min)
    end_of_day = datetime.combine(today_date, time.max)
    
    # 2. Fetch all events for this store today
    q_events = select(DBEvent).where(
        DBEvent.store_id == store_id,
        DBEvent.timestamp >= start_of_day,
        DBEvent.timestamp <= end_of_day
    )
    res_events = await db.execute(q_events)
    events = res_events.scalars().all()
    
    if not events:
        # Handle zero-traffic stores: return all metrics as 0/null, never crash
        return {
            "store_id": store_id,
            "date": today_date.isoformat(),
            "unique_visitors": 0,
            "conversion_rate": 0.0,
            "avg_dwell_per_zone": {},
            "queue_depth": 0,
            "abandonment_rate": 0.0,
            "data_quality_score": None,
            "low_confidence_event_pct": 0.0,
            "computed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        }

    # Filter out staff events for customer metrics
    customer_events = [e for e in events if not e.is_staff]
    
    # - unique_visitors: count distinct visitor_ids that have an ENTRY event today
    unique_visitors_set = {e.visitor_id for e in customer_events if e.event_type == "ENTRY"}
    unique_visitors = len(unique_visitors_set)
    
    # - conversion_rate: visitors who completed a purchase (matched_visitor in pos_transactions)
    #   divided by total unique visitors.
    # NOTE: We intentionally do NOT filter POS transactions by date — the 5-minute
    # correlation window in pos.py already ensures temporal proximity. Filtering by
    # today_date would silently exclude seed/demo data and always return 0.
    q_tx = select(DBPosTransaction.matched_visitor).where(
        DBPosTransaction.store_id == store_id,
        DBPosTransaction.matched_visitor.isnot(None)
    )
    res_tx = await db.execute(q_tx)
    matched_visitors_today = set(res_tx.scalars().all())
    
    # Only count matches if the visitor actually had an ENTRY event today
    purchasing_visitors = matched_visitors_today.intersection(unique_visitors_set)
    
    if unique_visitors > 0:
        conversion_rate = round(len(purchasing_visitors) / unique_visitors, 4)
    else:
        conversion_rate = 0.0
        
    # - avg_dwell_per_zone: average dwell_ms per zone_id from ZONE_DWELL events
    zone_dwells = {}
    for e in customer_events:
        if e.event_type == "ZONE_DWELL" and e.zone_id:
            if e.zone_id not in zone_dwells:
                zone_dwells[e.zone_id] = []
            zone_dwells[e.zone_id].append(e.dwell_ms)
            
    avg_dwell_per_zone = {}
    for zone, dwells in zone_dwells.items():
        avg_dwell_per_zone[zone] = int(sum(dwells) / len(dwells)) if dwells else 0
        
    # - queue_depth: latest queue_depth value from the most recent BILLING_QUEUE_JOIN event
    queue_joins = [e for e in customer_events if e.event_type == "BILLING_QUEUE_JOIN"]
    if queue_joins:
        # Sort by timestamp
        queue_joins.sort(key=lambda x: x.timestamp)
        queue_depth = queue_joins[-1].queue_depth or 0
    else:
        queue_depth = 0
        
    # - abandonment_rate: BILLING_QUEUE_ABANDON events ÷ BILLING_QUEUE_JOIN events
    abandons_count = sum(1 for e in customer_events if e.event_type == "BILLING_QUEUE_ABANDON")
    joins_count = len(queue_joins)
    
    if joins_count > 0:
        abandonment_rate = round(abandons_count / joins_count, 4)
    else:
        abandonment_rate = 0.0
        
    # - data_quality_score: 1 - (low_confidence_event_count / total_event_count)
    #   where low_confidence = confidence < 0.5
    total_events_count = len(events)
    low_confidence_count = sum(1 for e in events if e.confidence < 0.5)
    
    if total_events_count > 0:
        low_confidence_event_pct = round(low_confidence_count / total_events_count, 4)
        data_quality_score = round(1.0 - low_confidence_event_pct, 4)
    else:
        low_confidence_event_pct = 0.0
        data_quality_score = None
        
    return {
        "store_id": store_id,
        "date": today_date.isoformat(),
        "unique_visitors": unique_visitors,
        "conversion_rate": conversion_rate,
        "avg_dwell_per_zone": avg_dwell_per_zone,
        "queue_depth": queue_depth,
        "abandonment_rate": abandonment_rate,
        "data_quality_score": data_quality_score,
        "low_confidence_event_pct": low_confidence_event_pct,
        "computed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    }

@router.get("/stores/{store_id}/metrics")
async def get_store_metrics(
    store_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        metrics = await compute_metrics_data(store_id, db)
        return metrics
    except Exception as e:
        logger.error(f"Error serving metrics: {e}")
        # Wrap all DB errors in structured responses
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "INTERNAL_SERVER_ERROR", "message": str(e)}
        )
