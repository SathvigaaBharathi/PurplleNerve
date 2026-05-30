import logging
from datetime import datetime, time, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db, DBEvent, DBPosTransaction

router = APIRouter()
logger = logging.getLogger(__name__)

async def compute_funnel_data(store_id: str, db: AsyncSession) -> dict:
    # 1. Determine "today" based on the latest event timestamp for this store
    q_latest = select(func.max(DBEvent.timestamp)).where(DBEvent.store_id == store_id)
    res_latest = await db.execute(q_latest)
    latest_ts = res_latest.scalar_one_or_none()
    
    if latest_ts:
        today_date = latest_ts.date()
    else:
        today_date = datetime.utcnow().date()
        
    start_of_day = datetime.combine(today_date, time.min)
    end_of_day = datetime.combine(today_date, time.max)
    
    # 2. Fetch all non-staff events for this store today
    q_events = select(DBEvent).where(
        DBEvent.store_id == store_id,
        DBEvent.timestamp >= start_of_day,
        DBEvent.timestamp <= end_of_day,
        DBEvent.is_staff == False
    )
    res_events = await db.execute(q_events)
    events = res_events.scalars().all()
    
    # Initialize visitor sets for each stage
    entry_visitors = set()
    zone_visit_visitors = set()
    billing_queue_visitors = set()
    purchase_visitors = set()
    
    if events:
        # Stage 1: "Entry" - count of unique visitor_ids with any ENTRY today
        entry_visitors = {e.visitor_id for e in events if e.event_type == "ENTRY"}
        
        # Stage 2: "Zone visit" - visitors who have >= 1 ZONE_ENTER event in any non-billing zone today
        # Non-billing zones are zones that are not billing (BILLING, BILLING_COUNTER, CASHIER)
        non_billing_zones = {"BILLING", "BILLING_COUNTER", "CASHIER"}
        zone_visit_visitors = {
            e.visitor_id for e in events 
            if e.event_type == "ZONE_ENTER" and e.zone_id and e.zone_id.upper() not in non_billing_zones
        }
        # In a real funnel, we might expect Stage 2 visitors to be a subset of Stage 1, 
        # but to make it a clean funnel, let's intersect it with Entry visitors 
        # or let's keep it as is. Intersecting with entry_visitors ensures they had an entry.
        zone_visit_visitors = zone_visit_visitors.intersection(entry_visitors)
        
        # Stage 3: "Billing queue" - visitors who have >= 1 BILLING_QUEUE_JOIN event today
        billing_queue_visitors = {
            e.visitor_id for e in events 
            if e.event_type == "BILLING_QUEUE_JOIN"
        }.intersection(entry_visitors)
        
        # Stage 4: "Purchase" - visitors correlated to a POS transaction today
        q_tx = select(DBPosTransaction.matched_visitor).where(
            DBPosTransaction.store_id == store_id,
            DBPosTransaction.timestamp >= start_of_day,
            DBPosTransaction.timestamp <= end_of_day,
            DBPosTransaction.matched_visitor.isnot(None)
        )
        res_tx = await db.execute(q_tx)
        matched_visitors_today = set(res_tx.scalars().all())
        purchase_visitors = matched_visitors_today.intersection(entry_visitors)

    # Convert sets to counts
    count_entry = len(entry_visitors)
    count_zone = len(zone_visit_visitors)
    count_billing = len(billing_queue_visitors)
    count_purchase = len(purchase_visitors)
    
    # Calculate drop-off percentages
    # Drop-off % = (stage_N-1 - stage_N) / stage_N-1 * 100
    dropoff_entry = 0.0
    
    if count_entry > 0:
        dropoff_zone = round(((count_entry - count_zone) / count_entry) * 100, 1)
    else:
        dropoff_zone = 0.0
        
    if count_zone > 0:
        dropoff_billing = round(((count_zone - count_billing) / count_zone) * 100, 1)
    else:
        dropoff_billing = 0.0
        
    if count_billing > 0:
        dropoff_purchase = round(((count_billing - count_purchase) / count_billing) * 100, 1)
    else:
        dropoff_purchase = 0.0
        
    return {
        "store_id": store_id,
        "window": "today",
        "funnel": [
            {"stage": "Entry",         "visitors": count_entry, "dropoff_pct": dropoff_entry},
            {"stage": "Zone visit",    "visitors": count_zone, "dropoff_pct": dropoff_zone},
            {"stage": "Billing queue", "visitors": count_billing, "dropoff_pct": dropoff_billing},
            {"stage": "Purchase",      "visitors": count_purchase, "dropoff_pct": dropoff_purchase}
        ]
    }

@router.get("/stores/{store_id}/funnel")
async def get_store_funnel(
    store_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        funnel = await compute_funnel_data(store_id, db)
        return funnel
    except Exception as e:
        logger.error(f"Error serving funnel: {e}")
        # Wrap database errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "INTERNAL_SERVER_ERROR", "message": str(e)}
        )
