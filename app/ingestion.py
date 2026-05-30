import time
import uuid
import json
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from pydantic import ValidationError
from redis.asyncio import Redis

from app.db import get_db, DBEvent
from app.models import RetailEvent
from app.redis_client import get_redis, emit_event

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/events/ingest")
async def ingest_events(
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis)
):
    start_time = time.time()
    
    # Extract trace_id from request state (attached by middleware)
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
    
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "accepted": 0,
                "rejected": 0,
                "errors": [{"event_id": "unknown", "reason": f"Invalid JSON payload: {str(e)}"}],
                "trace_id": trace_id
            }
        )

    # Resolve event list
    raw_events = []
    if isinstance(body, list):
        raw_events = body
    elif isinstance(body, dict):
        raw_events = body.get("events", [])
        if not isinstance(raw_events, list):
            raw_events = [raw_events]
    else:
        raw_events = [body]

    accepted = 0
    rejected = 0
    errors = []
    valid_db_objects = []
    valid_events_to_redis = []
    stores_touched = set()

    for idx, raw_ev in enumerate(raw_events):
        event_id = raw_ev.get("event_id", f"unknown-index-{idx}")
        try:
            # Validate individual event against schema
            event = RetailEvent.model_validate(raw_ev)
            
            # Prepare db object dictionary
            db_obj = {
                "event_id": event.event_id,
                "store_id": event.store_id,
                "camera_id": event.camera_id,
                "visitor_id": event.visitor_id,
                "event_type": event.event_type,
                "timestamp": event.timestamp,
                "zone_id": event.zone_id,
                "dwell_ms": event.dwell_ms,
                "is_staff": event.is_staff,
                "confidence": event.confidence,
                "queue_depth": event.metadata.queue_depth,
                "sku_zone": event.metadata.sku_zone,
                "session_seq": event.metadata.session_seq
            }
            
            valid_db_objects.append(db_obj)
            valid_events_to_redis.append(event)
            stores_touched.add(event.store_id)
            accepted += 1
        except ValidationError as val_err:
            rejected += 1
            # Combine Pydantic error details
            reasons = []
            for err in val_err.errors():
                loc = ".".join(map(str, err["loc"]))
                reasons.append(f"{loc}: {err['msg']}")
            errors.append({
                "event_id": str(event_id),
                "reason": "; ".join(reasons)
            })
        except Exception as e:
            rejected += 1
            errors.append({
                "event_id": str(event_id),
                "reason": str(e)
            })

    # Bulk Insert to DB using ON CONFLICT DO NOTHING for idempotency
    if valid_db_objects:
        try:
            stmt = insert(DBEvent).values(valid_db_objects)
            # Instruct Postgres to do nothing on event_id conflict (idempotency)
            stmt = stmt.on_conflict_do_nothing(index_elements=["event_id"])
            await db.execute(stmt)
            await db.commit()
            
            # Concurrently refresh materialized view if any ZONE_DWELL events were ingested
            if any(obj["event_type"] == "ZONE_DWELL" for obj in valid_db_objects):
                await db.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY zone_dwell_agg;"))
                await db.commit()
        except Exception as db_err:
            logger.error(f"Database insertion failed: {db_err}")
            # Wrap all DB errors in structured responses to avoid leaking details
            await db.rollback()
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "SERVICE_UNAVAILABLE",
                    "message": "Database write failure",
                    "trace_id": trace_id
                }
            )

        # Emit to Redis Streams
        for event in valid_events_to_redis:
            try:
                await emit_event(redis, event)
            except Exception as red_err:
                # Log redis emission errors but don't fail HTTP request (degraded mode)
                logger.error(f"Failed emitting event to Redis: {red_err}")

    latency_ms = int((time.time() - start_time) * 1000)
    
    # Store information inside request state for middleware logging
    request.state.store_ids = list(stores_touched)
    request.state.event_count = len(raw_events)
    request.state.latency_ms = latency_ms

    status_code = status.HTTP_200_OK if rejected == 0 else status.HTTP_207_MULTI_STATUS
    
    return JSONResponse(
        status_code=status_code,
        content={
            "accepted": accepted,
            "rejected": rejected,
            "errors": errors,
            "trace_id": trace_id
        }
    )
