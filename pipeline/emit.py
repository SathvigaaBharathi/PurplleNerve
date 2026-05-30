import json
import logging
import os
from datetime import datetime
from redis.asyncio import Redis, from_url

# Import models from app
from app.models import RetailEvent, EventMetadata

logger = logging.getLogger(__name__)

class EventEmitter:
    """Handles schema validation and event serialization to JSONL, Redis Streams, and direct HTTP APIs."""
    def __init__(self, output_jsonl_path: str = None, redis_url: str = None, api_url: str = None):
        self.output_jsonl_path = output_jsonl_path
        self.redis_url = redis_url
        self.api_url = api_url
        self.redis_client = None
        # Once Redis is confirmed unreachable, skip all future reconnect attempts
        # so a missing Redis doesn't block/crash the HTTP-only pipeline path.
        self._redis_unavailable = False

    async def connect_redis(self):
        """Initialise Redis client if redis_url is provided.

        Sets _redis_unavailable=True on first failure so subsequent emit() calls
        never retry the blocking DNS/connect round-trip.
        """
        if self._redis_unavailable:
            return
        if self.redis_url and not self.redis_client:
            try:
                self.redis_client = from_url(self.redis_url, decode_responses=True)
                await self.redis_client.ping()
                logger.info(f"EventEmitter connected to Redis at {self.redis_url}")
            except Exception as e:
                logger.error(f"EventEmitter failed to connect to Redis: {e}")
                self.redis_client = None
                self._redis_unavailable = True  # stop retrying

    async def emit(
        self,
        store_id: str,
        camera_id: str,
        visitor_id: str,
        event_type: str,
        timestamp: datetime,
        zone_id: str = None,
        dwell_ms: int = 0,
        is_staff: bool = False,
        confidence: float = 1.0,
        queue_depth: int = None,
        sku_zone: str = None,
        session_seq: int = 0
    ) -> RetailEvent | None:
        """Constructs a RetailEvent, writes to JSONL, emits to Redis, and posts to HTTP API."""
        # 1. Build and validate event
        try:
            # Build metadata
            metadata = EventMetadata(
                queue_depth=queue_depth,
                sku_zone=sku_zone,
                session_seq=session_seq
            )
            
            # Build retail event
            event = RetailEvent(
                store_id=store_id,
                camera_id=camera_id,
                visitor_id=visitor_id,
                event_type=event_type,
                timestamp=timestamp,
                zone_id=zone_id,
                dwell_ms=dwell_ms,
                is_staff=is_staff,
                confidence=confidence,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Event schema validation failed: {e}")
            return None

        event_json = event.model_dump_json()
        event_dict = json.loads(event_json)

        # 2. Write to JSONL file
        if self.output_jsonl_path:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(os.path.abspath(self.output_jsonl_path)), exist_ok=True)
                with open(self.output_jsonl_path, "a") as f:
                    f.write(event_json + "\n")
            except Exception as e:
                logger.error(f"Failed to write event to JSONL file: {e}")

        # 3. Emit to Redis Stream (skipped silently if Redis is unavailable)
        if self.redis_url and not self._redis_unavailable:
            await self.connect_redis()
            if self.redis_client:
                try:
                    data = {"event": json.dumps(event_dict)}
                    await self.redis_client.xadd("retail:events", data)
                except Exception as e:
                    logger.error(f"Failed to add event to Redis stream: {e}")
                    self._redis_unavailable = True

        # 4. Post to HTTP API URL if provided
        if self.api_url:
            try:
                import httpx
                # HTTP POST to ingest endpoint expects a list of events
                async with httpx.AsyncClient() as client:
                    await client.post(f"{self.api_url}/events/ingest", json=[event_dict])
            except Exception as e:
                logger.error(f"EventEmitter failed to POST event to API: {e}")

        return event

    async def close(self):
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
