import os
import json
import logging
import asyncio
from redis.asyncio import Redis, from_url
from app.models import RetailEvent

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
STREAM_KEY = "retail:events"
API_GROUP = "api-ingest"
DASHBOARD_GROUP = "dashboard"

logger = logging.getLogger(__name__)

# Mock Redis Streams Client for local environments without running Redis
class MockRedis:
    def __init__(self, *args, **kwargs):
        self.streams = {}

    async def ping(self):
        return True

    async def xgroup_create(self, stream, group, id="0", mkstream=False):
        return True

    async def xadd(self, stream, data):
        import time
        entry_id = f"{int(time.time()*1000)}-0"
        if stream not in self.streams:
            self.streams[stream] = []
        self.streams[stream].append((entry_id, data))
        return entry_id

    async def xreadgroup(self, group, consumer, streams, count=100, block=None):
        if block:
            await asyncio.sleep(block / 1000.0)
        res = []
        for stream, last_id in streams.items():
            msgs = self.streams.get(stream, [])
            res.append((stream, msgs))
            self.streams[stream] = []
        return res

    async def xack(self, stream, group, entry_id):
        return True

    async def close(self):
        pass

mock_redis_instance = MockRedis()

_redis_client = None

async def get_redis() -> Redis:
    global _redis_client
    if _redis_client is not None:
        return _redis_client
        
    client = from_url(REDIS_URL, decode_responses=True)
    try:
        # Quick ping to verify connectivity
        await asyncio.wait_for(client.ping(), timeout=1.0)
        _redis_client = client
        return client
    except Exception:
        logger.warning(f"Redis connection to {REDIS_URL} failed. Falling back to in-memory MockRedis permanently.")
        await client.close()
        _redis_client = mock_redis_instance
        return mock_redis_instance

async def init_redis(redis: Redis):
    """Initialize stream and consumer groups if they don't exist."""
    for group in [API_GROUP, DASHBOARD_GROUP]:
        try:
            # Create stream and consumer group.
            await redis.xgroup_create(STREAM_KEY, group, id="0", mkstream=True)
            logger.info(f"Created consumer group {group} on stream {STREAM_KEY}")
        except Exception as e:
            # If group already exists, it raises BusyGroupError
            if "BUSYGROUP" in str(e):
                logger.debug(f"Consumer group {group} already exists")
            else:
                logger.error(f"Error creating consumer group {group}: {e}")

async def emit_event(redis: Redis, event: RetailEvent) -> str:
    """Append event to stream. Returns stream entry ID."""
    event_dict = json.loads(event.model_dump_json())
    data = {"event": json.dumps(event_dict)}
    return await redis.xadd(STREAM_KEY, data)

async def consume_events(redis: Redis, group: str, consumer: str, count: int = 100):
    """
    Blocking read from consumer group. Returns list of (entry_id, event_dict).
    """
    try:
        res = await redis.xreadgroup(group, consumer, {STREAM_KEY: ">"}, count=count, block=1000)
        events = []
        if res:
            for stream_name, messages in res:
                for entry_id, message_data in messages:
                    if "event" in message_data:
                        event_dict = json.loads(message_data["event"])
                        events.append((entry_id, event_dict))
        return events
    except Exception as e:
        logger.error(f"Error consuming events: {e}")
        return []

async def ack_event(redis: Redis, group: str, entry_id: str):
    """Acknowledge processed event to prevent redelivery."""
    await redis.xack(STREAM_KEY, group, entry_id)
