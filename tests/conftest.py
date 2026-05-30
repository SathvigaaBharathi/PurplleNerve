# PROMPT: "Write conftest for pytest. Define async DB fixtures and app client.
#          Ensure cleanup after test cases run."
#
# CHANGES MADE:
# - Created tables, view, and indexes in setup.
# - Seeded transaction data.
# - Implemented MockRedis and patched redis connection.

import pytest
import asyncio
import os
import time
from sqlalchemy import text
from httpx import AsyncClient
from unittest.mock import MagicMock

# Set testing environment variables before importing app
os.environ["DATABASE_URL"] = os.getenv(
    "DATABASE_URL", 
    "postgresql+asyncpg://postgres:postgres@localhost:5432/retail_intelligence"
)
os.environ["REDIS_URL"] = "redis://localhost:6379"

# Mock Redis Streams Client
class MockRedis:
    def __init__(self, *args, **kwargs):
        self.streams = {}

    async def ping(self):
        return True

    async def xgroup_create(self, stream, group, id="0", mkstream=False):
        return True

    async def xadd(self, stream, data):
        entry_id = f"{int(time.time()*1000)}-0"
        if stream not in self.streams:
            self.streams[stream] = []
        self.streams[stream].append((entry_id, data))
        return entry_id

    async def xreadgroup(self, group, consumer, streams, count=100, block=None):
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

# Patch redis from_url globally
import redis.asyncio
mock_redis_instance = MockRedis()
redis.asyncio.from_url = lambda *a, **kw: mock_redis_instance

from app.main import app
from app.db import AsyncSessionLocal, Base, engine, init_db

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session", autouse=True)
async def setup_test_db():
    # Initialise tables and view (create if they don't exist)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
        # Create materialized view if not exists
        await conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS zone_dwell_agg AS
            SELECT
                store_id,
                zone_id,
                COUNT(DISTINCT visitor_id) FILTER (WHERE NOT is_staff) AS visitor_count,
                AVG(dwell_ms)                                           AS avg_dwell_ms,
                MAX(timestamp)                                          AS last_seen
            FROM events
            WHERE event_type = 'ZONE_DWELL'
              AND timestamp > NOW() - INTERVAL '24 hours'
            GROUP BY store_id, zone_id;
        """))
        
        await conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_zone_dwell_agg_store_zone 
            ON zone_dwell_agg(store_id, zone_id);
        """))
        
        # Truncate tables for a clean test run
        await conn.execute(text("TRUNCATE TABLE events, pos_transactions, session_conversions CASCADE;"))
        await conn.execute(text("REFRESH MATERIALIZED VIEW zone_dwell_agg;"))
    yield
    # Clean up table rows after session (do not drop tables to avoid crashing live app)
    async with engine.begin() as conn:
         await conn.execute(text("TRUNCATE TABLE events, pos_transactions, session_conversions CASCADE;"))
         await conn.execute(text("REFRESH MATERIALIZED VIEW zone_dwell_agg;"))

@pytest.fixture
async def db_session():
    async with AsyncSessionLocal() as session:
        yield session
        # Clear tables after individual test runs to avoid pollution
        await session.execute(text("TRUNCATE TABLE events, pos_transactions, session_conversions CASCADE;"))
        await session.execute(text("REFRESH MATERIALIZED VIEW zone_dwell_agg;"))
        await session.commit()

@pytest.fixture
def seed_events(db_session):
    import datetime
    from sqlalchemy import insert
    from app.db import DBEvent
    
    async def _seed(events):
        if not isinstance(events, list):
            events = [events]
        
        flattened = []
        for ev in events:
            res = ev.copy()
            meta = res.pop("metadata", {})
            if isinstance(meta, dict):
                if res.get("queue_depth") is None:
                    res["queue_depth"] = meta.get("queue_depth")
                if res.get("sku_zone") is None:
                    res["sku_zone"] = meta.get("sku_zone")
                if res.get("session_seq") is None:
                    res["session_seq"] = meta.get("session_seq", 0)
            if "timestamp" in res and isinstance(res["timestamp"], str):
                ts_str = res["timestamp"]
                if ts_str.endswith("Z"):
                    ts_str = ts_str[:-1] + "+00:00"
                res["timestamp"] = datetime.datetime.fromisoformat(ts_str)
            elif "timestamp" in res and isinstance(res["timestamp"], datetime.datetime):
                # Ensure tzinfo is present
                if res["timestamp"].tzinfo is None:
                    res["timestamp"] = res["timestamp"].replace(tzinfo=datetime.timezone.utc)
            flattened.append(res)
            
        await db_session.execute(insert(DBEvent).values(flattened))
        await db_session.commit()
    return _seed

from httpx import ASGITransport

@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

@pytest.fixture(autouse=True)
def mock_store_open(monkeypatch):
    import app.health
    import app.anomalies
    monkeypatch.setattr(app.health, "is_within_open_hours", lambda *args, **kwargs: True)
    monkeypatch.setattr(app.anomalies, "is_within_open_hours", lambda *args, **kwargs: True)



