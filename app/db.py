import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import String, Integer, Boolean, Float, DateTime, Numeric, Index, text
from datetime import datetime
from typing import Optional

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://retail:retail_secret@postgres:5432/retail_intelligence"
)

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class DBEvent(Base):
    __tablename__ = "events"

    event_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    store_id: Mapped[str] = mapped_column(String(50), nullable=False)
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False)
    visitor_id: Mapped[str] = mapped_column(String(50), nullable=False)
    event_type: Mapped[str] = mapped_column(String(30), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    zone_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    dwell_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_staff: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    queue_depth: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    sku_zone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    session_seq: Mapped[int] = mapped_column(Integer, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=text("now()"), 
        nullable=False
    )

class DBPosTransaction(Base):
    __tablename__ = "pos_transactions"

    transaction_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    store_id: Mapped[str] = mapped_column(String(50), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    basket_value: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    matched_visitor: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

class DBSessionConversion(Base):
    __tablename__ = "session_conversions"

    store_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    visitor_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    converted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=text("now()"), 
        nullable=False
    )

# Indexes as specified
Index("idx_events_store_ts", DBEvent.store_id, DBEvent.timestamp.desc())
Index("idx_events_visitor", DBEvent.store_id, DBEvent.visitor_id, DBEvent.timestamp)
Index("idx_events_type", DBEvent.store_id, DBEvent.event_type, DBEvent.timestamp.desc())
Index("idx_events_zone", DBEvent.store_id, DBEvent.zone_id, DBEvent.timestamp.desc())
Index("idx_pos_store_ts", DBPosTransaction.store_id, DBPosTransaction.timestamp.desc())

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

async def init_db():
    async with engine.begin() as conn:
        # Create tables
        await conn.run_sync(Base.metadata.create_all)
        
        # Create materialized view and its unique index
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
