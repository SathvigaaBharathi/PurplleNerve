from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, List
import uuid
from datetime import datetime

EventType = Literal[
    "ENTRY", "EXIT", "ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL",
    "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON", "REENTRY"
]

class EventMetadata(BaseModel):
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: int = Field(ge=0)

class RetailEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    store_id: str
    camera_id: str
    visitor_id: str
    event_type: EventType
    timestamp: datetime
    zone_id: Optional[str] = None
    dwell_ms: int = Field(ge=0)
    is_staff: bool
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: EventMetadata

    @field_validator('event_id')
    @classmethod
    def validate_uuid(cls, v):
        try:
            uuid.UUID(str(v))  # raises if invalid
        except ValueError:
            raise ValueError('Invalid UUID format')
        return str(v)

    @field_validator('zone_id')
    @classmethod
    def zone_required_for_zone_events(cls, v, info):
        if info.data.get('event_type') in ('ZONE_ENTER', 'ZONE_EXIT', 'ZONE_DWELL') and v is None:
            raise ValueError('zone_id required for zone events')
        return v

class IngestPayload(BaseModel):
    events: List[RetailEvent]
