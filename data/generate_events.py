import json
import uuid
import random
from datetime import datetime, timedelta

def generate_events():
    events = []
    start_time = datetime(2026, 3, 3, 10, 0, 0)
    
    # 1. Visitor 1 (completed funnel with purchase TXN_00441 at 14:38:12)
    v1_id = "VIS_v1_" + str(uuid.uuid4())[:8]
    v1_ts = datetime(2026, 3, 3, 14, 30, 0)
    events.extend([
        {"event_id": str(uuid.uuid4()), "store_id": "STORE_BLR_002", "camera_id": "CAM_ENTRY_01", "visitor_id": v1_id, "event_type": "ENTRY", "timestamp": v1_ts.isoformat() + "Z", "zone_id": None, "dwell_ms": 0, "is_staff": False, "confidence": 0.95, "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 0}},
        {"event_id": str(uuid.uuid4()), "store_id": "STORE_BLR_002", "camera_id": "CAM_FLOOR_01", "visitor_id": v1_id, "event_type": "ZONE_ENTER", "timestamp": (v1_ts + timedelta(seconds=10)).isoformat() + "Z", "zone_id": "SKINCARE", "dwell_ms": 0, "is_staff": False, "confidence": 0.90, "metadata": {"queue_depth": None, "sku_zone": "MOISTURISER", "session_seq": 1}},
        {"event_id": str(uuid.uuid4()), "store_id": "STORE_BLR_002", "camera_id": "CAM_FLOOR_01", "visitor_id": v1_id, "event_type": "ZONE_DWELL", "timestamp": (v1_ts + timedelta(seconds=40)).isoformat() + "Z", "zone_id": "SKINCARE", "dwell_ms": 30000, "is_staff": False, "confidence": 0.88, "metadata": {"queue_depth": None, "sku_zone": "MOISTURISER", "session_seq": 2}},
        {"event_id": str(uuid.uuid4()), "store_id": "STORE_BLR_002", "camera_id": "CAM_FLOOR_01", "visitor_id": v1_id, "event_type": "ZONE_EXIT", "timestamp": (v1_ts + timedelta(seconds=70)).isoformat() + "Z", "zone_id": "SKINCARE", "dwell_ms": 60000, "is_staff": False, "confidence": 0.91, "metadata": {"queue_depth": None, "sku_zone": "MOISTURISER", "session_seq": 3}},
        {"event_id": str(uuid.uuid4()), "store_id": "STORE_BLR_002", "camera_id": "CAM_BILLING_01", "visitor_id": v1_id, "event_type": "ZONE_ENTER", "timestamp": (v1_ts + timedelta(seconds=80)).isoformat() + "Z", "zone_id": "BILLING", "dwell_ms": 0, "is_staff": False, "confidence": 0.92, "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 4}},
        {"event_id": str(uuid.uuid4()), "store_id": "STORE_BLR_002", "camera_id": "CAM_BILLING_01", "visitor_id": v1_id, "event_type": "BILLING_QUEUE_JOIN", "timestamp": (v1_ts + timedelta(seconds=90)).isoformat() + "Z", "zone_id": "BILLING", "dwell_ms": 0, "is_staff": False, "confidence": 0.94, "metadata": {"queue_depth": 3, "sku_zone": None, "session_seq": 5}},
        {"event_id": str(uuid.uuid4()), "store_id": "STORE_BLR_002", "camera_id": "CAM_BILLING_01", "visitor_id": v1_id, "event_type": "ZONE_DWELL", "timestamp": (v1_ts + timedelta(seconds=120)).isoformat() + "Z", "zone_id": "BILLING", "dwell_ms": 30000, "is_staff": False, "confidence": 0.89, "metadata": {"queue_depth": 3, "sku_zone": None, "session_seq": 6}},
        {"event_id": str(uuid.uuid4()), "store_id": "STORE_BLR_002", "camera_id": "CAM_ENTRY_01", "visitor_id": v1_id, "event_type": "EXIT", "timestamp": (v1_ts + timedelta(seconds=500)).isoformat() + "Z", "zone_id": None, "dwell_ms": 0, "is_staff": False, "confidence": 0.90, "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 7}}
    ])
    
    # 2. Staff Member (should be excluded from metrics)
    staff_id = "STAFF_s1_" + str(uuid.uuid4())[:8]
    staff_ts = datetime(2026, 3, 3, 11, 0, 0)
    events.extend([
        {"event_id": str(uuid.uuid4()), "store_id": "STORE_BLR_002", "camera_id": "CAM_ENTRY_01", "visitor_id": staff_id, "event_type": "ENTRY", "timestamp": staff_ts.isoformat() + "Z", "zone_id": None, "dwell_ms": 0, "is_staff": True, "confidence": 0.97, "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 0}},
        {"event_id": str(uuid.uuid4()), "store_id": "STORE_BLR_002", "camera_id": "CAM_FLOOR_01", "visitor_id": staff_id, "event_type": "ZONE_ENTER", "timestamp": (staff_ts + timedelta(seconds=20)).isoformat() + "Z", "zone_id": "SKINCARE", "dwell_ms": 0, "is_staff": True, "confidence": 0.96, "metadata": {"queue_depth": None, "sku_zone": "MOISTURISER", "session_seq": 1}},
        {"event_id": str(uuid.uuid4()), "store_id": "STORE_BLR_002", "camera_id": "CAM_ENTRY_01", "visitor_id": staff_id, "event_type": "EXIT", "timestamp": (staff_ts + timedelta(minutes=30)).isoformat() + "Z", "zone_id": None, "dwell_ms": 0, "is_staff": True, "confidence": 0.95, "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 2}}
    ])

    # 3. Visitor 2 (re-entry, same visitor_id returns after exit, or soft-exit grace window)
    v2_id = "VIS_v2_" + str(uuid.uuid4())[:8]
    v2_ts = datetime(2026, 3, 3, 15, 0, 0)
    events.extend([
        {"event_id": str(uuid.uuid4()), "store_id": "STORE_BLR_002", "camera_id": "CAM_ENTRY_01", "visitor_id": v2_id, "event_type": "ENTRY", "timestamp": v2_ts.isoformat() + "Z", "zone_id": None, "dwell_ms": 0, "is_staff": False, "confidence": 0.93, "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 0}},
        {"event_id": str(uuid.uuid4()), "store_id": "STORE_BLR_002", "camera_id": "CAM_ENTRY_01", "visitor_id": v2_id, "event_type": "EXIT", "timestamp": (v2_ts + timedelta(seconds=120)).isoformat() + "Z", "zone_id": None, "dwell_ms": 0, "is_staff": False, "confidence": 0.91, "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 1}},
        # Re-entry event
        {"event_id": str(uuid.uuid4()), "store_id": "STORE_BLR_002", "camera_id": "CAM_ENTRY_01", "visitor_id": v2_id, "event_type": "REENTRY", "timestamp": (v2_ts + timedelta(seconds=180)).isoformat() + "Z", "zone_id": None, "dwell_ms": 0, "is_staff": False, "confidence": 0.92, "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 2}},
        {"event_id": str(uuid.uuid4()), "store_id": "STORE_BLR_002", "camera_id": "CAM_FLOOR_01", "visitor_id": v2_id, "event_type": "ZONE_ENTER", "timestamp": (v2_ts + timedelta(seconds=200)).isoformat() + "Z", "zone_id": "MOISTURISER", "dwell_ms": 0, "is_staff": False, "confidence": 0.90, "metadata": {"queue_depth": None, "sku_zone": "MOISTURISER", "session_seq": 3}},
        {"event_id": str(uuid.uuid4()), "store_id": "STORE_BLR_002", "camera_id": "CAM_ENTRY_01", "visitor_id": v2_id, "event_type": "EXIT", "timestamp": (v2_ts + timedelta(seconds=400)).isoformat() + "Z", "zone_id": None, "dwell_ms": 0, "is_staff": False, "confidence": 0.89, "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 4}}
    ])

    # 4. Fill up to 200 events with random but realistic schema-compliant events
    event_types = ["ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL"]
    zones = ["SKINCARE", "MOISTURISER", "BILLING"]
    cameras = ["CAM_FLOOR_01", "CAM_BILLING_01"]
    
    current_ts = start_time
    for i in range(len(events), 200):
        # Create a visitor
        visitor_id = f"VIS_gen_{i % 15}"
        e_type = random.choice(event_types)
        z_id = random.choice(zones)
        if z_id == "BILLING":
            cam_id = "CAM_BILLING_01"
        else:
            cam_id = "CAM_FLOOR_01"
            
        current_ts += timedelta(seconds=random.randint(10, 60))
        
        ev = {
            "event_id": str(uuid.uuid4()),
            "store_id": "STORE_BLR_002",
            "camera_id": cam_id,
            "visitor_id": visitor_id,
            "event_type": e_type,
            "timestamp": current_ts.isoformat() + "Z",
            "zone_id": z_id,
            "dwell_ms": 30000 if e_type == "ZONE_DWELL" else 0,
            "is_staff": False,
            "confidence": round(random.uniform(0.4, 0.99), 2),
            "metadata": {
                "queue_depth": None if e_type != "BILLING_QUEUE_JOIN" else random.randint(1, 5),
                "sku_zone": "MOISTURISER" if z_id == "MOISTURISER" else None,
                "session_seq": i // 15
            }
        }
        events.append(ev)
        
    with open("D:\\purplle\\store-intelligence\\data\\sample_events.jsonl", "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

if __name__ == "__main__":
    generate_events()
