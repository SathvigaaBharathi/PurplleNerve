import time
import logging
import uuid
from datetime import datetime, timezone
from pipeline.reid import compute_cosine_similarity

logger = logging.getLogger(__name__)

GRACE_WINDOW_SECONDS = 8.0

class SessionManager:
    """
    Tracks active visitor sessions with a soft-exit buffer.
    
    When a track disappears from all cameras, we do NOT immediately emit EXIT.
    Instead, we hold the session open for GRACE_WINDOW_SECONDS.
    
    If the same Re-ID embedding reappears within that window
    (cosine similarity > 0.82 to any active grace session),
    we continue the existing session — it was occlusion, not exit.
    
    If similarity is between 0.72 and 0.82 during the grace window,
    we treat it as a NEW visitor (different person, same direction).
    
    Only after GRACE_WINDOW_SECONDS does EXIT fire.
    """
    def __init__(self):
        # active_sessions: visitor_id -> dict of session data
        self.active_sessions = {}
        # grace_sessions: visitor_id -> dict of session data (hold-exit window)
        self.grace_sessions = {}
        # track_to_visitor: track_id -> visitor_id (convenience mapping)
        self.track_to_visitor = {}
        
    def get_visitor_id(self, track_id: int) -> str | None:
        return self.track_to_visitor.get(track_id)

    def register_track(
        self, 
        track_id: int, 
        embedding, 
        timestamp: datetime,
        store_id: str,
        camera_id: str,
        is_staff: bool,
        confidence: float
    ) -> tuple[str, str | None]:
        """
        Registers a new track. Checks for re-entry match in grace sessions.
        Returns (visitor_id, event_to_emit).
        """
        # 1. Check if track already mapped to active session
        if track_id in self.track_to_visitor:
            vid = self.track_to_visitor[track_id]
            # Update last seen timestamp
            if vid in self.active_sessions:
                self.active_sessions[vid]["last_seen"] = timestamp
            return vid, None

        # 2. Check for match in grace sessions
        best_match_vid = None
        best_sim = -1.0
        
        for vid, grace_data in self.grace_sessions.items():
            sim = compute_cosine_similarity(embedding, grace_data["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_match_vid = vid

        # 3. Handle matching scenarios
        # Case A: Matches active grace session (> 0.82) -> RE-ASSOCIATE (Occlusion merge)
        if best_match_vid and best_sim > 0.82:
            logger.info(f"Re-associating track {track_id} to grace session {best_match_vid} (sim: {best_sim:.3f})")
            
            # Retrieve from grace
            session_data = self.grace_sessions.pop(best_match_vid)
            session_data["last_seen"] = timestamp
            session_data["track_id"] = track_id
            
            # Put back into active
            self.active_sessions[best_match_vid] = session_data
            self.track_to_visitor[track_id] = best_match_vid
            
            # Emit REENTRY event
            return best_match_vid, "REENTRY"

        # Case B: Similarity between 0.72 and 0.82 or lower -> NEW visitor
        # Create a new visitor session
        new_vid = f"VIS_{str(uuid.uuid4())[:8]}"
        session_data = {
            "visitor_id": new_vid,
            "track_id": track_id,
            "store_id": store_id,
            "camera_id": camera_id,
            "embedding": embedding,
            "last_seen": timestamp,
            "started_at": timestamp,
            "is_staff": is_staff,
            "confidence": confidence,
            "current_zone": None,
            "zone_entry_time": None,
            "session_seq": 0
        }
        
        self.active_sessions[new_vid] = session_data
        self.track_to_visitor[track_id] = new_vid
        
        return new_vid, "ENTRY"

    def disappear_track(self, track_id: int, timestamp: datetime) -> list[dict]:
        """
        Called when a track is no longer detected. Moves track to grace session hold.
        Does NOT emit exit immediately.
        """
        events_emitted = []
        vid = self.track_to_visitor.get(track_id)
        if not vid:
            return events_emitted
            
        if vid in self.active_sessions:
            # Move session data from active_sessions to grace_sessions
            session_data = self.active_sessions.pop(vid)
            session_data["last_seen"] = timestamp
            self.grace_sessions[vid] = session_data
            
            # Remove track mapping since the track is dead, but session stays in grace
            self.track_to_visitor.pop(track_id, None)
            
            # If they were in a zone, emit ZONE_EXIT
            curr_zone = session_data.get("current_zone")
            if curr_zone:
                dwell_ms = int((timestamp - session_data["zone_entry_time"]).total_seconds() * 1000)
                events_emitted.append({
                    "visitor_id": vid,
                    "event_type": "ZONE_EXIT",
                    "zone_id": curr_zone,
                    "dwell_ms": dwell_ms,
                    "timestamp": timestamp,
                    "store_id": session_data["store_id"],
                    "camera_id": session_data["camera_id"],
                    "is_staff": session_data["is_staff"],
                    "confidence": session_data["confidence"]
                })
                
        return events_emitted

    def update_grace_sessions(self, current_time: datetime) -> list[dict]:
        """
        Evaluates grace sessions. Fires EXIT for sessions that expired.
        """
        expired_vids = []
        exit_events = []
        
        for vid, grace_data in self.grace_sessions.items():
            last_seen = grace_data["last_seen"]
            # Ensure timezone compatibility
            if last_seen.tzinfo is None:
                last_seen = last_seen.replace(tzinfo=timezone.utc)
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=timezone.utc)
                
            elapsed = (current_time - last_seen).total_seconds()
            
            if elapsed >= GRACE_WINDOW_SECONDS:
                expired_vids.append(vid)
                
                # Create EXIT event
                exit_events.append({
                    "visitor_id": vid,
                    "event_type": "EXIT",
                    "zone_id": None,
                    "dwell_ms": int((last_seen - grace_data["started_at"]).total_seconds() * 1000),
                    "timestamp": last_seen, # exit timestamp matches when they were last seen
                    "store_id": grace_data["store_id"],
                    "camera_id": grace_data["camera_id"],
                    "is_staff": grace_data["is_staff"],
                    "confidence": grace_data["confidence"]
                })

        for vid in expired_vids:
            self.grace_sessions.pop(vid)
            
        return exit_events
