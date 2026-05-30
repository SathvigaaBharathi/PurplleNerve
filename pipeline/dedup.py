import numpy as np
import logging
from pipeline.reid import compute_cosine_similarity

logger = logging.getLogger(__name__)

def calculate_iou(boxA, boxB):
    """Calculate the Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0
        
    return interArea / float(unionArea)

class SpatialRegistry:
    """
    Cross-camera deduplication using a shared spatial map.
    
    Problem: The main floor camera overlaps with the entry camera FOV.
    The same person walking through the entry zone appears in both feeds.
    """
    def __init__(self):
        # registry: (store_id, timestamp_bucket) -> list of active tracks (cam_id, box, embedding, visitor_id)
        self.registry = {}
        
    def _get_bucket(self, timestamp) -> int:
        # 1-second timestamp bucket
        if hasattr(timestamp, "timestamp"):
            return int(timestamp.timestamp())
        return int(timestamp)

    def project_box_homography(self, box, homography_matrix):
        """Projects bounding box using a 3x3 homography matrix."""
        if homography_matrix is None:
            return box
            
        try:
            # Bounding box coordinates: [x1, y1, x2, y2]
            # Convert to centroid in homogeneous coordinates [cx, cy, 1]
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            pt = np.array([cx, cy, 1.0])
            
            # Project using H * pt
            projected = np.dot(np.array(homography_matrix), pt)
            px, py = projected[0]/projected[2], projected[1]/projected[2]
            
            # Reconstruct dummy bounding box around projected point
            half_w = (box[2] - box[0]) / 2.0
            half_h = (box[3] - box[1]) / 2.0
            return [px - half_w, py - half_h, px + half_w, py + half_h]
        except Exception as e:
            logger.error(f"Error projecting box: {e}")
            return box

    def should_suppress(
        self, 
        store_id: str, 
        camera_id: str, 
        box: list[float], 
        embedding: np.ndarray, 
        timestamp,
        homography_matrix = None
    ) -> tuple[bool, str | None]:
        """
        Checks if a detection should be suppressed.
        Returns (should_suppress, matching_visitor_id).
        """
        bucket = self._get_bucket(timestamp)
        key = (store_id, bucket)
        
        if key not in self.registry:
            # Initialize bucket registry list
            self.registry[key] = []
            
        # Check against existing entries in this bucket
        for entry in self.registry[key]:
            ent_cam_id, ent_box, ent_emb, ent_vid = entry
            
            if ent_cam_id == camera_id:
                continue # don't de-duplicate against same camera
                
            # Compute embedding similarity
            similarity = compute_cosine_similarity(embedding, ent_emb)
            
            # Project box to align spaces if homography is available
            projected_box = self.project_box_homography(box, homography_matrix)
            iou = calculate_iou(projected_box, ent_box)
            
            # Check suppression conditions:
            # - Bounding boxes IoU > 0.40 after homography
            # - OR Re-ID cosine similarity > 0.82
            if iou > 0.40 or similarity > 0.82:
                logger.info(
                    f"Deduplicated track in {camera_id}: matched existing {ent_cam_id} track "
                    f"with IoU {iou:.2f} and Re-ID similarity {similarity:.2f}"
                )
                return True, ent_vid

        return False, None

    def register_detection(self, store_id: str, camera_id: str, box: list[float], embedding: np.ndarray, visitor_id: str, timestamp):
        """Registers a non-suppressed detection in the registry."""
        bucket = self._get_bucket(timestamp)
        key = (store_id, bucket)
        if key not in self.registry:
            self.registry[key] = []
            
        self.registry[key].append((camera_id, box, embedding, visitor_id))
        
    def prune_old_buckets(self, current_timestamp, max_age_seconds=10):
        """Removes entries older than max_age_seconds from registry."""
        curr_bucket = self._get_bucket(current_timestamp)
        keys_to_delete = []
        for key in self.registry.keys():
            store_id, bucket = key
            if curr_bucket - bucket > max_age_seconds:
                keys_to_delete.append(key)
                
        for k in keys_to_delete:
            self.registry.pop(k, None)
