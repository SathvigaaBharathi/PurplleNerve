import logging

logger = logging.getLogger(__name__)

def point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    """Ray casting algorithm to determine if point (x, y) is inside a polygon."""
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def get_zone_for_centroid(
    cx: int, cy: int, 
    frame_w: int, frame_h: int, 
    camera_zones: dict
) -> tuple[str | None, float | None]:
    """
    Maps pixel coordinates to store zones using polygon containment.
    If multiple zones overlap, assigns the zone with the smallest area.
    """
    if not camera_zones:
        return None, None
        
    # 1. Normalise coordinates to [0, 1]
    nx = cx / frame_w
    ny = cy / frame_h
    
    matched_zones = []
    
    # 2. Run ray-casting test
    for zone_id, zone_data in camera_zones.items():
        polygon = zone_data.get("polygon")
        area = zone_data.get("area", 1.0)
        
        if not polygon:
            continue
            
        if point_in_polygon(nx, ny, polygon):
            matched_zones.append((zone_id, area))
            
    if not matched_zones:
        logger.debug(f"Centroid ({cx},{cy}) normalized to ({nx:.2f},{ny:.2f}) is UNZONED")
        return None, None
        
    # 3. Sort by area (ascending) to resolve overlaps (assign smallest area)
    matched_zones.sort(key=lambda x: x[1])
    return matched_zones[0][0], matched_zones[0][1]
