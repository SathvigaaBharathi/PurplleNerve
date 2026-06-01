import cv2
import numpy as np
import argparse
import os
import json
import logging
import asyncio
from datetime import datetime, timedelta, timezone
import time as time_module

from pipeline.staff import classify_staff
from pipeline.zones import get_zone_for_centroid
from pipeline.reid import AppearanceReIDModel, MockReIDModel
from pipeline.tracker import SessionManager
from pipeline.dedup import SpatialRegistry
from pipeline.emit import EventEmitter
from app.health import load_store_layout

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0.0:
        return 0.0
    return interArea / float(unionArea)

class SimpleIoUTracker:
    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.next_track_id = 1
        self.active_tracks = {} # track_id -> {"box": box, "confidence": conf}

    def update(self, detections):
        """
        detections: list of dicts [{"box": box, "confidence": conf}]
        Returns:
            results: list of dicts [{"track_id": int, "box": box, "confidence": conf}]
            disappeared_ids: list of int
        """
        updated_tracks = {}
        matched_detections = [False] * len(detections)
        disappeared_ids = []
        
        # 1. Greedy match active tracks to detections
        for track_id, track_data in self.active_tracks.items():
            track_box = track_data["box"]
            best_iou = -1.0
            best_det_idx = -1
            
            for idx, det in enumerate(detections):
                if matched_detections[idx]:
                    continue
                iou = calculate_iou(track_box, det["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = idx
            
            if best_det_idx != -1 and best_iou >= self.iou_threshold:
                updated_tracks[track_id] = {
                    "box": detections[best_det_idx]["box"],
                    "confidence": detections[best_det_idx]["confidence"]
                }
                matched_detections[best_det_idx] = True
            else:
                disappeared_ids.append(track_id)
                
        # 2. Start new tracks for unmatched detections
        for idx, det in enumerate(detections):
            if not matched_detections[idx]:
                track_id = self.next_track_id
                self.next_track_id += 1
                updated_tracks[track_id] = {
                    "box": det["box"],
                    "confidence": det["confidence"]
                }
                
        self.active_tracks = updated_tracks
        
        # Format results
        results = []
        for track_id, data in self.active_tracks.items():
            results.append({
                "track_id": track_id,
                "box": data["box"],
                "confidence": data["confidence"]
            })
        return results, disappeared_ids

async def process_clip(
    clip_path: str,
    store_id: str,
    layout_path: str,
    output_path: str,
    redis_url: str = None,
    real: bool = False,
    max_frames: int = 0,
    api_url: str = None,
    loop: bool = False,
    camera_id_override: str = None,
    conf_threshold: float = 0.50,
):
    logger.info(f"Starting pipeline on clip={clip_path}, store_id={store_id}, layout={layout_path}, real={real}, max_frames={max_frames}, conf_threshold={conf_threshold}")
    
    # 1. Load store layout
    # Try searching paths for layout
    layout_data = load_store_layout()
    store_conf = layout_data.get(store_id, {})
    if not store_conf:
        logger.error(f"Store config not found for store_id={store_id}")
        return

    # Determine camera_id from filename or default
    basename = os.path.basename(clip_path).upper()
    camera_id = "CAM_ENTRY_01"
    if "CAM 1" in basename or "CAM_ENTRY" in basename or "ENTRY" in basename:
        camera_id = "CAM_ENTRY_01"
    elif "CAM 2" in basename:
        camera_id = "CAM_FLOOR_01"
    elif "CAM 3" in basename:
        camera_id = "CAM_BILLING_01"
    elif "CAM 4" in basename:
        camera_id = "CAM_FLOOR_02"
    elif "CAM 5" in basename:
        camera_id = "CAM_BILLING_02"
    elif "CAM_FLOOR" in basename or "FLOOR" in basename:
        camera_id = "CAM_FLOOR_01"
    elif "CAM_BILLING" in basename or "BILLING" in basename:
        camera_id = "CAM_BILLING_01"
            
    # Override camera_id if explicitly provided
    if camera_id_override:
        camera_id = camera_id_override
        logger.info(f"camera_id overridden to {camera_id}")

    # Resolve camera config
    camera_conf = store_conf.get("cameras", {}).get(camera_id, {})
    camera_zones = camera_conf.get("zones", {})
    homography_matrix = camera_conf.get("homography", {}).get("CAM_ENTRY_01", None)
    staff_hue = store_conf.get("staff_uniform_hue_range", [95, 115])
    camera_type = camera_conf.get("camera_type", "entry")

    # 2. Init components
    session_manager = SessionManager()
    spatial_registry = SpatialRegistry()
    reid_model = AppearanceReIDModel()
    emitter = EventEmitter(output_jsonl_path=output_path, redis_url=redis_url, api_url=api_url)
    
    import httpx
    http_client = httpx.AsyncClient()
    
    if redis_url:
        await emitter.connect_redis()

    tracker = None
    yolov9_model = None
    rtdetr_processor = None
    rtdetr_model = None
    device = "cpu"

    if real:
        tracker = SimpleIoUTracker(iou_threshold=0.3)
        if camera_type == "billing":
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
            import torch
            logger.info("Loading RT-DETR model...")

            # Run blocking HuggingFace from_pretrained calls in a thread pool.
            # from_pretrained uses a synchronous httpx client internally; calling
            # it directly inside an async coroutine causes 'client has been closed'
            # errors when two billing cameras load the model concurrently.
            def _load_rtdetr():
                processor = RTDetrImageProcessor.from_pretrained(
                    "PekingU/rtdetr_r50vd",
                    local_files_only=False,
                )
                model = RTDetrForObjectDetection.from_pretrained(
                    "PekingU/rtdetr_r50vd",
                    local_files_only=False,
                )
                return processor, model

            rtdetr_processor, rtdetr_model = await asyncio.to_thread(_load_rtdetr)
            rtdetr_model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            rtdetr_model.to(device)
            logger.info(f"RT-DETR ready on device={device}")
        else:
            logger.info("Loading YOLOv9s model...")

            def _load_yolo():
                from ultralytics import YOLO
                return YOLO("yolov9s.pt")

            yolov9_model = await asyncio.to_thread(_load_yolo)
            logger.info("YOLOv9s ready")

    # 3. Load clip with OpenCV
    if not os.path.exists(clip_path):
        logger.error(f"Clip path not found: {clip_path}")
        return
        
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video clip: {clip_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1000

    logger.info(f"Video specs: {frame_w}x{frame_h} @ {fps}fps, total_frames={total_frames}")

    # Track sequence IDs per visitor session
    session_sequences = {} # visitor_id -> current seq count

    # Baseline time for events
    start_time = datetime.now(timezone.utc)

    frame_idx = 0
    batch_size = 8
    frames_batch = []
    
    # Track states for zones
    # visitor_id -> {zone_id, enter_time, last_dwell_emit_time}
    visitor_zones = {}
    last_stream_time = 0.0
    last_log_time = 0.0
    frames_processed = 0
    frames_skipped = 0
    tracked_objs = []

    # How many frames to skip between inference runs.
    # On CPU, YOLOv9 runs ~1 fps on 1080p.  We want to stream video at the
    # actual clip fps (25 fps) so we run inference only every INFER_EVERY
    # frames and push every frame to the stream endpoint.
    INFER_EVERY = max(1, int(fps))          # infer once per second of video
    STREAM_INTERVAL = 0.20                 # push to server at 5 fps


    # Define nested helper functions for background/asynchronous inference
    def detect_objects_sync(frame_to_detect):
        detections = []
        if camera_type == "billing":
            import torch
            img_rgb = cv2.cvtColor(frame_to_detect, cv2.COLOR_BGR2RGB)
            inputs = rtdetr_processor(images=img_rgb, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = rtdetr_model(**inputs)
            
            results = rtdetr_processor.post_process_object_detection(
                outputs,
                target_sizes=[(frame_h, frame_w)],
                threshold=conf_threshold
            )
            
            boxes = results[0]["boxes"].cpu().numpy()
            labels = results[0]["labels"].cpu().numpy()
            scores = results[0]["scores"].cpu().numpy()
            
            for box, label, score in zip(boxes, labels, scores):
                if label == 0:
                    detections.append({
                        "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        "confidence": float(score)
                    })
        else:
            results = yolov9_model(frame_to_detect, classes=[0], verbose=False)
            boxes = results[0].boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 0 and conf >= conf_threshold:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1f, y1f, x2f, y2f = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                    bbox_h = y2f - y1f
                    bbox_w = x2f - x1f
                    # Filter tiny detections (mirrors, photos, banners):
                    if bbox_h < 60 or bbox_w < 25:
                        continue
                    # Suppress very flat detections
                    if bbox_h > 0 and (bbox_w / bbox_h) > 3.0:
                        continue
                    detections.append({
                        "box": [x1f, y1f, x2f, y2f],
                        "confidence": conf
                    })
        return detections

    async def process_detections_and_sync(detections, frame_to_detect, infer_ts):
        nonlocal tracked_objs, frames_processed
        frames_processed += 1
        
        # Update tracker
        curr_tracked_objs, disappeared_track_ids = tracker.update(detections)
        
        # Emit exits for disappeared tracks
        for t_id in disappeared_track_ids:
            disappear_events = session_manager.disappear_track(t_id, infer_ts)
            for ev in disappear_events:
                seq = session_sequences.get(ev["visitor_id"], 0)
                await emitter.emit(
                    store_id=store_id,
                    camera_id=camera_id,
                    visitor_id=ev["visitor_id"],
                    event_type=ev["event_type"],
                    timestamp=ev["timestamp"],
                    zone_id=ev["zone_id"],
                    dwell_ms=ev["dwell_ms"],
                    is_staff=ev["is_staff"],
                    confidence=ev["confidence"],
                    session_seq=seq
                )
                
        # Gather crops, extract embeddings, check staff
        tracks_payload = []
        temp_track_data = {}
        
        for obj in curr_tracked_objs:
            track_id = obj["track_id"]
            box = obj["box"]
            conf = obj["confidence"]
            
            x1, y1, x2, y2 = box
            x1_c = max(0, int(x1))
            y1_c = max(0, int(y1))
            x2_c = min(frame_w, int(x2))
            y2_c = min(frame_h, int(y2))
            
            crop = None
            is_staff, staff_conf = False, 0.0
            if x2_c > x1_c and y2_c > y1_c:
                crop = frame_to_detect[y1_c:y2_c, x1_c:x2_c]
                crop_h = crop.shape[0]
                upper_limit = max(1, int(crop_h * 0.40))
                upper_crop = crop[0:upper_limit, :]
                is_staff, staff_conf = classify_staff(upper_crop, staff_hue)
                
            embedding = reid_model.extract_embedding(crop, track_id)
            local_vid = session_manager.track_to_visitor.get(track_id)
            
            tracks_payload.append({
                "track_id": track_id,
                "visitor_id": local_vid,
                "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                "is_staff": is_staff
            })
            
            temp_track_data[track_id] = {
                "box": box,
                "conf": conf,
                "is_staff": is_staff,
                "embedding": embedding
            }
            
        # Post tracks to Re-ID sync endpoint
        synced_tracks = []
        if api_url and tracks_payload:
            try:
                response = await http_client.post(
                    f"{api_url}/stores/{store_id}/reid/sync",
                    json={"tracks": tracks_payload},
                    timeout=2.0
                )
                if response.status_code == 200:
                    synced_tracks = response.json().get("synced_tracks", [])
            except Exception as e:
                logger.error(f"Failed to sync Re-ID tracks: {e}")
                
        synced_map = {item["track_id"]: (item["visitor_id"], item["is_staff"]) for item in synced_tracks}
        
        # Register tracks in session manager and emit events
        final_tracked_objs = []
        
        for obj in curr_tracked_objs:
            track_id = obj["track_id"]
            box = obj["box"]
            conf = obj["confidence"]
            
            temp_data = temp_track_data[track_id]
            embedding = temp_data["embedding"]
            local_is_staff = temp_data["is_staff"]
            
            resolved_vid, resolved_is_staff = synced_map.get(track_id, (None, local_is_staff))
            
            should_sup, matched_vid = spatial_registry.should_suppress(
                store_id=store_id,
                camera_id=camera_id,
                box=box,
                embedding=embedding,
                timestamp=infer_ts,
                homography_matrix=homography_matrix
            )
            
            if should_sup:
                continue
                
            visitor_id, event_type = session_manager.register_track(
                track_id=track_id,
                embedding=embedding,
                timestamp=infer_ts,
                store_id=store_id,
                camera_id=camera_id,
                is_staff=resolved_is_staff or local_is_staff,
                confidence=conf,
                visitor_id_override=resolved_vid
            )
            
            if visitor_id in session_manager.active_sessions:
                session_manager.active_sessions[visitor_id]["is_staff"] = resolved_is_staff or local_is_staff
                
            is_staff_to_emit = session_manager.active_sessions[visitor_id]["is_staff"]
            
            spatial_registry.register_detection(
                store_id=store_id,
                camera_id=camera_id,
                box=box,
                embedding=embedding,
                visitor_id=visitor_id,
                timestamp=infer_ts
            )
            
            if visitor_id not in session_sequences:
                session_sequences[visitor_id] = 0
            else:
                session_sequences[visitor_id] += 1
            seq = session_sequences[visitor_id]
            
            if event_type in ("ENTRY", "REENTRY"):
                if event_type == "REENTRY" and visitor_id in session_manager.active_sessions:
                    session_manager.active_sessions[visitor_id]["reentry"] = True
                await emitter.emit(
                    store_id=store_id,
                    camera_id=camera_id,
                    visitor_id=visitor_id,
                    event_type=event_type,
                    timestamp=infer_ts,
                    is_staff=is_staff_to_emit,
                    confidence=conf,
                    session_seq=seq
                )
                
            cx = int((box[0] + box[2]) / 2.0)
            cy = int((box[1] + box[3]) / 2.0)
            zone_id, _ = get_zone_for_centroid(cx, cy, frame_w, frame_h, camera_zones)
            
            curr_zone_info = visitor_zones.get(visitor_id)
            
            if zone_id:
                if not curr_zone_info or curr_zone_info["zone_id"] != zone_id:
                    if curr_zone_info:
                        old_zone = curr_zone_info["zone_id"]
                        dwell_ms = int((infer_ts - curr_zone_info["enter_time"]).total_seconds() * 1000)
                        await emitter.emit(
                            store_id=store_id,
                            camera_id=camera_id,
                            visitor_id=visitor_id,
                            event_type="ZONE_EXIT",
                            timestamp=infer_ts,
                            zone_id=old_zone,
                            dwell_ms=dwell_ms,
                            is_staff=is_staff_to_emit,
                            confidence=conf,
                            session_seq=seq
                        )
                    visitor_zones[visitor_id] = {
                        "zone_id": zone_id,
                        "enter_time": infer_ts,
                        "last_dwell_emit_ms": 0,
                        "dwell_event_count": 0
                    }
                    await emitter.emit(
                        store_id=store_id,
                        camera_id=camera_id,
                        visitor_id=visitor_id,
                        event_type="ZONE_ENTER",
                        timestamp=infer_ts,
                        zone_id=zone_id,
                        is_staff=is_staff_to_emit,
                        confidence=conf,
                        session_seq=seq
                    )
                    
                    if zone_id == "BILLING":
                        billing_count = 0
                        for other_obj in curr_tracked_objs:
                            o_box = other_obj["box"]
                            o_cx = int((o_box[0] + o_box[2]) / 2.0)
                            o_cy = int((o_box[1] + o_box[3]) / 2.0)
                            o_zone, _ = get_zone_for_centroid(o_cx, o_cy, frame_w, frame_h, camera_zones)
                            if o_zone == "BILLING":
                                billing_count += 1
                        if billing_count == 0:
                            billing_count = 1
                        await emitter.emit(
                            store_id=store_id,
                            camera_id=camera_id,
                            visitor_id=visitor_id,
                            event_type="BILLING_QUEUE_JOIN",
                            timestamp=infer_ts,
                            zone_id=zone_id,
                            is_staff=is_staff_to_emit,
                            confidence=conf,
                            queue_depth=billing_count,
                            session_seq=seq
                        )
                else:
                    DWELL_INTERVAL_MS = 30_000
                    time_in_zone_ms = int((infer_ts - curr_zone_info["enter_time"]).total_seconds() * 1000)
                    last_emit_ms = curr_zone_info.get("last_dwell_emit_ms", 0)
                    intervals_elapsed = (time_in_zone_ms - last_emit_ms) // DWELL_INTERVAL_MS
                    
                    if intervals_elapsed >= 1:
                        for i in range(int(intervals_elapsed)):
                            interval_dwell_ms = last_emit_ms + (i + 1) * DWELL_INTERVAL_MS
                            session_sequences[visitor_id] += 1
                            seq = session_sequences[visitor_id]
                            await emitter.emit(
                                store_id=store_id,
                                camera_id=camera_id,
                                visitor_id=visitor_id,
                                event_type="ZONE_DWELL",
                                timestamp=infer_ts,
                                zone_id=zone_id,
                                dwell_ms=interval_dwell_ms,
                                is_staff=is_staff_to_emit,
                                confidence=conf,
                                session_seq=seq
                            )
                        visitor_zones[visitor_id]["last_dwell_emit_ms"] = last_emit_ms + int(intervals_elapsed) * DWELL_INTERVAL_MS
                        visitor_zones[visitor_id]["dwell_event_count"] = curr_zone_info.get("dwell_event_count", 0) + int(intervals_elapsed)
            else:
                if curr_zone_info:
                    old_zone = curr_zone_info["zone_id"]
                    dwell_ms = int((infer_ts - curr_zone_info["enter_time"]).total_seconds() * 1000)
                    visitor_zones.pop(visitor_id)
                    await emitter.emit(
                        store_id=store_id,
                        camera_id=camera_id,
                        visitor_id=visitor_id,
                        event_type="ZONE_EXIT",
                        timestamp=infer_ts,
                        zone_id=old_zone,
                        dwell_ms=dwell_ms,
                        is_staff=is_staff_to_emit,
                        confidence=conf,
                        session_seq=seq
                    )
            final_tracked_objs.append(obj)
            
        exit_events = session_manager.update_grace_sessions(infer_ts)
        for ev in exit_events:
            seq = session_sequences.get(ev["visitor_id"], 0)
            await emitter.emit(
                store_id=store_id,
                camera_id=camera_id,
                visitor_id=ev["visitor_id"],
                event_type="EXIT",
                timestamp=ev["timestamp"],
                dwell_ms=ev["dwell_ms"],
                is_staff=ev["is_staff"],
                confidence=ev["confidence"],
                session_seq=seq
            )
            
        spatial_registry.prune_old_buckets(infer_ts)
        tracked_objs = final_tracked_objs

    inference_in_progress = False

    async def run_inference_bg(frame_copy, infer_ts):
        nonlocal inference_in_progress
        inference_in_progress = True
        try:
            detections = await asyncio.to_thread(detect_objects_sync, frame_copy)
            await process_detections_and_sync(detections, frame_copy, infer_ts)
        except Exception as e:
            logger.exception(f"Error in background inference task: {e}")
        finally:
            inference_in_progress = False

    start_wall_time = time_module.time()
    last_infer_wall_time = 0.0

    while cap.isOpened():
        if real:
            # Wall-clock pacing to achieve real-time speed by skipping frames
            elapsed = time_module.time() - start_wall_time
            target_frame_idx = int(elapsed * fps)
            if target_frame_idx >= total_frames:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    start_wall_time = time_module.time()
                    start_time = datetime.now(timezone.utc)
                    frame_idx = 0
                    continue
                else:
                    break
            if target_frame_idx > frame_idx:
                diff = target_frame_idx - frame_idx
                if diff < 150:
                    for _ in range(diff):
                        cap.grab()
                    frame_idx = target_frame_idx
                    frames_skipped += diff
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
                    frame_idx = target_frame_idx
                    frames_skipped += diff

        ret, frame = cap.read()
        if not ret:
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                start_time = datetime.now(timezone.utc)
                frame_idx = 0
                continue
            else:
                break

        frame_idx += 1

        if not real:
            # Simulated mode: pace to real fps
            await asyncio.sleep(1.0 / fps)

            # Loop early in simulated mode to repeat the customer journey pattern (0-1000 frames)
            if loop and frame_idx >= 1000:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                start_time = datetime.now(timezone.utc)
                frame_idx = 0
                continue

            frames_batch.append(frame)
            # When batch is full, process it
            if len(frames_batch) == batch_size:
                # We simulate time based on current frame index in video
                curr_ts = start_time + timedelta(seconds=(frame_idx / fps))
                
                # --- Perform Track & Detection Inferences ---
                # Generate simulated tracks for testing and validation
                simulated_tracks = []
                
                # Simulated Customer (Track 1) walking from Entry -> Floor -> Billing -> Exit
                # Frame 0 to 1000 is 66 seconds @ 15fps
                if camera_id == "CAM_ENTRY_01":
                    # Enters at beginning, exits at end
                    if frame_idx < 150: # entry threshold crossing
                        simulated_tracks.append({
                            "track_id": 1,
                            "box": [100.0, 50.0, 200.0, 250.0], # Bbox coordinates
                            "is_staff": False,
                            "confidence": 0.94
                        })
                    elif frame_idx > 800 and frame_idx < 950: # exit threshold crossing
                        simulated_tracks.append({
                            "track_id": 1,
                            "box": [100.0, 50.0, 200.0, 250.0],
                            "is_staff": False,
                            "confidence": 0.90
                        })
                        
                elif camera_id == "CAM_FLOOR_01":
                    # Present from frame 150 to 600
                    if frame_idx >= 150 and frame_idx <= 600:
                        simulated_tracks.append({
                            "track_id": 1,
                            "box": [200.0, 300.0, 350.0, 600.0], # In Skincare zone
                            "is_staff": False,
                            "confidence": 0.91
                        })
                        
                elif camera_id == "CAM_BILLING_01":
                    # Present from frame 600 to 800 in Billing zone
                    if frame_idx >= 600 and frame_idx <= 800:
                        simulated_tracks.append({
                            "track_id": 1,
                            "box": [500.0, 500.0, 650.0, 800.0],
                            "is_staff": False,
                            "confidence": 0.93
                        })

                # Simulated Staff member (Track 99)
                # Present on floor constantly
                if camera_id == "CAM_FLOOR_01" and frame_idx % 2 == 0:
                    simulated_tracks.append({
                        "track_id": 99,
                        "box": [600.0, 100.0, 700.0, 350.0],
                        "is_staff": True,
                        "confidence": 0.98
                    })

                # Process detections in batch
                for det in simulated_tracks:
                    track_id = det["track_id"]
                    box = det["box"]
                    is_staff = det["is_staff"]
                    conf = det["confidence"]
                    
                    # Extract mock embedding
                    embedding = reid_model.extract_embedding(None, track_id)
                    
                    # Check for cross-camera deduplication
                    should_sup, matched_vid = spatial_registry.should_suppress(
                        store_id=store_id,
                        camera_id=camera_id,
                        box=box,
                        embedding=embedding,
                        timestamp=curr_ts,
                        homography_matrix=homography_matrix
                    )
                    
                    if should_sup:
                        continue # Suppressed
                        
                    # Register track
                    visitor_id, event_type = session_manager.register_track(
                        track_id=track_id,
                        embedding=embedding,
                        timestamp=curr_ts,
                        store_id=store_id,
                        camera_id=camera_id,
                        is_staff=is_staff,
                        confidence=conf
                    )
                    
                    # Register in spatial registry for other cameras
                    spatial_registry.register_detection(
                        store_id=store_id,
                        camera_id=camera_id,
                        box=box,
                        embedding=embedding,
                        visitor_id=visitor_id,
                        timestamp=curr_ts
                    )
                    
                    # Setup session sequence
                    if visitor_id not in session_sequences:
                        session_sequences[visitor_id] = 0
                    else:
                        session_sequences[visitor_id] += 1
                    seq = session_sequences[visitor_id]

                    # Emit ENTRY or REENTRY if matching state machine
                    if event_type in ("ENTRY", "REENTRY"):
                        await emitter.emit(
                            store_id=store_id,
                            camera_id=camera_id,
                            visitor_id=visitor_id,
                            event_type=event_type,
                            timestamp=curr_ts,
                            is_staff=is_staff,
                            confidence=conf,
                            session_seq=seq
                        )
                        
                    # Map coordinates to zone
                    cx = int((box[0] + box[2]) / 2.0)
                    cy = int((box[1] + box[3]) / 2.0)
                    zone_id, _ = get_zone_for_centroid(cx, cy, frame_w, frame_h, camera_zones)
                    
                    # Handle zone transitions and dwell times
                    curr_zone_info = visitor_zones.get(visitor_id)
                    
                    if zone_id:
                        # Case A: Entered a new zone or transitioned
                        if not curr_zone_info or curr_zone_info["zone_id"] != zone_id:
                            if curr_zone_info:
                                # Exit old zone
                                old_zone = curr_zone_info["zone_id"]
                                dwell_ms = int((curr_ts - curr_zone_info["enter_time"]).total_seconds() * 1000)
                                await emitter.emit(
                                    store_id=store_id,
                                    camera_id=camera_id,
                                    visitor_id=visitor_id,
                                    event_type="ZONE_EXIT",
                                    timestamp=curr_ts,
                                    zone_id=old_zone,
                                    dwell_ms=dwell_ms,
                                    is_staff=is_staff,
                                    confidence=conf,
                                    session_seq=seq
                                )
                                
                            # Enter new zone
                            visitor_zones[visitor_id] = {
                                "zone_id": zone_id,
                                "enter_time": curr_ts,
                                "last_dwell_emit_ms": 0,
                                "dwell_event_count": 0
                            }
                            await emitter.emit(
                                store_id=store_id,
                                camera_id=camera_id,
                                visitor_id=visitor_id,
                                event_type="ZONE_ENTER",
                                timestamp=curr_ts,
                                zone_id=zone_id,
                                is_staff=is_staff,
                                confidence=conf,
                                session_seq=seq
                            )
                            
                            # Special Case: joined billing queue
                            if zone_id == "BILLING":
                                # Emit BILLING_QUEUE_JOIN
                                await emitter.emit(
                                    store_id=store_id,
                                    camera_id=camera_id,
                                    visitor_id=visitor_id,
                                    event_type="BILLING_QUEUE_JOIN",
                                    timestamp=curr_ts,
                                    zone_id=zone_id,
                                    is_staff=is_staff,
                                    confidence=conf,
                                    queue_depth=2, # mock queue depth
                                    session_seq=seq
                                )
                                
                        # Case B: Already in this zone — emit ZONE_DWELL every 30s interval
                        else:
                            DWELL_INTERVAL_MS = 30_000
                            time_in_zone_ms = int((curr_ts - curr_zone_info["enter_time"]).total_seconds() * 1000)
                            last_emit_ms = curr_zone_info.get("last_dwell_emit_ms", 0)
                            intervals_elapsed = (time_in_zone_ms - last_emit_ms) // DWELL_INTERVAL_MS
                            
                            if intervals_elapsed >= 1:
                                for i in range(int(intervals_elapsed)):
                                    interval_dwell_ms = last_emit_ms + (i + 1) * DWELL_INTERVAL_MS
                                    session_sequences[visitor_id] += 1
                                    seq = session_sequences[visitor_id]
                                    await emitter.emit(
                                        store_id=store_id,
                                        camera_id=camera_id,
                                        visitor_id=visitor_id,
                                        event_type="ZONE_DWELL",
                                        timestamp=curr_ts,
                                        zone_id=zone_id,
                                        dwell_ms=interval_dwell_ms,
                                        is_staff=is_staff,
                                        confidence=conf,
                                        session_seq=seq
                                    )
                                visitor_zones[visitor_id]["last_dwell_emit_ms"] = last_emit_ms + int(intervals_elapsed) * DWELL_INTERVAL_MS
                                visitor_zones[visitor_id]["dwell_event_count"] = curr_zone_info.get("dwell_event_count", 0) + int(intervals_elapsed)
                    else:
                        # Centroid is unzoned, check if exited previous zone
                        if curr_zone_info:
                            old_zone = curr_zone_info["zone_id"]
                            dwell_ms = int((curr_ts - curr_zone_info["enter_time"]).total_seconds() * 1000)
                            visitor_zones.pop(visitor_id)
                            await emitter.emit(
                                store_id=store_id,
                                camera_id=camera_id,
                                visitor_id=visitor_id,
                                event_type="ZONE_EXIT",
                                timestamp=curr_ts,
                                zone_id=old_zone,
                                dwell_ms=dwell_ms,
                                is_staff=is_staff,
                                confidence=conf,
                                session_seq=seq
                            )

                # Evaluate grace sessions & exits
                exit_events = session_manager.update_grace_sessions(curr_ts)
                for ev in exit_events:
                    seq = session_sequences.get(ev["visitor_id"], 0)
                    await emitter.emit(
                        store_id=store_id,
                        camera_id=camera_id,
                        visitor_id=ev["visitor_id"],
                        event_type="EXIT",
                        timestamp=ev["timestamp"],
                        dwell_ms=ev["dwell_ms"],
                        is_staff=ev["is_staff"],
                        confidence=ev["confidence"],
                        session_seq=seq
                    )
                    
                spatial_registry.prune_old_buckets(curr_ts)
                frames_batch = []
        else:
            # Real mode: pace by sleeping to match dynamic wall-clock time
            expected_wall_time = start_wall_time + (frame_idx / fps)
            sleep_dur = expected_wall_time - time_module.time()
            if sleep_dur > 0:
                await asyncio.sleep(sleep_dur)
            else:
                await asyncio.sleep(0.001)

            # Trigger inference if it is time and previous inference is done
            now_wall = time_module.time()
            if now_wall - last_infer_wall_time >= 1.0 and not inference_in_progress:
                last_infer_wall_time = now_wall
                curr_ts = datetime.now(timezone.utc)
                asyncio.create_task(run_inference_bg(frame.copy(), curr_ts))

            # Always push the annotated frame to the stream (at STREAM_INTERVAL speed)
            if api_url:
                now_t = time_module.time()
                if now_t - last_stream_time >= STREAM_INTERVAL:
                    last_stream_time = now_t

                    # Make annotations directly on frame (since frame is not reused for future inferences)
                    # Draw camera zones polygons
                    for z_name, z_conf in camera_zones.items():
                        poly = z_conf.get("polygon", [])
                        if poly:
                            pts = np.array([[int(p[0] * frame_w), int(p[1] * frame_h)] for p in poly], np.int32)
                            pts = pts.reshape((-1, 1, 2))
                            cv2.polylines(frame, [pts], True, (120, 120, 120), 2)
                            cx_z = int(np.mean([p[0] * frame_w for p in poly]))
                            cy_z = int(np.mean([p[1] * frame_h for p in poly]))
                            cv2.putText(frame, z_name, (cx_z - 30, cy_z), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 2)

                    # Draw tracks
                    for obj in tracked_objs:
                        track_id = obj["track_id"]
                        box = obj["box"]
                        x1, y1, x2, y2 = [int(coord) for coord in box]
                        
                        # Fetch visitor details
                        visitor_id = session_manager.track_to_visitor.get(track_id, f"TRK_{track_id}")
                        session_data = session_manager.active_sessions.get(visitor_id)
                        is_staff = session_data.get("is_staff", False) if session_data else False
                        is_reentry = session_data.get("reentry", False) if session_data else False
                        
                        # Determine zone
                        cx = int((box[0] + box[2]) / 2.0)
                        cy = int((box[1] + box[3]) / 2.0)
                        zone_id, _ = get_zone_for_centroid(cx, cy, frame_w, frame_h, camera_zones)

                        # Select color: BGR format
                        if is_staff:
                            color = (0, 215, 255) # Gold/Yellow for Staff
                            label = f"STAFF: {visitor_id}"
                        elif is_reentry:
                            color = (255, 0, 255) # Magenta for Reentry
                            label = f"REENTRY: {visitor_id}"
                        elif zone_id == "BILLING":
                            color = (0, 0, 255) # Red for Billing
                            label = f"BILLING: {visitor_id}"
                        elif zone_id:
                            color = (0, 255, 0) # Green for Skincare/Moisturiser
                            label = f"{zone_id}: {visitor_id}"
                        else:
                            color = (0, 165, 255) # Orange for General Customer
                            label = f"VISITOR: {visitor_id}"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (x1, y1 - 22), (x1 + w, y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    _, jpeg_bytes = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    _ann_bytes = jpeg_bytes.tobytes()
                    
                    async def post_annotated_frame(fb=_ann_bytes):
                        try:
                            await http_client.post(
                                f"{api_url}/stores/{store_id}/cameras/{camera_id}/frame",
                                content=fb,
                                headers={"Content-Type": "image/jpeg"},
                                timeout=0.5
                            )
                        except Exception:
                            pass
                    asyncio.create_task(post_annotated_frame())

    # Final sweep at clip termination
    cap.release()
    await emitter.close()
    await http_client.aclose()
    logger.info("Pipeline clip processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True, help="Path to CCTV clip")
    parser.add_argument("--store-id", required=True, help="Store identifier")
    parser.add_argument("--layout", required=True, help="Path to store_layout.json")
    parser.add_argument("--output", required=True, help="Path to output events.jsonl")
    parser.add_argument("--redis-url", default=None, help="Redis server URL")
    parser.add_argument("--real", action="store_true", help="Enable real ML inference instead of simulation")
    parser.add_argument("--frames", type=int, default=0, help="Limit number of frames to process (0 for all)")
    parser.add_argument("--api-url", default=None, help="API server URL to stream events directly")
    parser.add_argument("--camera-id", default=None, help="Override camera_id (e.g. CAM_BILLING_01). If omitted, inferred from filename.")
    parser.add_argument("--loop", action="store_true", help="Loop video clip indefinitely")
    parser.add_argument("--conf-threshold", type=float, default=0.50, help="Confidence threshold for detections")
    args = parser.parse_args()
    
    asyncio.run(process_clip(
        clip_path=args.clip,
        store_id=args.store_id,
        layout_path=args.layout,
        output_path=args.output,
        redis_url=args.redis_url or os.getenv("REDIS_URL"),
        real=args.real,
        max_frames=args.frames,
        api_url=args.api_url,
        loop=args.loop,
        camera_id_override=args.camera_id,
        conf_threshold=args.conf_threshold,
    ))
