# pipeline/benchmark.py
import cv2, time, json, argparse, statistics
from pathlib import Path
from ultralytics import YOLO
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
import torch

def extract_frames(clip_path, n=60):
    cap = cv2.VideoCapture(clip_path)
    frames, count = [], 0
    while cap.isOpened() and count < n:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame); count += 1
    cap.release()
    return frames

def bench_yolov9(frames, model_name="yolov9s.pt"):
    model = YOLO(model_name)
    latencies, person_counts = [], []
    for frame in frames:
        t0 = time.perf_counter()
        results = model(frame, classes=[0], verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)
        person_counts.append(len(results[0].boxes))
    return latencies, person_counts

def bench_rtdetr(frames):
    proc = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
    model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")
    model.eval()
    latencies, person_counts = [], []
    with torch.no_grad():
        for frame in frames:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = proc(images=img_rgb, return_tensors="pt")
            t0 = time.perf_counter()
            outputs = model(**inputs)
            latencies.append((time.perf_counter() - t0) * 1000)
            results = proc.post_process_object_detection(
                outputs,
                target_sizes=[(frame.shape[0], frame.shape[1])],
                threshold=0.35
            )
            persons = [l for l in results[0]["labels"].tolist() if l == 0]
            person_counts.append(len(persons))
    return latencies, person_counts

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    p.add_argument("--frames", type=int, default=60)
    args = p.parse_args()

    frames = extract_frames(args.clip, args.frames)
    print(f"Extracted {len(frames)} frames. Running YOLOv9s...")
    y_lat, y_cnt = bench_yolov9(frames)
    print("Running RT-DETR...")
    r_lat, r_cnt = bench_rtdetr(frames)

    results = {
        "frames_tested": len(frames),
        "yolov9s": {
            "mean_latency_ms":            round(statistics.mean(y_lat), 1),
            "p95_latency_ms":             round(sorted(y_lat)[int(.95 * len(y_lat))], 1),
            "total_persons_detected":     sum(y_cnt),
            "frames_with_zero_detections": y_cnt.count(0)
        },
        "rtdetr_r50vd": {
            "mean_latency_ms":            round(statistics.mean(r_lat), 1),
            "p95_latency_ms":             round(sorted(r_lat)[int(.95 * len(r_lat))], 1),
            "total_persons_detected":     sum(r_cnt),
            "frames_with_zero_detections": r_cnt.count(0)
        }
    }

    out = Path("benchmark_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
