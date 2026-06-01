import os
import sys
import time
import subprocess

# Helper to load .env file manually without external dependencies
def load_env(env_path=".env"):
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip()
                        # Strip optional surrounding quotes
                        if val.startswith('"') and val.endswith('"'):
                            val = val[1:-1]
                        elif val.startswith("'") and val.endswith("'"):
                            val = val[1:-1]
                        os.environ[key] = val

def main():
    # Set the working directory to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Load environment variables
    load_env()
    
    db_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/retail_intelligence")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    clips_dir = os.getenv("CCTV_CLIPS_DIR", "CCTV Footage")
    api_url = os.getenv("API_URL", "http://127.0.0.1:8000")
    
    print("=" * 60)
    print("🔮 PurplleNerve: Local Orchestrator Service")
    print("=" * 60)
    print(f"PostgreSQL DB URL:  {db_url}")
    print(f"Redis Cache Broker:  {redis_url}")
    print(f"Local CCTV Clips:    {clips_dir}")
    print(f"Central Ingestion:   {api_url}")
    print("=" * 60)
    
    if not os.path.exists(clips_dir):
        print(f"ERROR: Local CCTV clips folder '{clips_dir}' does not exist.")
        print("Please configure CCTV_CLIPS_DIR correctly in your local .env file.")
        sys.exit(1)
        
    processes = []
    
    # Clean up output logs folder first
    os.makedirs("output", exist_ok=True)
    
    # 1. Start Uvicorn API Server
    print("\n[+] Starting FastAPI Uvicorn Server on 127.0.0.1:8000...")
    env = os.environ.copy()
    env["DATABASE_URL"] = db_url
    env["REDIS_URL"] = redis_url
    
    uvicorn_cmd = [
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", "127.0.0.1",
        "--port", "8000"
    ]
    
    p_uv = subprocess.Popen(uvicorn_cmd, env=env)
    processes.append(p_uv)
    
    # Give uvicorn some time to boot up and initialize database tables
    time.sleep(4)
    
    # 2. Start the 5 Camera Pipelines sequentially
    cams = [
        {"name": "CAM 1 (Entry)", "cam_id": "CAM_ENTRY_01", "file": "CAM 1.mp4", "out": "output/events_blr_cam1.jsonl"},
        {"name": "CAM 2 (Floor)", "cam_id": "CAM_FLOOR_01", "file": "CAM 2.mp4", "out": "output/events_blr_cam2.jsonl"},
        {"name": "CAM 3 (Billing - RT-DETR)", "cam_id": "CAM_BILLING_01", "file": "CAM 3.mp4", "out": "output/events_blr_cam3.jsonl"},
        {"name": "CAM 4 (Floor)", "cam_id": "CAM_FLOOR_02", "file": "CAM 4.mp4", "out": "output/events_blr_cam4.jsonl"},
        {"name": "CAM 5 (Billing - RT-DETR)", "cam_id": "CAM_BILLING_02", "file": "CAM 5.mp4", "out": "output/events_blr_cam5.jsonl"},
    ]
    
    for cam in cams:
        clip_path = os.path.join(clips_dir, cam["file"])
        if not os.path.exists(clip_path):
            print(f"[-] Warning: Clip for {cam['name']} not found at '{clip_path}'. Skipping.")
            continue
            
        print(f"[+] Launching edge pipeline for {cam['name']}...")
        pipeline_cmd = [
            sys.executable, "-m", "pipeline.detect",
            "--clip", clip_path,
            "--store-id", "STORE_BLR_002",
            "--layout", "data/store_layout.json",
            "--output", cam["out"],
            "--api-url", api_url,
            "--real",
            "--loop",
            "--conf-threshold", "0.50"
        ]
        
        # Save output logs to a log file inside output/ to prevent stdout congestion
        log_file = open(f"output/{cam['cam_id'].lower()}.log", "w", encoding="utf-8")
        p_cam = subprocess.Popen(pipeline_cmd, stdout=log_file, stderr=log_file, env=env)
        processes.append(p_cam)
        
        # Stagger loading to prevent GPU/CPU initialization contention
        time.sleep(2)
        
    print("\n" + "=" * 60)
    print("🚀 All services and camera pipelines are active!")
    print("   Dashboard: http://127.0.0.1:8000/dashboard")
    print("   Press Ctrl+C to terminate all processes cleanly.")
    print("=" * 60 + "\n")
    
    try:
        while True:
            # Check if Uvicorn has died
            if p_uv.poll() is not None:
                print("[-] Uvicorn server has exited unexpectedly. Terminating orchestrator.")
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[+] Ctrl+C detected. Terminating all processes...")
    finally:
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=2.0)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        print("[+] All processes stopped successfully. Goodbye!")

if __name__ == "__main__":
    main()
