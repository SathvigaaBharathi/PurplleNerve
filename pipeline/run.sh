#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

CLIPS_DIR=${1:-"/clips"}
LAYOUT_FILE=${2:-"/data/store_layout.json"}
OUTPUT_FILE=${3:-"/output/events.jsonl"}

echo "Running detection pipeline on all clips in $CLIPS_DIR using layout $LAYOUT_FILE"
echo "Results will be written to $OUTPUT_FILE"

# Find all MP4 files in clips directory
for clip in "$CLIPS_DIR"/*.mp4; do
  if [ -e "$clip" ]; then
    filename=$(basename "$clip")
    echo "Processing clip: $filename"
    
    # We assign default store ID since we are processing Apex Retail
    store_id="STORE_BLR_002"
    
    # Invoke detect.py script
    python pipeline/detect.py \
      --clip "$clip" \
      --store-id "$store_id" \
      --layout "$LAYOUT_FILE" \
      --output "$OUTPUT_FILE"
  fi
done

echo "Pipeline processing complete. All clips processed."
