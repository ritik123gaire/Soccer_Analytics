import cv2
import sys
from pathlib import Path
import logging

# Path setup to find config.py
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import VIDEO_RAW_DIR, DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(message)s')

def extract_frames_for_annotation(video_name="sample_match.mp4", num_frames=20):
    """
    Extracts 'num_frames' evenly spaced from the video to create a dataset 
    for manual annotation (Homework 3).
    """
    input_path = VIDEO_RAW_DIR / video_name
    
    # Create a new directory for the images
    images_dir = DATA_DIR / "images_for_annotation"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        logging.error(f"Video not found at: {input_path}")
        return

    cap = cv2.VideoCapture(str(input_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate interval to get exactly 'num_frames' spread across the video
    interval = total_frames // num_frames
    
    count = 0
    saved_count = 0
    
    logging.info(f"Extracting {num_frames} frames from {video_name}...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Save frame if it matches the interval
        if count % interval == 0 and saved_count < num_frames:
            # Filename: match_frame_000.jpg, match_frame_001.jpg, etc.
            filename = f"match_frame_{saved_count:03d}.jpg"
            save_path = images_dir / filename
            
            cv2.imwrite(str(save_path), frame)
            logging.info(f"Saved {save_path}")
            saved_count += 1
            
        count += 1

    cap.release()
    logging.info(f"Done! {saved_count} images saved to {images_dir}")

if __name__ == "__main__":
    extract_frames_for_annotation()