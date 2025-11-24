import cv2
from ultralytics import YOLO
from pathlib import Path
import logging
import sys

# Import paths from our config file
# We use sys.path hack to ensure we can import config if running as a script
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import VIDEO_RAW_DIR, PROCESSED_DATA_DIR

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_players_and_ball(video_name="sample_match.mp4"):
    """
    Runs YOLOv8 object detection on a video file.
    Detects Players (Class 0) and Ball (Class 32).
    Saves as .mp4 with H.264 codec for browser compatibility.
    """
    
    # 1. Setup Paths
    input_path = VIDEO_RAW_DIR / video_name
    output_path = PROCESSED_DATA_DIR / "output_video.mp4"
    
    if not input_path.exists():
        logging.error(f"Video file not found at: {input_path}")
        logging.error("Please place 'sample_match.mp4' in your 'data/video_raw' folder.")
        return

    # 2. Load YOLOv8 Model
    # 'yolov8n.pt' is the "Nano" model (fastest). Use 'yolov8m.pt' for better accuracy.
    logging.info("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt') 

    # 3. Open Video
    cap = cv2.VideoCapture(str(input_path))
    
    if not cap.isOpened():
        logging.error(f"Could not open video file: {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # --- CRITICAL UPDATE: CODEC FOR MP4 ---
    # We try 'avc1' (H.264) first as it is best for browsers.
    # If that fails, we fall back to 'mp4v'.
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not out.isOpened():
            raise Exception("avc1 codec failed")
        logging.info("Initialized VideoWriter with 'avc1' codec (Browser Friendly).")
    except Exception as e:
        logging.warning(f"Could not use 'avc1' codec ({e}). Falling back to 'mp4v'...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    logging.info(f"Processing video: {width}x{height} @ {fps} FPS")
    logging.info("Press 'q' in the popup window to stop early...")

    frame_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processing frame {frame_count}...", end='\r')

        # 4. Run Detection
        # classes=[0, 32] filters for Person (0) and Sports Ball (32)
        # conf=0.3 means we only keep detections with >30% confidence
        results = model.predict(frame, classes=[0, 32], conf=0.3, verbose=False)
        
        # 5. Draw Bounding Boxes
        # .plot() automatically draws the boxes and labels on the frame
        annotated_frame = results[0].plot()
        
        # 6. Write Frame to Output
        out.write(annotated_frame)
        
        # Show preview window
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        # Allow quitting by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    logging.info(f"\nProcessing complete! Video saved to: {output_path}")

if __name__ == "__main__":
    detect_players_and_ball()