import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
import logging
from pathlib import Path
import sys

# Path setup
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import VIDEO_RAW_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(message)s')

# --- CONFIGURATION ---
RADAR_WIDTH = 300
RADAR_HEIGHT = 200
RADAR_COLOR = (50, 50, 50)
COLOR_TOLERANCE = 60  # Higher = Stricter (Ref gets ignored). Lower = Looser.

def get_player_color(image, box):
    """Extracts dominant shirt color."""
    x1, y1, x2, y2 = map(int, box)
    h, w = y2 - y1, x2 - x1
    # Tight upper-body crop to avoid grass/shorts
    crop = image[int(y1+h*0.15):int(y1+h*0.5), int(x1+w*0.25):int(x1+w*0.75)]
    if crop.size == 0: return np.array([0, 0, 0])
    return np.mean(crop, axis=(0, 1))

def calibrate_team_colors(model, video_path):
    """Auto-detects team colors from the first frame."""
    logging.info("🎨 Calibrating Team Colors...")
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    
    if not ret: return (255,0,0), (0,0,255)

    # Use High-Res for calibration to catch everyone
    results = model.predict(frame, classes=[0], conf=0.15, imgsz=1280, verbose=False)[0]
    colors = []
    
    for box in results.boxes:
        colors.append(get_player_color(frame, box.xyxy[0].cpu().numpy()))
    
    if len(colors) < 2: return (255,0,0), (0,0,255)

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(colors)
    return kmeans.cluster_centers_[0], kmeans.cluster_centers_[1]

def draw_radar(frame, home_pos, away_pos, ball_pos, home_color, away_color):
    """Draws a 2D Minimap."""
    h, w, _ = frame.shape
    radar = np.full((RADAR_HEIGHT, RADAR_WIDTH, 3), RADAR_COLOR, dtype=np.uint8)
    cv2.rectangle(radar, (0,0), (RADAR_WIDTH, RADAR_HEIGHT), (255,255,255), 2)
    cv2.line(radar, (RADAR_WIDTH//2, 0), (RADAR_WIDTH//2, RADAR_HEIGHT), (150,150,150), 1)
    
    def map_coords(pos_list):
        mapped = []
        for x, y in pos_list:
            rx = int((x / w) * RADAR_WIDTH)
            ry = int((y / h) * RADAR_HEIGHT)
            mapped.append((rx, ry))
        return mapped

    home_map = map_coords(home_pos)
    away_map = map_coords(away_pos)
    
    for x, y in home_map:
        cv2.circle(radar, (x, y), 3, tuple(map(int, home_color)), -1)
    for x, y in away_map:
        cv2.circle(radar, (x, y), 3, tuple(map(int, away_color)), -1)
        
    if ball_pos:
        bx, by = map_coords([ball_pos])[0]
        cv2.circle(radar, (bx, by), 4, (255, 255, 255), -1)

    frame[h-RADAR_HEIGHT-20 : h-20, w-RADAR_WIDTH-20 : w-20] = radar
    return frame

def process_advanced_video(video_name="sample_match.mp4"):
    input_path = VIDEO_RAW_DIR / video_name
    output_path = PROCESSED_DATA_DIR / "output_video.mp4"
    
    if not input_path.exists():
        logging.error(f"❌ Video missing: {input_path}")
        return

    # Use Medium model
    model = YOLO('yolov8m.pt') 
    color_1, color_2 = calibrate_team_colors(model, input_path)
    
    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    logging.info("🚀 Starting Pro Analysis (Ref Filter + High-Res)...")
    
    frame_count = 0
    possession_stats = {"Team 1": 0, "Team 2": 0}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        # 1. TRACK PLAYERS: imgsz=1280 IS CRITICAL FOR STRIKERS
        player_results = model.track(
            frame, 
            persist=True, 
            tracker="soccer_tracker.yaml",
            classes=[0], 
            conf=0.15,     # Low threshold to catch distant players
            imgsz=1280,    # High Resolution to "see" far players
            verbose=False
        )[0]
        
        # 2. DETECT BALL
        ball_results = model.predict(frame, classes=[32], conf=0.15, imgsz=1280, verbose=False)[0]
        
        ball_pos = None
        if len(ball_results.boxes) > 0:
            best_ball = sorted(ball_results.boxes, key=lambda x: x.conf[0], reverse=True)[0]
            bx = (best_ball.xyxy[0][0] + best_ball.xyxy[0][2]) / 2
            by = (best_ball.xyxy[0][1] + best_ball.xyxy[0][3]) / 2
            ball_pos = (float(bx), float(by))
            cv2.circle(frame, (int(bx), int(by)), 8, (0, 255, 255), 2)

        home_pos_list = []
        away_pos_list = []
        current_possessor_team = None
        closest_dist = float('inf')

        if player_results.boxes.id is not None:
            boxes = player_results.boxes.xyxy.cpu().numpy()
            ids = player_results.boxes.id.cpu().numpy()
            
            for box, track_id in zip(boxes, ids):
                p_color = get_player_color(frame, box)
                dist1 = np.linalg.norm(p_color - color_1)
                dist2 = np.linalg.norm(p_color - color_2)
                
                # --- REFEREE FILTER ---
                # If color is far from BOTH teams, it's a Ref/GK
                if min(dist1, dist2) > COLOR_TOLERANCE:
                    # Draw as "Other" (Gray) and SKIP possession logic
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), 2)
                    # Skip the rest of the loop for this person
                    continue 

                x1, y1, x2, y2 = map(int, box)
                center_x, center_y = (x1+x2)/2, (y1+y2)/2
                
                if dist1 < dist2:
                    team = "Home"; color = color_1; home_pos_list.append((center_x, center_y))
                else:
                    team = "Away"; color = color_2; away_pos_list.append((center_x, center_y))
                
                # Draw Team Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), tuple(map(int, color)), 2)
                cv2.putText(frame, f"{int(track_id)}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                # Possession
                if ball_pos:
                    dist_to_ball = np.linalg.norm(np.array([center_x, center_y]) - np.array(ball_pos))
                    if dist_to_ball < closest_dist:
                        closest_dist = dist_to_ball
                        if dist_to_ball < 70:
                            current_possessor_team = team

        # Stats Update
        if current_possessor_team == "Home": possession_stats["Team 1"] += 1
        elif current_possessor_team == "Away": possession_stats["Team 2"] += 1
            
        total = sum(possession_stats.values())
        if total == 0: total = 1
        t1_pct = int((possession_stats["Team 1"] / total) * 100)
        t2_pct = 100 - t1_pct

        # Draw Dashboard
        cv2.rectangle(frame, (0, 0), (width, 50), (0,0,0), -1)
        cv2.putText(frame, f"HOME: {t1_pct}%", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"AWAY: {t2_pct}%", (width-250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        bar_w = 400
        bar_start = (width // 2) - (bar_w // 2)
        cv2.rectangle(frame, (bar_start, 15), (bar_start + int(bar_w * (t1_pct/100)), 35), tuple(map(int, color_1)), -1)
        cv2.rectangle(frame, (bar_start + int(bar_w * (t1_pct/100)), 15), (bar_start + bar_w, 35), tuple(map(int, color_2)), -1)

        # Draw Radar
        frame = draw_radar(frame, home_pos_list, away_pos_list, ball_pos, color_1, color_2)

        out.write(frame)
        if frame_count % 30 == 0: print(f"Processed {frame_count} frames...", end='\r')

    cap.release()
    out.release()
    logging.info(f"\n✅ Processing complete! Saved to {output_path}")

if __name__ == "__main__":
    process_advanced_video()