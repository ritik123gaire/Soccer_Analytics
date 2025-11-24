import cv2
import numpy as np
from ultralytics import YOLO
import logging
import sys
from pathlib import Path

# Path setup
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import VIDEO_RAW_DIR, PROCESSED_DATA_DIR, HOME_TEAM_COLOR_BGR, AWAY_TEAM_COLOR_BGR

logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_dominant_color(image, box):
    """
    Extracts the average color from the center of the bounding box (shirt area).
    """
    x1, y1, x2, y2 = map(int, box)
    
    # Crop to the center-top half (shirt area)
    center_y = int(y1 + (y2 - y1) * 0.25) # Slightly higher to catch shoulders
    center_h = int((y2 - y1) * 0.35)
    center_x = int(x1 + (x2 - x1) * 0.3)
    center_w = int((x2 - x1) * 0.4)
    
    crop = image[center_y:center_y+center_h, center_x:center_x+center_w]
    
    if crop.size == 0:
        return np.array([0, 0, 0])

    avg_color_per_row = np.average(crop, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color

def classify_team(requested_color, home_color, away_color, threshold=60):
    """
    Determines team. Returns 'Other' if color is too different from both teams.
    Threshold: Higher number = more lenient, Lower number = stricter.
    """
    dist_home = np.linalg.norm(requested_color - np.array(home_color))
    dist_away = np.linalg.norm(requested_color - np.array(away_color))
    
    # 1. Check if it's too far from BOTH (Referees/Goalkeepers)
    if dist_home > threshold and dist_away > threshold:
        return "Other", (0, 255, 255) # Yellow for Other
    
    # 2. Assign to closest
    if dist_home < dist_away:
        return "Home", home_color
    else:
        return "Away", away_color

def detect_game_stats(video_name="sample_match.mp4"):
    input_path = VIDEO_RAW_DIR / video_name
    output_path = PROCESSED_DATA_DIR / "output_video.mp4"
    
    if not input_path.exists():
        logging.error(f"Video not found: {input_path}")
        return

    model = YOLO('yolov8n.pt') 
    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Video Writer
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not out.isOpened(): raise Exception
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    home_poss_frames = 0
    away_poss_frames = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        # 1. Detect
        results = model.predict(frame, classes=[0, 32], conf=0.3, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        
        ball_box = None
        players = []

        for box, cls in zip(boxes, classes):
            if int(cls) == 32: ball_box = box
            elif int(cls) == 0: players.append(box)

        # 2. Classify Teams (filtering out Refs)
        home_players = []
        away_players = []
        
        for p_box in players:
            p_color = get_dominant_color(frame, p_box)
            
            # --- NEW CLASSIFICATION LOGIC ---
            team_name, color_bgr = classify_team(p_color, HOME_TEAM_COLOR_BGR, AWAY_TEAM_COLOR_BGR, threshold=70)
            
            x1, y1, x2, y2 = map(int, p_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
            
            # Only add to list if they are actually Home or Away
            if team_name == "Home":
                home_players.append(p_box)
            elif team_name == "Away":
                away_players.append(p_box)
            # If "Other", we draw the box (Yellow) but DO NOT add them to player lists
            # This prevents Refs/GKs from counting towards possession.

        # 3. Possession Logic
        possession_status = "Neutral"
        
        if ball_box is not None:
            bx1, by1, bx2, by2 = map(int, ball_box)
            ball_center = np.array([(bx1+bx2)/2, (by1+by2)/2])
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            
            min_dist = float('inf')
            closest_team = None
            
            # Only check legit players (no refs)
            for p in home_players:
                px1, py1, px2, py2 = p
                p_center = np.array([(px1+px2)/2, (py1+py2)/2])
                dist = np.linalg.norm(ball_center - p_center)
                if dist < min_dist:
                    min_dist = dist
                    closest_team = "Home"
            
            for p in away_players:
                px1, py1, px2, py2 = p
                p_center = np.array([(px1+px2)/2, (py1+py2)/2])
                dist = np.linalg.norm(ball_center - p_center)
                if dist < min_dist:
                    min_dist = dist
                    closest_team = "Away"
            
            if min_dist < 100: 
                possession_status = closest_team
                if closest_team == "Home": home_poss_frames += 1
                if closest_team == "Away": away_poss_frames += 1

        # 4. Dashboard
        total_poss = home_poss_frames + away_poss_frames
        if total_poss > 0:
            home_pct = int((home_poss_frames / total_poss) * 100)
            away_pct = 100 - home_pct
        else:
            home_pct, away_pct = 50, 50

        # Draw Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 20), (450, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        cv2.rectangle(frame, (50, 50), (50 + home_pct*4, 80), HOME_TEAM_COLOR_BGR, -1)
        cv2.rectangle(frame, (50 + home_pct*4, 50), (450, 80), AWAY_TEAM_COLOR_BGR, -1)
        
        # Text color is white for readability
        cv2.putText(frame, f"Home: {home_pct}%", (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Away: {away_pct}%", (350, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Possession: {possession_status}", (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out.write(frame)
        cv2.imshow("Soccer Analytics Vision Pro", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logging.info(f"Processing complete! Saved to {output_path}")

if __name__ == "__main__":
    detect_game_stats()