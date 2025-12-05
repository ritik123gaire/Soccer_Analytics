from pathlib import Path

# Define the root directory of the project
ROOT_DIR = Path(__file__).parent

# Define paths for data
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VIDEO_RAW_DIR = DATA_DIR / "video_raw"

# Define path for models
MODELS_DIR = ROOT_DIR / "models"



# Team Colors (BGR Format)
# Real Madrid (White)
HOME_TEAM_COLOR_BGR = (255, 255, 255) 

# Barcelona (Dark Blue/Red) - Averages to a dark color
AWAY_TEAM_COLOR_BGR = (70, 40, 40)