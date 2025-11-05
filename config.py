from pathlib import Path

# Define the root directory of the project
# Path(__file__) is this config.py file
# .parent is the 'soccer-analytics-project' folder
ROOT_DIR = Path(__file__).parent

# Define paths for data
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Define path for models
MODELS_DIR = ROOT_DIR / "models"