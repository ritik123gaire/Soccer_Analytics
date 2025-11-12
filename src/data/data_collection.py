import pandas as pd
from statsbombpy import sb
from pathlib import Path
import logging
import time

# Import the paths from our main config file
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# --- Configuration ---

# We will download all available La Liga seasons
SEASONS_TO_DOWNLOAD = [
    (11, 90),  # 2020/2021
    (11, 42),  # 2019/2020
    (11, 4),   # 2018/2019
    (11, 1),   # 2017/2018
    (11, 2),   # 2016/2017
    (11, 27),  # 2015/2016
  
]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_raw_data():
    """
    Downloads all match event data for the specified competition/season pairs
    and saves each match's events as a separate JSON file in data/raw/.
    """

    all_matches_metadata = []
    
    # Ensure the raw data directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting download for {len(SEASONS_TO_DOWNLOAD)} seasons...")
    
    # 1. Loop through each competition and season
    for comp_id, season_id in SEASONS_TO_DOWNLOAD:
        
        logging.info(f"--- Fetching matches for competition {comp_id}, season {season_id} ---")
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
            all_matches_metadata.append(matches)
        except Exception as e:
            logging.error(f"Could not fetch matches for {comp_id}/{season_id}. Error: {e}")
            continue  # Skip to the next season

        if matches.empty:
            logging.warning(f"No matches found for {comp_id}/{season_id}. Skipping.")
            continue

        match_ids = matches['match_id'].unique()
        logging.info(f"Found {len(match_ids)} matches for this season. Starting event data download...")

        # 2. Loop through each match ID and download its event data
        for i, match_id in enumerate(match_ids):
            file_path = RAW_DATA_DIR / f"{match_id}.json"
            
            # Check if file already exists to avoid re-downloading
            if file_path.exists():
                logging.info(f"({i+1}/{len(match_ids)}) Data for match {match_id} already exists. Skipping.")
                continue

            try:
                logging.info(f"({i+1}/{len(match_ids)}) Fetching events for match {match_id}...")
                events = sb.events(match_id=match_id)
                
                if events.empty:
                    logging.warning(f"No event data found for match {match_id}. Skipping.")
                    continue
                    
                events.to_json(file_path, orient='records', lines=True)
                logging.info(f"Successfully saved data for match {match_id}")

                # Add a small delay to be polite to the API
                time.sleep(1)

            except Exception as e:
                logging.error(f"Could not download or save data for match {match_id}. Error: {e}")

    logging.info("--- ALL DATA DOWNLOADS COMPLETE. ---")

    # --- ADD THIS BLOCK ---
    logging.info("Saving all match metadata...")
    all_matches_df = pd.concat(all_matches_metadata)

    # Ensure processed dir exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_matches_df.to_csv(PROCESSED_DATA_DIR / "all_matches_metadata.csv", index=False)
    logging.info(f"Saved metadata for {len(all_matches_df)} matches.")
    # --- END OF BLOCK ---

if __name__ == "__main__":
    download_raw_data()