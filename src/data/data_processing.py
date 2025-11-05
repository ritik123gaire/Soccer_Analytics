import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm

# Import the paths from our main config file
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_match_file(file_path: Path) -> dict:
    """
    Processes a single raw match JSONL file and extracts key match-level stats.
    
    Args:
        file_path: The Path object to the raw JSONL file.
        
    Returns:
        A dictionary containing the processed data for one match.
    """
    try:
        # Load the entire event stream for one match into a DataFrame
        events_df = pd.read_json(file_path, lines=True)
        
        if events_df.empty:
            logging.warning(f"Empty file: {file_path.name}. Skipping.")
            return None

        # --- 1. Get Match Metadata ---
        match_id = int(file_path.stem)
        
        # Get team names from the first "Starting XI" event
        # We find the first event, get its 'team' field, and assign as home
        # We find the last event of this type, get its 'team', and assign as away
        starting_xi_events = events_df[events_df['type'] == 'Starting XI']
        if len(starting_xi_events) < 2:
            logging.warning(f"Not enough Starting XI events in {match_id}. Skipping.")
            return None
            
        home_team = starting_xi_events.iloc[0]['team']
        away_team = starting_xi_events.iloc[1]['team']
        
        # Get match date from the first event
        match_date = pd.to_datetime(events_df['timestamp'].iloc[0]).date()

        # --- 2. Extract Statistics ---
        
        # Get Shots and xG
        shots_df = events_df[events_df['type'] == 'Shot'].copy()
        
        # Handle cases where 'shot_statsbomb_xg' might be missing (though it shouldn't be for shots)
        if 'shot_statsbomb_xg' in shots_df.columns:
            shots_df['shot_statsbomb_xg'] = shots_df['shot_statsbomb_xg'].fillna(0)
            home_xg = shots_df[shots_df['team'] == home_team]['shot_statsbomb_xg'].sum()
            away_xg = shots_df[shots_df['team'] == away_team]['shot_statsbomb_xg'].sum()
        else:
            home_xg = 0.0
            away_xg = 0.0

        # Get Final Score by counting "Goal" outcomes
        if 'shot_outcome' in shots_df.columns:
            home_goals = (shots_df[(shots_df['team'] == home_team) & (shots_df['shot_outcome'] == 'Goal')]).shape[0]
            away_goals = (shots_df[(shots_df['team'] == away_team) & (shots_df['shot_outcome'] == 'Goal')]).shape[0]
        else:
            home_goals = 0
            away_goals = 0

        # Get Pass Counts
        passes_df = events_df[events_df['type'] == 'Pass']
        home_passes = (passes_df['team'] == home_team).sum()
        away_passes = (passes_df['team'] == away_team).sum()

        # Get Possession (sum of event durations)
        # We filter for events that have a duration and are not "Half Start"
        possession_df = events_df[events_df['duration'].notna() & (events_df['type'] != 'Half Start')]
        
        home_possession_time = possession_df[possession_df['team'] == home_team]['duration'].sum()
        away_possession_time = possession_df[possession_df['team'] == away_team]['duration'].sum()
        total_possession_time = home_possession_time + away_possession_time
        
        # Avoid division by zero if a match has no possession data
        if total_possession_time > 0:
            home_possession_pct = home_possession_time / total_possession_time
            away_possession_pct = away_possession_time / total_possession_time
        else:
            home_possession_pct = 0.5
            away_possession_pct = 0.5
            
        # --- 3. Assemble Dictionary ---
        match_data = {
            'match_id': match_id,
            'match_date': match_date,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_goals,
            'away_score': away_goals,
            'home_xg': home_xg,
            'away_xg': away_xg,
            'home_passes': home_passes,
            'away_passes': away_passes,
            'home_possession': home_possession_pct,
            'away_possession': away_possession_pct
        }
        
        return match_data

    except Exception as e:
        logging.error(f"Error processing file {file_path.name}: {e}")
        return None

def main():
    """
    Main function to orchestrate the data processing pipeline.
    """
    logging.info("Starting raw data processing...")
    
    # Find all raw JSON files
    raw_files = list(RAW_DATA_DIR.glob("*.json"))
    if not raw_files:
        logging.error("No raw data files found. Did you run data_collection.py?")
        return

    logging.info(f"Found {len(raw_files)} match files to process.")
    
    all_matches_data = []
    
    # Loop through files with a progress bar
    for file in tqdm(raw_files, desc="Processing Matches"):
        match_data = process_match_file(file)
        if match_data:
            all_matches_data.append(match_data)
            
    if not all_matches_data:
        logging.error("No data was successfully processed.")
        return

    # Convert list of dictionaries to a DataFrame
    final_df = pd.DataFrame(all_matches_data)
    
    # Sort by date
    final_df = final_df.sort_values(by='match_date').reset_index(drop=True)
    
    # --- Save Processed Data ---
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "matches.csv"
    
    final_df.to_csv(output_path, index=False)
    
    logging.info(f"Successfully processed {len(final_df)} matches.")
    logging.info(f"Clean data saved to {output_path}")
    logging.info("--- Data processing complete. ---")

if __name__ == "__main__":
    main()