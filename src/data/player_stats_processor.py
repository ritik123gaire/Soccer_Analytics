import pandas as pd
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import sys

# Path setup
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Import the paths from our main config file
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_player_minutes(events_df, match_id):
    """
    Calculates the minutes played for each player in a match.
    """
    player_minutes = {}
    
    # 1. Get Starters
    starting_xi_events = events_df[events_df['type'] == 'Starting XI']
    if starting_xi_events.empty:
        return pd.DataFrame() 
        
    for _, event in starting_xi_events.iterrows():
        team_id = event['team_id']
        team_name = event['team']
        
        # Validation checks
        if not isinstance(event.get('tactics'), dict): continue
        
        for player_data in event['tactics']['lineup']:
            try:
                player = player_data.get('player')
                player_id = player.get('id')
                player_name = player.get('name')
                
                if player_id and player_name:
                    player_minutes[player_id] = {
                        'player_name': player_name,
                        'team_id': team_id,
                        'team_name': team_name,
                        'minute_on': 0,
                        'minute_off': 90  # Default
                    }
            except:
                continue

    # 2. Find max match minute
    half_end_events = events_df[events_df['type'] == 'Half End']
    max_minute = 90 if half_end_events.empty else half_end_events['minute'].max()

    # Update default 'minute_off'
    for pid in player_minutes:
        player_minutes[pid]['minute_off'] = max_minute

    # 3. Process Substitutions
    sub_events = events_df[events_df['type'] == 'Substitution']
    for _, sub in sub_events.iterrows():
        try:
            sub_minute = sub['minute']
            player_off_id = sub['player_id']
            player_on = sub['substitution_replacement']
            
            # Sub OFF
            if player_off_id in player_minutes:
                player_minutes[player_off_id]['minute_off'] = sub_minute
            
            # Sub ON
            player_on_id = player_on.get('id')
            player_on_name = player_on.get('name')

            if player_on_id and player_on_id not in player_minutes:
                player_minutes[player_on_id] = {
                    'player_name': player_on_name,
                    'team_id': sub['team_id'],
                    'team_name': sub['team'],
                    'minute_on': sub_minute,
                    'minute_off': max_minute
                }
        except:
            continue
        
    # 4. Calculate final minutes
    final_minutes_list = []
    for player_id, data in player_minutes.items():
        minutes_played = data['minute_off'] - data['minute_on']
        if minutes_played < 0: minutes_played = 0
        
        data['player_id'] = player_id
        data['minutes_played'] = minutes_played
        final_minutes_list.append(data)
    
    return pd.DataFrame(final_minutes_list)


def process_match_for_players(file_path: Path, correct_match_date) -> pd.DataFrame:
    """
    Processes a single raw match JSONL file and extracts stats for every player.
    """
    try:
        match_id = int(file_path.stem)
        # Handle JSON decoding errors safely
        try:
            events_df = pd.read_json(file_path, lines=True) # Try JSONL
        except ValueError:
            events_df = pd.read_json(file_path) # Try standard JSON

        if events_df.empty: return None

        # --- 1. Get Minutes Played ---
        players_df = get_player_minutes(events_df, match_id)
        if players_df.empty: return None
            
        players_df['match_id'] = match_id

        # --- 2. Get Player Stats (Shots, xG, Goals) ---
        shots_df = events_df[events_df['type'] == 'Shot'].copy()
        
        if not shots_df.empty:
            # Aggregate stats per player
            player_stats = shots_df.groupby('player_id').agg(
                total_shots=('id', 'count'),
                total_xg=('shot_statsbomb_xg', 'sum'),
                goals_scored=('shot_outcome', lambda x: (x == 'Goal').sum())
            ).reset_index()
            
            # Merge
            final_players_df = players_df.merge(player_stats, on='player_id', how='left')
        else:
            final_players_df = players_df
            final_players_df['total_shots'] = 0
            final_players_df['total_xg'] = 0
            final_players_df['goals_scored'] = 0

        # Fill NaNs with 0
        stats_cols = ['total_shots', 'total_xg', 'goals_scored']
        final_players_df[stats_cols] = final_players_df[stats_cols].fillna(0)
        
        # Add Date
        final_players_df['match_date'] = correct_match_date
        
        # Filter out unused subs (0 minutes)
        final_players_df = final_players_df[final_players_df['minutes_played'] > 0].copy()
        
        return final_players_df

    except Exception as e:
        logging.error(f"Error processing file {file_path.name}: {e}")
        return None

def main():
    logging.info("Starting player stats processing...")
    
    # --- 1. Load Match Metadata (Improved Reliability) ---
    logging.info("Loading match dates...")
    
    # TRY STRATEGY A: Metadata file
    try:
        metadata_df = pd.read_csv(PROCESSED_DATA_DIR / "all_matches_metadata.csv")
        match_dates = pd.to_datetime(metadata_df['match_date']).dt.date
        match_date_map = dict(zip(metadata_df['match_id'], match_dates))
    except FileNotFoundError:
        # TRY STRATEGY B: matches.csv (Backup)
        try:
            logging.info("Metadata file missing. Falling back to matches.csv...")
            metadata_df = pd.read_csv(PROCESSED_DATA_DIR / "matches.csv")
            match_dates = pd.to_datetime(metadata_df['match_date']).dt.date
            match_date_map = dict(zip(metadata_df['match_id'], match_dates))
        except FileNotFoundError:
            logging.error("❌ Critical Error: Neither 'matches.csv' nor 'all_matches_metadata.csv' found.")
            return

    logging.info(f"Loaded {len(match_date_map)} match dates.")
    
    # --- 2. Find Raw Files ---
    raw_files = list(RAW_DATA_DIR.glob("*.json"))
    if not raw_files:
        logging.error("❌ No raw data files found in data/raw/")
        return

    logging.info(f"Found {len(raw_files)} raw match files.")
    
    all_player_stats = []
    
    # --- 3. Loop Through Files ---
    # We limit to 500 for speed, but you can remove [:500] to run all
    for file in tqdm(raw_files, desc="Processing Player Stats"):
        
        match_id = int(file.stem)
        correct_match_date = match_date_map.get(match_id)
        
        if correct_match_date is None:
            continue
        
        match_player_data = process_match_for_players(file, correct_match_date) 
        
        if match_player_data is not None and not match_player_data.empty:
            all_player_stats.append(match_player_data)
            
    if not all_player_stats:
        logging.error("No player data processed.")
        return

    # --- 4. Save Final DataFrame ---
    final_df = pd.concat(all_player_stats)
    final_df = final_df.sort_values(by=['match_date', 'player_id']).reset_index(drop=True)
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "player_stats_per_match.csv"
    
    final_df.to_csv(output_path, index=False)
    
    logging.info(f"✅ Success! Processed {len(final_df)} player records.")
    logging.info(f"Saved to {output_path}")

if __name__ == "__main__":
    main()