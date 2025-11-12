import pandas as pd
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm

# Import the paths from our main config file
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_player_minutes(events_df, match_id):
    """
    Calculates the minutes played for each player in a match.
    This version is hardened to handle schema inconsistencies in old data.
    """
    player_minutes = {}
    
    # 1. Get Starters
    starting_xi_events = events_df[events_df['type'] == 'Starting XI']
    if starting_xi_events.empty:
        logging.warning(f"No Starting XI event found for match {match_id}")
        return pd.DataFrame() # Return empty DF
        
    for _, event in starting_xi_events.iterrows():
        team_id = event['team_id']
        team_name = event['team']
        
        # Check if tactics data is valid
        if not isinstance(event.get('tactics'), dict) or not isinstance(event.get('tactics', {}).get('lineup'), list):
            logging.warning(f"Malformed or missing tactics data in match {match_id}. Skipping team.")
            continue

        for player_data in event['tactics']['lineup']:
            try:
                # Check for modern dict-in-dict structure
                player = player_data.get('player')
                if isinstance(player, dict):
                    player_id = player.get('id')
                    player_name = player.get('name')
                else:
                    # Handle if player data is malformed
                    logging.warning(f"Skipping malformed player data (not dict) in match {match_id}: {player}")
                    continue
                
                if player_id is None or player_name is None:
                    logging.warning(f"Skipping player with missing ID or Name in match {match_id}")
                    continue

                player_minutes[player_id] = {
                    'player_name': player_name,
                    'team_id': team_id,
                    'team_name': team_name,
                    'minute_on': 0,
                    'minute_off': 90  # Default, will be updated
                }
            except (TypeError, KeyError, AttributeError) as e:
                # This will catch errors like "string indices must be integers"
                logging.warning(f"Skipping player with unexpected data structure in match {match_id}. Error: {e}. Data: {player_data}")
                continue

    # 2. Find max match minute
    half_end_events = events_df[events_df['type'] == 'Half End']
    if half_end_events.empty:
        logging.warning(f"No 'Half End' events found in match {match_id}. Defaulting to 90 min.")
        max_minute = 90
    else:
        # Use max() in case of extra time (period 4)
        max_minute = half_end_events['minute'].max() 

    # Update default 'minute_off' for all starters
    for player_id in player_minutes:
        player_minutes[player_id]['minute_off'] = max_minute

    # 3. Process Substitutions
    sub_events = events_df[events_df['type'] == 'Substitution']
    for _, sub in sub_events.iterrows():
        try:
            sub_minute = sub['minute']
            player_off_id = sub['player_id'] # Player coming OFF
            player_on = sub['substitution_replacement'] # Player coming ON
            
            # Player being subbed OFF
            if player_off_id in player_minutes:
                player_minutes[player_off_id]['minute_off'] = sub_minute
            
            # Player being subbed ON
            # Check if player_on is the dict we expect
            if not isinstance(player_on, dict):
                logging.warning(f"Skipping substitution with malformed 'replacement' data in match {match_id}: {player_on}")
                continue
                
            player_on_id = player_on.get('id')
            player_on_name = player_on.get('name')

            if player_on_id is None or player_on_name is None:
                logging.warning(f"Skipping substitution with missing ID/Name in match {match_id}")
                continue

            # Add the new player to our dictionary
            if player_on_id not in player_minutes:
                player_minutes[player_on_id] = {
                    'player_name': player_on_name,
                    'team_id': sub['team_id'],
                    'team_name': sub['team'],
                    'minute_on': sub_minute,
                    'minute_off': max_minute
                }
        except (TypeError, KeyError, AttributeError) as e:
            # This will catch the "string indices must be integers" error if it happens here
            logging.warning(f"Error processing substitution in match {match_id}. Error: {e}. Sub event: {sub['id']}")
            continue
        
    # 4. Calculate final minutes played
    final_minutes_list = []
    for player_id, data in player_minutes.items():
        minutes_played = data['minute_off'] - data['minute_on']
        # Handle cases of extra time or weird data
        if minutes_played < 0:
            minutes_played = 0
        data['player_id'] = player_id
        data['minutes_played'] = minutes_played
        final_minutes_list.append(data)
    
    if not final_minutes_list:
        logging.warning(f"No players processed for match {match_id}.")
        return pd.DataFrame() # Return empty DF

    return pd.DataFrame(final_minutes_list)


def process_match_for_players(file_path: Path, correct_match_date) -> pd.DataFrame:
    """
    Processes a single raw match JSONL file and extracts stats for *every* player.
    """
    try:
        match_id = int(file_path.stem)
        events_df = pd.read_json(file_path, lines=True)
        
        if events_df.empty:
            logging.warning(f"Empty file: {file_path.name}. Skipping.")
            return None

        # --- 1. Get Minutes Played for all players ---
        players_df = get_player_minutes(events_df, match_id)
        if players_df.empty:
            logging.warning(f"No player minutes data found for match {match_id}. Skipping.")
            return None
            
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
            
            # --- 3. Combine Stats with Player List ---
            final_players_df = players_df.merge(player_stats, on='player_id', how='left')
        else:
            # No shots in the match, just return the player list
            final_players_df = players_df
            final_players_df['total_shots'] = 0
            final_players_df['total_xg'] = 0
            final_players_df['goals_scored'] = 0

        # Fill NaNs with 0 for players who played but didn't shoot
        stats_cols = ['total_shots', 'total_xg', 'goals_scored']
        final_players_df[stats_cols] = final_players_df[stats_cols].fillna(0)
        
        # Add the CORRECT match date
        final_players_df['match_date'] = correct_match_date
        
        # Filter out players with 0 minutes (unused subs, etc.)
        final_players_df = final_players_df[final_players_df['minutes_played'] > 0].copy()
        
        return final_players_df

    except Exception as e:
        logging.error(f"Error processing file {file_path.name}: {e}")
        return None

def main():
    """
    Main function to orchestrate the player data processing pipeline.
    """
    logging.info("Starting player stats processing...")
    
    # --- 1. Load Match Metadata ---
    logging.info("Loading match metadata...")
    try:
        metadata_df = pd.read_csv(PROCESSED_DATA_DIR / "all_matches_metadata.csv")
        # Create a simple match_id -> match_date dictionary for fast lookup
        # Parse date string, then get just the date part
        match_dates = pd.to_datetime(metadata_df['match_date']).dt.date
        match_date_map = dict(zip(metadata_df['match_id'], match_dates))
        logging.info(f"Loaded {len(match_date_map)} match dates.")
    except FileNotFoundError:
        logging.error("all_matches_metadata.csv not found! Please run data_collection.py again.")
        return
    # --- END OF BLOCK ---
    
    # --- 2. Find Raw Files (TEST MODE: Only first 10) ---
    raw_files = list(RAW_DATA_DIR.glob("*.json"))
    if not raw_files:
        logging.error("No raw data files found. Did you run data_collection.py?")
        return

    logging.info(f"Found {len(raw_files)} match files to process (TEST MODE).")
    
    all_player_stats = []
    
    # --- 3. Loop Through Files ---
    for file in tqdm(raw_files, desc="Processing Player Stats"):
        
        # Find the correct date for this match
        match_id = int(file.stem) # Get match_id from filename
        correct_match_date = match_date_map.get(match_id)
        
        if correct_match_date is None:
            logging.warning(f"No date found in metadata for match {match_id}. Skipping.")
            continue
        
        # Process the file, passing in the correct date
        match_player_data = process_match_for_players(file, correct_match_date) 
        
        if match_player_data is not None and not match_player_data.empty:
            all_player_stats.append(match_player_data)
            
    if not all_player_stats:
        logging.error("No player data was successfully processed. All files may have failed.")
        return

    # --- 4. Save Final DataFrame ---
    final_df = pd.concat(all_player_stats)
    
    # Sort by date and player
    final_df = final_df.sort_values(by=['match_date', 'player_id']).reset_index(drop=True)
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "player_stats_per_match.csv"
    
    final_df.to_csv(output_path, index=False)
    
    logging.info(f"Successfully processed {len(final_df)} player-match records.")
    logging.info(f"Clean player data saved to {output_path}")
    logging.info("--- Player stats processing complete. ---")

if __name__ == "__main__":
    main()