import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Import the paths from our main config file
from config import PROCESSED_DATA_DIR

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
ROLLING_WINDOW_SIZE = 5

def main():
    """
    Main function to load player stats, engineer features, and save the final dataset.
    """
    logging.info("Starting player feature engineering...")
    
    # --- 1. Load Data ---
    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / "player_stats_per_match.csv", parse_dates=['match_date'])
        logging.info("Loaded 'player_stats_per_match.csv'.")
    except FileNotFoundError:
        logging.error("Could not find 'player_stats_per_match.csv'. Please run player_stats_processor.py first.")
        return

    # Sort by date and player to ensure rolling windows are causal
    df = df.sort_values(by=['player_id', 'match_date'])

    # --- 2. Create Target Variable ---
    # Our target is simple: did the player score in this match? (1=Yes, 0=No)
    df['scored_goal'] = (df['goals_scored'] > 0).astype(int)

    # --- 3. Create "Per 90 Minutes" Stats ---
    # This normalizes stats for players who were subbed on/off.
    # We avoid division by zero if minutes_played is 0 (though we filtered this).
    df['minutes_played'] = df['minutes_played'].replace(0, np.nan)
    df['xg_per_90'] = (df['total_xg'] / df['minutes_played']) * 90
    df['shots_per_90'] = (df['total_shots'] / df['minutes_played']) * 90
    
    # Fill any NaNs (from 0 minutes) with 0
    df[['xg_per_90', 'shots_per_90']] = df[['xg_per_90', 'shots_per_90']].fillna(0)

    # --- 4. Calculate Rolling Averages ---
    # We group by each player and calculate their rolling form.
    
    stats_to_roll = ['xg_per_90', 'shots_per_90', 'goals_scored', 'minutes_played']
    
    # .shift(1) is CRITICAL! It prevents data leakage.
    # It ensures we are only using a player's form from *before* the current match.
    rolling_stats_df = df.groupby('player_id')[stats_to_roll].rolling(
        window=ROLLING_WINDOW_SIZE, min_periods=1
    ).mean().shift(1)

    # *** THIS IS THE FIX ***
    # The groupby creates a MultiIndex (player_id, original_index).
    # We must drop the 'player_id' level to join back on the original index.
    rolling_stats_df = rolling_stats_df.reset_index(level='player_id', drop=True)
    # *** END FIX ***

    # Rename columns to reflect they are rolling features
    rolling_stats_df.columns = [f'{col}_roll{ROLLING_WINDOW_SIZE}' for col in stats_to_roll]

    # --- 5. Combine Features ---
    # This join will now work
    final_df = df.join(rolling_stats_df)
    
    # The first few games for each player will have NaN (no past data), so we drop them.
    final_df = final_df.dropna(subset=rolling_stats_df.columns)

    # --- 6. Save Final Features Dataset ---
    output_path = PROCESSED_DATA_DIR / "player_features.csv"
    
    # Select only the columns we need for modeling
    model_cols = [
        'match_id', 'player_id', 'player_name', 'team_name', 'match_date',
        'xg_per_90_roll5', 'shots_per_90_roll5', 'goals_scored_roll5', 'minutes_played_roll5',
        'scored_goal' # This is our target (y)
    ]
    
    final_df = final_df[model_cols]
    
    final_df.to_csv(output_path, index=False)
    
    logging.info(f"Successfully created {len(final_df.columns)} features for {len(final_df)} player-match records.")
    logging.info(f"Final player feature dataset saved to {output_path}")
    logging.info("--- Player feature engineering complete. ---")

if __name__ == "__main__":
    main()