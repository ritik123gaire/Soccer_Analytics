import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Import the paths from our main config file
from config import PROCESSED_DATA_DIR

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
ROLLING_WINDOW_SIZE = 5

def create_target_variable(df):
    """Creates the target variable 'match_result' (0=Draw, 1=Home Win, 2=Away Win)."""
    
    conditions = [
        (df['home_score'] > df['away_score']), # Home Win
        (df['home_score'] < df['away_score']), # Away Win
    ]
    choices = [1, 2] # 1 = Home Win, 2 = Away Win
    
    # Default is 0 (Draw)
    df['match_result'] = np.select(conditions, choices, default=0)
    return df

def get_team_rolling_stats(df, team_col, stats_cols, window_size):
    """
    Calculates rolling average statistics for a team.
    
    Args:
        df: The main DataFrame.
        team_col: The column name for the team ('home_team' or 'away_team').
        stats_cols: A dictionary mapping original stats to new feature names 
                    (e.g., {'home_xg': 'xg_for', 'away_xg': 'xg_against'}).
        window_size: The number of games to average over.
        
    Returns:
        A DataFrame with rolling stats calculated for each team.
    """
    team_stats = []
    
    # Iterate over each unique team
    for team in df[team_col].unique():
        # Get all matches for this team (both home and away)
        team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values(by='match_date')
        
        # Create 'team-centric' columns (e.g., xg_for, xg_against)
        team_centric_df = pd.DataFrame()
        
        # When team is home
        home_matches = team_matches[team_matches['home_team'] == team]
        for orig_col, new_col in stats_cols.items():
            team_centric_df.loc[home_matches.index, new_col] = home_matches[orig_col]
            
        # When team is away
        away_matches = team_matches[team_matches['away_team'] == team]
        for orig_col, new_col in stats_cols.items():
            # Flip the logic for away matches
            # e.g., 'home_xg' (orig) becomes 'xg_against' (new)
            # e.g., 'away_xg' (orig) becomes 'xg_for' (new)
            if 'home' in orig_col:
                flipped_new_col = new_col.replace('_for', '_against') if '_for' in new_col else new_col.replace('_against', '_for')
                team_centric_df.loc[away_matches.index, flipped_new_col] = away_matches[orig_col]
            elif 'away' in orig_col:
                flipped_new_col = new_col.replace('_for', '_against') if '_for' in new_col else new_col.replace('_against', 'for')
                team_centric_df.loc[away_matches.index, flipped_new_col] = away_matches[orig_col]

        # Calculate rolling averages
        # .shift(1) is CRITICAL. It ensures we only use data from *past* games.
        rolling_stats = team_centric_df.rolling(window=window_size, min_periods=1).mean().shift(1)
        
        # Rename columns to reflect they are rolling stats
        rolling_stats = rolling_stats.add_suffix(f'_roll_{window_size}')
        
        # Combine with original match_id for joining
        rolling_stats = pd.concat([team_matches['match_id'], rolling_stats], axis=1)
        
        team_stats.append(rolling_stats)
        
    # Combine all team dataframes
    all_team_stats = pd.concat(team_stats)
    
    # Create columns to join back to the main df (e.g., 'home_xg_for_roll_5')
    final_stats_df = pd.DataFrame(index=df.index)
    
    # Merge home team stats
    home_join = df.merge(
        all_team_stats, 
        left_on=['match_id', 'home_team'], 
        right_on=['match_id', all_team_stats.index.isin(df[df['home_team'] == df[team_col]].index)] # This is a bit complex, but works
    ).set_index(df.index)
    
    # Merge away team stats
    away_join = df.merge(
        all_team_stats, 
        left_on=['match_id', 'away_team'], 
        right_on=['match_id', all_team_stats.index.isin(df[df['away_team'] == df[team_col]].index)]
    ).set_index(df.index)

    # A better way to merge the stats back
    final_df = df.copy()
    
    # Map stats for home team
    home_stats_map = all_team_stats.set_index('match_id')
    home_cols_to_add = {f'home_{col}': home_stats_map[col] for col in home_stats_map.columns if col != 'match_id'}

    # Map stats for away team
    away_stats_map = all_team_stats.set_index('match_id')
    away_cols_to_add = {f'away_{col}': away_stats_map[col] for col in away_stats_map.columns if col != 'match_id'}

    # This is complex. Let's simplify.
    
    # Let's re-think the merge.
    
    # We have `all_team_stats` which has match_id and rolling stats for *one team* in that match.
    # We need to merge it twice.
    
    all_team_stats = all_team_stats.drop_duplicates(subset=['match_id'])
    
    # Merge for home team
    home_stats_cols = [col for col in all_team_stats.columns if col != 'match_id']
    away_stats_cols = [col for col in all_team_stats.columns if col != 'match_id']
    
    home_rolling_df = all_team_stats.rename(columns={col: f'home_{col}' for col in home_stats_cols})
    away_rolling_df = all_team_stats.rename(columns={col: f'away_{col}' for col in away_stats_cols})

    df = df.merge(home_rolling_df, on='match_id', how='left')
    df = df.merge(away_rolling_df, on='match_id', how='left')

    return df

def main():
    """
    Main function to load processed data, engineer features, and save the final dataset.
    """
    logging.info("Starting feature engineering...")
    
    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / "matches.csv", parse_dates=['match_date'])
        logging.info("Loaded processed 'matches.csv' data.")
    except FileNotFoundError:
        logging.error("Could not find 'matches.csv'. Please run data_processing.py first.")
        return

    # --- 1. Create Target Variable ---
    df = create_target_variable(df)
    
    # --- 2. Create Rolling Features ---
    # We need to process all matches together to calculate stats correctly
    
    # Sort by date to ensure rolling windows are causal
    df = df.sort_values(by='match_date')
    
    # Define the stats we want to track for each team
    # 'xg_for', 'xg_against', 'passes_for', 'passes_against' etc.
    
    # This requires gathering all matches for a team, home or away
    
    all_team_matches = []
    
    for team in df['home_team'].unique():
        team_df = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
        team_df['team'] = team
        
        # Create team-centric stats
        team_df['xg_for'] = np.where(team_df['home_team'] == team, team_df['home_xg'], team_df['away_xg'])
        team_df['xg_against'] = np.where(team_df['home_team'] == team, team_df['away_xg'], team_df['home_xg'])
        team_df['passes_for'] = np.where(team_df['home_team'] == team, team_df['home_passes'], team_df['away_passes'])
        team_df['passes_against'] = np.where(team_df['home_team'] == team, team_df['away_passes'], team_df['home_passes'])
        team_df['possession'] = np.where(team_df['home_team'] == team, team_df['home_possession'], team_df['away_possession'])
        
        team_df['score_for'] = np.where(team_df['home_team'] == team, team_df['home_score'], team_df['away_score'])
        team_df['score_against'] = np.where(team_df['home_team'] == team, team_df['away_score'], team_df['home_score'])
        
        # Add points
        team_df['points'] = np.where(team_df['score_for'] > team_df['score_against'], 3,
                                   np.where(team_df['score_for'] == team_df['score_against'], 1, 0))
        
        all_team_matches.append(team_df)

    # Combine back into one big dataframe
    team_centric_df = pd.concat(all_team_matches).sort_values(by='match_date')

    # Define stats to roll
    stats_to_roll = ['xg_for', 'xg_against', 'passes_for', 'passes_against', 'possession', 'points']
    
    # Group by team and calculate rolling averages
    # .shift(1) is CRITICAL! It prevents data leakage by only using past games.
    rolling_stats_df = team_centric_df.groupby('team')[stats_to_roll].rolling(
        window=ROLLING_WINDOW_SIZE, min_periods=1
    ).mean().shift(1).reset_index()

    # Rename columns
    rolling_stats_df.columns = ['team', 'level_1'] + [f'{col}_roll{ROLLING_WINDOW_SIZE}' for col in stats_to_roll]
    
    # Merge rolling stats back into the team-centric dataframe
    final_team_df = team_centric_df.reset_index().merge(
        rolling_stats_df, 
        left_on=['team', 'index'], 
        right_on=['team', 'level_1']
    ).set_index('index')

    # --- 3. Re-assemble the Match-Level DataFrame ---
    # We now have rolling stats for *each team* for *each match*.
    # We need to merge them back into our original `df`
    
    features_df = df.copy()
    
    # Get columns to merge
    rolling_cols = [col for col in rolling_stats_df.columns if col not in ['team', 'level_1']]
    
    # Merge home team stats
    features_df = features_df.merge(
        final_team_df[['match_id', 'team'] + rolling_cols],
        left_on=['match_id', 'home_team'],
        right_on=['match_id', 'team'],
        suffixes=('', '_home_temp')
    ).rename(columns={col: f'home_{col}' for col in rolling_cols})
    
    # Merge away team stats
    features_df = features_df.merge(
        final_team_df[['match_id', 'team'] + rolling_cols],
        left_on=['match_id', 'away_team'],
        right_on=['match_id', 'team'],
        suffixes=('', '_away_temp')
    ).rename(columns={col: f'away_{col}' for col in rolling_cols})

    # Clean up extra columns from merge
    features_df = features_df.drop(columns=[col for col in features_df.columns if '_temp' in col or col == 'team_x' or col == 'team_y' or col == 'level_1_x' or col == 'level_1_y'])
    
    # Drop rows with NaN values (from the first few games where no rolling avg exists)
    features_df = features_df.dropna()
    
    # --- 4. Save Final Features Dataset ---
    output_path = PROCESSED_DATA_DIR / "match_features.csv"
    features_df.to_csv(output_path, index=False)
    
    logging.info(f"Successfully created {len(features_df.columns)} features.")
    logging.info(f"Final feature dataset saved to {output_path}")
    logging.info("--- Feature engineering complete. ---")


if __name__ == "__main__":
    main()