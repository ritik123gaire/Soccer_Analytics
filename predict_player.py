import joblib
import numpy as np
import pandas as pd
from config import MODELS_DIR

def predict_player_score(player_stats, player_name=""):
    """
    Loads the saved model and scaler to predict if a player will score.
    
    Args:
        player_stats (list): A list of the 4 rolling stats for the player.
        player_name (str): A label for the player.
    """
    
    # --- 1. Load Model, Scaler, and Feature Names ---
    try:
        model = joblib.load(MODELS_DIR / "player_scorer_model.pkl")
        scaler = joblib.load(MODELS_DIR / "player_scorer_scaler.pkl")
        feature_names = joblib.load(MODELS_DIR / "player_scorer_features.pkl")
    except FileNotFoundError:
        print("Error: Model/scaler/features not found. Run player_scorer.py again.")
        return

    # --- 2. Create the Feature Vector ---
    raw_features = player_stats
    
    if len(raw_features) != len(feature_names):
        print(f"Error: Input has {len(raw_features)} features, but model expects {len(feature_names)}.")
        return
        
    # Create the DataFrame *with the correct columns*
    input_df = pd.DataFrame([raw_features], columns=feature_names)

    print(f"\n------------------------------")
    print(f"Profile: {player_name}")
    print("Input features (raw):")
    print(input_df)
    
    # --- 3. Scale the Features ---
    scaled_features = scaler.transform(input_df)

    # --- 4. Make Prediction ---
    prediction_proba = model.predict_proba(scaled_features)
    prob_of_scoring = prediction_proba[0][1] # Prob of class '1'

    # --- 5. Interpret Results ---
    print(f"\n--- PREDICTION ({player_name}) ---")
    print(f"Probability of Scoring: {prob_of_scoring:.2%}")
    if prob_of_scoring > 0.5:
        print("Prediction: GOAL (Prob > 50%)")
    else:
        print("Prediction: NO GOAL (Prob <= 50%)")


if __name__ == "__main__":
    # --- Here are our 5 test players ---
    
    # The 4 features are:
    # [xg_per_90_roll5, shots_per_90_roll5, goals_scored_roll5, minutes_played_roll5]
    
    # 1. "Hot Striker" (e.g., Lewandowski)
    hot_striker_stats = [0.85, 4.2, 0.8, 88.0]
    predict_player_score(hot_striker_stats, "Hot Striker")
    
    # 2. "Cold Defender" (e.g., a Center Back)
    cold_defender_stats = [0.05, 0.2, 0.0, 90.0]
    predict_player_score(cold_defender_stats, "Cold Defender")

    # 3. "Average Midfielder" (Box-to-box)
    avg_midfielder_stats = [0.15, 1.1, 0.1, 80.0]
    predict_player_score(avg_midfielder_stats, "Average Midfielder")

    # 4. "Unlucky Finisher" (High xG, no goals)
    unlucky_finisher_stats = [0.70, 3.5, 0.0, 85.0]
    predict_player_score(unlucky_finisher_stats, "Unlucky Finisher")
    
    # 5. "Super-Sub" (Low minutes, but decent stats)
    super_sub_stats = [0.60, 3.0, 0.4, 30.0]
    predict_player_score(super_sub_stats, "Super-Sub")