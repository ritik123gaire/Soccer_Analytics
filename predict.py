import joblib
import numpy as np
import pandas as pd
from config import MODELS_DIR

def predict_match(home_stats, away_stats):
    """
    Loads the saved model and scaler to predict a single match.
    """
    
    # --- 1. Load Model, Scaler, and Feature Names ---
    try:
        model = joblib.load(MODELS_DIR / "match_outcome_model.pkl")
        scaler = joblib.load(MODELS_DIR / "match_outcome_scaler.pkl")
        feature_names = joblib.load(MODELS_DIR / "match_outcome_features.pkl")
    except FileNotFoundError:
        print("Error: Model/scaler/features not found. Run match_predictor.py again.")
        return

    # --- 2. Create the Feature Vector ---
    raw_features = home_stats + away_stats
    
    if len(raw_features) != len(feature_names):
        print(f"Error: Input has {len(raw_features)} features, but model expects {len(feature_names)}.")
        return
        
    # Create the DataFrame *with the correct columns*
    input_df = pd.DataFrame([raw_features], columns=feature_names)

    print("Input features (raw):")
    print(input_df)
    
    # --- 3. Scale the Features ---
    scaled_features = scaler.transform(input_df)

    # --- 4. Make Prediction ---
    prediction = model.predict(scaled_features)
    prediction_proba = model.predict_proba(scaled_features)

    # --- 5. Interpret Results ---
    result_map = {0: "Draw", 1: "Home Win", 2: "Away Win"}
    predicted_class = prediction[0]
    
    print("\n--- PREDICTION ---")
    print(f"Predicted Outcome: {result_map[predicted_class]} (Class {predicted_class})")
    
    print("\nPrediction Probabilities:")
    print(f"  Draw (0):     {prediction_proba[0][0]:.2%}")
    print(f"  Home Win (1): {prediction_proba[0][1]:.2%}")
    print(f"  Away Win (2): {prediction_proba[0][2]:.2%}")


if __name__ == "__main__":
    # Let's invent some stats for "Team A" (Home) vs. "Team B" (Away)
    
    # Team A is in good form: high xG, high points
    # [xg_for, xg_against, passes_for, passes_against, possession, points]
    home_team_stats = [2.1,   0.8,         600,         350,            0.60,      2.5]
    
    # Team B is in poor form: low xG, low points
    # [xg_for, xg_against, passes_for, passes_against, possession, points]
    away_team_stats = [0.9,   1.9,         400,         550,            0.40,      0.8]

    predict_match(home_team_stats, away_team_stats)