import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path

# Model Selection
from sklearn.model_selection import train_test_split

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Evaluation
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Import the paths from our main config file
from config import PROCESSED_DATA_DIR, MODELS_DIR

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TEST_SIZE = 0.2
RANDOM_STATE = 42

def train_model():
    """
    Loads the engineered features, splits the data, trains, evaluates,
    and saves the match outcome prediction model.
    """
    
    # --- 1. Load Data ---
    logging.info("Loading feature-engineered data...")
    try:
        features_df = pd.read_csv(PROCESSED_DATA_DIR / "match_features.csv")
    except FileNotFoundError:
        logging.error("match_features.csv not found. Please run feature engineering first.")
        return
        
    if features_df.empty:
        logging.error("Feature data is empty. Exiting.")
        return

    # --- 2. Define Features (X) and Target (y) ---
    
    # The target is what we want to predict
    target_col = 'match_result'
    
    # The features are all the rolling stats
    # We exclude IDs, dates, team names, and the actual match results (score, xg, etc.)
    features_to_drop = [
        'match_id', 'match_date', 'home_team', 'away_team', 'team',
        'home_score', 'away_score', 'home_xg', 'away_xg', 
        'home_passes', 'away_passes', 'home_possession', 'away_possession',
        'match_result'
    ]
    
    # Get all columns that are *not* in the drop list
    feature_cols = [col for col in features_df.columns if col not in features_to_drop]
    
    # Handle potential NaN values from the first few rolling games
    features_df = features_df.dropna(subset=feature_cols + [target_col])

    X = features_df[feature_cols]
    y = features_df[target_col]
    
    logging.info(f"Using {len(feature_cols)} features.")
    logging.info(f"Target distribution:\n{y.value_counts(normalize=True)}")

    # --- 3. Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    logging.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # --- 4. Feature Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 5. Train Models ---
    
    # Model 1: Logistic Regression
    logging.info("Training Logistic Regression...")
    log_reg = LogisticRegression(random_state=RANDOM_STATE, multi_class='multinomial', max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    
    # Model 2: Random Forest
    logging.info("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    # Random Forest doesn't strictly *need* scaled data, but it doesn't hurt
    rf.fit(X_train_scaled, y_train)

    # --- 6. Evaluate Models ---
    
    logging.info("--- Logistic Regression Evaluation ---")
    y_pred_log_reg = log_reg.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_log_reg, target_names=['Draw (0)', 'Home Win (1)', 'Away Win (2)']))
    
    logging.info("--- Random Forest Evaluation ---")
    y_pred_rf = rf.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_rf, target_names=['Draw (0)', 'Home Win (1)', 'Away Win (2)']))

    # Let's pick the best model (Random Forest is usually better for this)
    best_model = rf
    model_name = "match_outcome_model.pkl"
    scaler_name = "match_outcome_scaler.pkl"

    # --- 7. Save Model and Scaler ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = MODELS_DIR / model_name
    scaler_path = MODELS_DIR / scaler_name
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    logging.info(f"Model saved to {model_path}")
    logging.info(f"Scaler saved to {scaler_path}")
    logging.info("--- Model training complete. ---")


if __name__ == "__main__":
    train_model()