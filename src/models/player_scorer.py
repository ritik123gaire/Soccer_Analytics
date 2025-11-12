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
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support

# Import the paths from our main config file
from config import PROCESSED_DATA_DIR, MODELS_DIR

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TEST_SIZE = 0.2
RANDOM_STATE = 42

def train_player_model():
    """
    Loads engineered player features, splits data, trains, evaluates,
    and saves the player scoring prediction model.
    """
    
    # --- 1. Load Data ---
    logging.info("Loading player feature-engineered data...")
    try:
        features_df = pd.read_csv(PROCESSED_DATA_DIR / "player_features.csv")
    except FileNotFoundError:
        logging.error("player_features.csv not found. Please run player_features.py first.")
        return
        
    if features_df.empty:
        logging.error("Player feature data is empty. Exiting.")
        return

    # --- 2. Define Features (X) and Target (y) ---
    
    # The target is simple: did they score?
    target_col = 'scored_goal'
    
    # The features are the player's rolling stats
    feature_cols = [
        'xg_per_90_roll5', 
        'shots_per_90_roll5', 
        'goals_scored_roll5', 
        'minutes_played_roll5'
    ]
    
    X = features_df[feature_cols]
    y = features_df[target_col]
    
    logging.info(f"Using {len(feature_cols)} features.")
    
    # Check for imbalance
    target_dist = y.value_counts(normalize=True)
    logging.info(f"Target distribution (Imbalance check):\n{target_dist}")
    if target_dist.get(1, 0) < 0.01:
        logging.warning("Target 'scored_goal=1' is <1% of data. Model may struggle.")

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
    # 'class_weight="balanced"' helps the model pay more attention to the rare '1' class
    logging.info("Training Logistic Regression (with class_weight='balanced')...")
    log_reg = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced')
    log_reg.fit(X_train_scaled, y_train)
    
    # Model 2: Random Forest
    logging.info("Training Random Forest (with class_weight='balanced')...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced')
    rf.fit(X_train_scaled, y_train)

    # --- 6. Evaluate Models ---
    
    logging.info("--- Logistic Regression Evaluation ---")
    y_pred_log_reg = log_reg.predict(X_test_scaled)
    y_prob_log_reg = log_reg.predict_proba(X_test_scaled)[:, 1] # Prob of class '1'
    
    print(classification_report(y_test, y_pred_log_reg, target_names=['No Goal (0)', 'Goal (1)']))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_log_reg):.4f}")

    logging.info("--- Random Forest Evaluation ---")
    y_pred_rf = rf.predict(X_test_scaled)
    y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1] # Prob of class '1'
    
    print(classification_report(y_test, y_pred_rf, target_names=['No Goal (0)', 'Goal (1)']))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_rf):.4f}")

    # *** UPDATED ***
    # We pick Logistic Regression, which had better recall and ROC-AUC
    best_model = log_reg
    model_name = "player_scorer_model.pkl"
    scaler_name = "player_scorer_scaler.pkl"

    # --- 7. Save Model and Scaler ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = MODELS_DIR / model_name
    scaler_path = MODELS_DIR / scaler_name
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # *** UPDATED ***
    # Save the feature columns so the prediction script knows what they are
    joblib.dump(X_train.columns, MODELS_DIR / "player_scorer_features.pkl")
    logging.info(f"Feature names saved.")
    
    logging.info(f"Model saved to {model_path}")
    logging.info(f"Scaler saved to {scaler_path}")
    logging.info("--- Player model training complete. ---")


if __name__ == "__main__":
    train_player_model()