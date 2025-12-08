import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import PROCESSED_DATA_DIR, MODELS_DIR

def train_player_model():
    print("Training Player Scoring Model (Real Features)...")
    
    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / "player_features.csv")
    except FileNotFoundError:
        print("❌ player_features.csv not found. Run src/features/player_features.py first.")
        return

    # THESE MUST MATCH YOUR FEATURE ENGINEERING SCRIPT EXACTLY
    features = [
        'xg_per_90_roll5', 
        'shots_per_90_roll5', 
        'goals_scored_roll5', 
        'minutes_played_roll5'
    ]
    target = 'scored_goal'
    
    # Drop rows with NaNs (first 5 games for each player will be empty)
    df_clean = df.dropna(subset=features)
    
    X = df_clean[features]
    y = df_clean[target]
    
    print(f"Training on {len(X)} player records...")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "player_scoring_model.pkl")
    print("✅ Player model saved successfully!")

if __name__ == "__main__":
    train_player_model()