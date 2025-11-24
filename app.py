import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from config import MODELS_DIR, PROCESSED_DATA_DIR

# --- Page Configuration ---
st.set_page_config(
    page_title="Soccer Analytics AI",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ AI Soccer Analytics Dashboard")
st.markdown("### Powered by Machine Learning & Computer Vision")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🏆 Match Predictor", "🏃 Player Scorer", "📹 Computer Vision"])

# ==============================================================================
# TAB 1: MATCH PREDICTOR
# ==============================================================================
with tab1:
    st.header("Predict Match Outcome")
    st.write("Enter the 5-game rolling stats for both teams to predict the winner.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏠 Home Team Stats")
        h_xg = st.number_input("Home Avg xG", 0.0, 5.0, 1.2)
        h_xg_a = st.number_input("Home Avg xG Conceded", 0.0, 5.0, 1.0)
        h_pass = st.number_input("Home Avg Passes", 200, 1000, 450)
        h_pass_a = st.number_input("Home Avg Passes Conceded", 200, 1000, 450)
        h_poss = st.slider("Home Avg Possession %", 0.0, 1.0, 0.5)
        h_pts = st.number_input("Home Avg Points", 0.0, 3.0, 1.5)

    with col2:
        st.subheader("✈️ Away Team Stats")
        a_xg = st.number_input("Away Avg xG", 0.0, 5.0, 1.0)
        a_xg_a = st.number_input("Away Avg xG Conceded", 0.0, 5.0, 1.2)
        a_pass = st.number_input("Away Avg Passes", 200, 1000, 400)
        a_pass_a = st.number_input("Away Avg Passes Conceded", 200, 1000, 500)
        a_poss = st.slider("Away Avg Possession %", 0.0, 1.0, 0.45)
        a_pts = st.number_input("Away Avg Points", 0.0, 3.0, 1.0)

    if st.button("Predict Match Result"):
        try:
            # Load artifacts
            model = joblib.load(MODELS_DIR / "match_outcome_model.pkl")
            scaler = joblib.load(MODELS_DIR / "match_outcome_scaler.pkl")
            features = joblib.load(MODELS_DIR / "match_outcome_features.pkl")

            # Prepare Input
            input_data = [h_xg, h_xg_a, h_pass, h_pass_a, h_poss, h_pts,
                          a_xg, a_xg_a, a_pass, a_pass_a, a_poss, a_pts]
            
            input_df = pd.DataFrame([input_data], columns=features)
            scaled_input = scaler.transform(input_df)

            # Predict
            prediction = model.predict(scaled_input)[0]
            probs = model.predict_proba(scaled_input)[0]

            # Result Mapping
            result_map = {0: "Draw", 1: "Home Win", 2: "Away Win"}
            
            st.success(f"Prediction: **{result_map[prediction]}**")
            
            # Show Probability Chart
            prob_df = pd.DataFrame({
                "Outcome": ["Draw", "Home Win", "Away Win"],
                "Probability": probs
            })
            st.bar_chart(prob_df.set_index("Outcome"))

        except Exception as e:
            st.error(f"Error loading model: {e}")


# ==============================================================================
# TAB 2: PLAYER SCORER
# ==============================================================================
with tab2:
    st.header("Will They Score? 🥅")
    st.write("Enter a player's recent form (last 5 games) to predict if they will score.")

    c1, c2, c3, c4 = st.columns(4)
    
    p_xg = c1.number_input("xG per 90", 0.0, 3.0, 0.4)
    p_shots = c2.number_input("Shots per 90", 0.0, 10.0, 2.5)
    p_goals = c3.number_input("Avg Goals per Game", 0.0, 3.0, 0.2)
    p_mins = c4.number_input("Avg Minutes Played", 0.0, 90.0, 60.0)

    if st.button("Predict Goal Probability"):
        try:
            # Load artifacts
            p_model = joblib.load(MODELS_DIR / "player_scorer_model.pkl")
            p_scaler = joblib.load(MODELS_DIR / "player_scorer_scaler.pkl")
            p_features = joblib.load(MODELS_DIR / "player_scorer_features.pkl")

            # Prepare Input
            p_input = [p_xg, p_shots, p_goals, p_mins]
            p_df = pd.DataFrame([p_input], columns=p_features)
            p_scaled = p_scaler.transform(p_df)

            # Predict
            p_prob = p_model.predict_proba(p_scaled)[0][1] # Prob of Goal

            st.metric(label="Scoring Probability", value=f"{p_prob:.1%}")

            if p_prob > 0.5:
                st.balloons()
                st.success("The model predicts: **GOAL!**")
            else:
                st.warning("The model predicts: **NO GOAL**")

        except Exception as e:
            st.error(f"Error loading model: {e}")


# ==============================================================================
# TAB 3: COMPUTER VISION
# ==============================================================================
with tab3:
    st.header("Computer Vision Analysis 📹")
    
    video_file = PROCESSED_DATA_DIR / "output_video.mp4"
    
    if video_file.exists():
        st.write("This video was processed using YOLOv8 to detect players and the ball.")
        
        # Note: Browsers sometimes struggle with .avi. 
        # If it doesn't play, we might need to convert to .mp4.
        st.video(str(video_file))
    else:
        st.error("Video file not found. Run 'player_detection.py' first.")

st.markdown("---")
st.markdown("Created by **Ritik Gaire** | Soccer Analytics Project")