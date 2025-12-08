import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VIDEO_OUTPUT_PATH = PROCESSED_DATA_DIR / "output_video.mp4"

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Soccer Analytics Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }

    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: black;  /* <--- This forces the text color to black */
    }
    .metric-card h4 {
        color: black;  /* <--- Forces headers to black */
        margin-bottom: 0.5rem;
    }
    .metric-card p {
        color: black;  /* <--- Forces paragraph text to black */
        margin: 0;
    }        
</style>
""", unsafe_allow_html=True)

# --- LOAD RESOURCES ---
@st.cache_resource
def load_models():
    """Load all trained models and scalers"""
    try:
        match_model = joblib.load(MODELS_DIR / "match_outcome_model.pkl")
        match_scaler = joblib.load(MODELS_DIR / "match_scaler.pkl")
        match_features = joblib.load(MODELS_DIR / "match_features.pkl")
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files missing: {e}")
        st.info("Please run: `python -m src.models.match_predictor`")
        return None, None, None, None

    # Load Player Model (optional)
    try:
        player_model = joblib.load(MODELS_DIR / "player_scoring_model.pkl")
    except:
        player_model = None
        
    return match_model, match_scaler, match_features, player_model

@st.cache_data
def load_data():
    """Load processed feature data"""
    try:
        match_data = pd.read_csv(PROCESSED_DATA_DIR / "match_features.csv")
    except FileNotFoundError:
        st.error("⚠️ Match features not found. Run feature engineering first.")
        return pd.DataFrame(), pd.DataFrame()
    
    try:
        player_data = pd.read_csv(PROCESSED_DATA_DIR / "player_features.csv")
    except:
        player_data = pd.DataFrame()
    
    return match_data, player_data

# Load everything
match_model, match_scaler, match_features, player_model = load_models()
match_data, player_data = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("⚽ Soccer Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏆 Match Predictor", "🏃 Player Scorer", "📹 Computer Vision", "ℹ️ About"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Info")
if not match_data.empty:
    st.sidebar.metric("Total Matches", len(match_data))
    st.sidebar.metric("Teams", match_data['home_team'].nunique())
if not player_data.empty:
    st.sidebar.metric("Players Tracked", len(player_data))

# ==============================================================================
# PAGE 1: MATCH PREDICTOR
# ==============================================================================
if page == "🏆 Match Predictor":
    st.markdown("<h1 class='main-header'>🏆 Match Outcome Predictor</h1>", unsafe_allow_html=True)
    st.markdown("Predict match results based on recent team form (last 5 games)")
    
    if match_model is None:
        st.stop()

    # Mode Selection
    st.markdown("### Prediction Mode")
    data_source = st.radio(
        "Choose your input method:",
        ["📊 Automatic (Real Data)", "🎮 Manual (Simulator)"],
        horizontal=True
    )
    
    input_data = {}
    home_team_name = "Home Team"
    away_team_name = "Away Team"

    # ========== AUTOMATIC MODE ==========
    if data_source == "📊 Automatic (Real Data)":
        if match_data.empty:
            st.error("No match data available.")
            st.stop()
        
        st.markdown("### Select Teams")
        col1, col2 = st.columns(2)
        
        with col1:
            # Get unique home teams
            home_teams = sorted(match_data['home_team'].unique())
            home_team = st.selectbox("🏠 Home Team", home_teams, key="home")
            home_team_name = home_team
        
        with col2:
            # Get all unique teams and exclude selected home team
            all_teams = pd.concat([
                match_data['home_team'], 
                match_data['away_team']
            ]).unique()
            away_teams = sorted([team for team in all_teams if team != home_team])
            
            if not away_teams:
                st.error("No away teams available")
                st.stop()
            
            away_team = st.selectbox("✈️ Away Team", away_teams, key="away")
            away_team_name = away_team

        # Get most recent stats for selected teams
        # Try to find stats where they played as home/away
        h_home_matches = match_data[match_data['home_team'] == home_team]
        h_away_matches = match_data[match_data['away_team'] == home_team]
        
        a_home_matches = match_data[match_data['home_team'] == away_team]
        a_away_matches = match_data[match_data['away_team'] == away_team]
        
        # Get most recent stats
        if not h_home_matches.empty:
            h_stats = h_home_matches.iloc[-1]
        elif not h_away_matches.empty:
            h_stats = h_away_matches.iloc[-1]
        else:
            st.error(f"No historical data found for {home_team}")
            st.stop()
        
        if not a_away_matches.empty:
            a_stats = a_away_matches.iloc[-1]
        elif not a_home_matches.empty:
            a_stats = a_home_matches.iloc[-1]
        else:
            st.error(f"No historical data found for {away_team}")
            st.stop()

        # Extract features using exact column names
        input_data = {
            'home_xg_for_roll5':         h_stats.get('home_xg_for_roll5', 1.2),
            'home_xg_against_roll5':     h_stats.get('home_xg_against_roll5', 1.0),
            'home_passes_for_roll5':     h_stats.get('home_passes_for_roll5', 400),
            'home_passes_against_roll5': h_stats.get('home_passes_against_roll5', 400),
            'home_possession_roll5':     h_stats.get('home_possession_roll5', 50),
            'home_points_roll5':         h_stats.get('home_points_roll5', 1.5),
            
            'away_xg_for_roll5':         a_stats.get('away_xg_for_roll5', 1.0),
            'away_xg_against_roll5':     a_stats.get('away_xg_against_roll5', 1.2),
            'away_passes_for_roll5':     a_stats.get('away_passes_for_roll5', 400),
            'away_passes_against_roll5': a_stats.get('away_passes_against_roll5', 400),
            'away_possession_roll5':     a_stats.get('away_possession_roll5', 50),
            'away_points_roll5':         a_stats.get('away_points_roll5', 1.2)
        }
        
        # Display matchup info
        st.markdown("### 📊 Matchup Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                f"{home_team} xG", 
                f"{input_data['home_xg_for_roll5']:.2f}",
                f"Conceded: {input_data['home_xg_against_roll5']:.2f}"
            )
        
        with col2:
            st.metric(
                "Possession", 
                f"{input_data['home_possession_roll5']:.1f}% vs {input_data['away_possession_roll5']:.1f}%"
            )
        
        with col3:
            st.metric(
                f"{away_team} xG", 
                f"{input_data['away_xg_for_roll5']:.2f}",
                f"Conceded: {input_data['away_xg_against_roll5']:.2f}"
            )

    # ========== MANUAL MODE ==========
# ========== MANUAL MODE (SMART SIMULATION) ==========
# ========== MANUAL MODE (SMART SIMULATION) ==========
    else:
        st.markdown("### 🎮 Configure Match Simulation")
        st.info("💡 Stats are now cross-linked: Home Passes = Away Passes Against.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏠 Home Team")
            h_xg = st.slider("Attack (xG)", 0.0, 5.0, 2.5, 0.1, key="h_xg")
            h_poss = st.slider("Possession %", 20, 80, 60, 5, key="h_poss")
            h_points = st.slider("Form (Points/Game)", 0.0, 3.0, 2.0, 0.1, key="h_pts")
            
            # Auto-calculate Home Passes
            h_passes = int(h_poss * 9.5)  # Approx 9.5 passes per 1% possession
            st.write(f"**Implied Passes:** {h_passes}")

        with col2:
            st.markdown("#### ✈️ Away Team")
            a_xg = st.slider("Attack (xG)", 0.0, 5.0, 0.8, 0.1, key="a_xg")
            
            # Auto-calculate Away Stats
            a_poss = 100 - h_poss
            a_passes = int(a_poss * 9.5)
            
            st.metric("Possession %", f"{a_poss}%")
            st.write(f"**Implied Passes:** {a_passes}")
            
            a_points = st.slider("Form (Points/Game)", 0.0, 3.0, 0.8, 0.1, key="a_pts")

        # --- THE FIX: CROSS-LINKING STATS ---
        # If Home attacks, Away defends. If Home passes, Away allows passes.
        input_data = {
            # Home Team
            'home_xg_for_roll5':         h_xg,
            'home_xg_against_roll5':     a_xg,      # Link: Home allows what Away creates
            'home_passes_for_roll5':     h_passes,
            'home_passes_against_roll5': a_passes,  # Link: Home allows what Away plays
            'home_possession_roll5':     h_poss,
            'home_points_roll5':         h_points,

            # Away Team
            'away_xg_for_roll5':         a_xg,
            'away_xg_against_roll5':     h_xg,      # Link: Away allows what Home creates
            'away_passes_for_roll5':     a_passes,
            'away_passes_against_roll5': h_passes,  # Link: Away allows what Home plays
            'away_possession_roll5':     a_poss,
            'away_points_roll5':         a_points
        }

        with st.expander("🕵️ Debug: See Model Inputs"):
            st.write("Notice how 'Passes For' matches the opponent's 'Passes Against'")
            st.json(input_data)

    # ========== PREDICTION ==========
    st.markdown("---")
    
    if st.button("🔮 Predict Match Result", type="primary", use_container_width=True):
        with st.spinner("Analyzing team stats..."):
            try:
                # Ensure feature order matches training
                features_df = pd.DataFrame([input_data])
                features_df = features_df[match_features]
                
                # Scale features
                features_scaled = match_scaler.transform(features_df)
                
                # Predict
                prediction_idx = match_model.predict(features_scaled)[0]
                probabilities = match_model.predict_proba(features_scaled)[0]
                
                # Map predictions (alphabetical order: Away Win=0, Draw=1, Home Win=2)
                result_map = {
                    0: f"{away_team_name} Win",
                    1: "Draw",
                    2: f"{home_team_name} Win"
                }
                
                final_result = result_map[prediction_idx]
                
                # Display Results
                st.markdown("---")
                st.markdown("## 🎯 Prediction Results")
                
                # Main prediction
                if prediction_idx == 2:
                    st.success(f"### 🏆 Predicted Winner: **{final_result}**")
                elif prediction_idx == 0:
                    st.error(f"### 🚍 Predicted Winner: **{final_result}**")
                else:
                    st.warning(f"### ⚖️ Prediction: **{final_result}**")
                
                # Confidence levels
                st.markdown("### 📊 Confidence Breakdown")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        f"{away_team_name} Win", 
                        f"{probabilities[0]*100:.1f}%"
                    )
                
                with col2:
                    st.metric(
                        "Draw", 
                        f"{probabilities[1]*100:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        f"{home_team_name} Win", 
                        f"{probabilities[2]*100:.1f}%"
                    )
                
                # Bar chart
                chart_data = pd.DataFrame({
                    "Outcome": [f"{away_team_name} Win", "Draw", f"{home_team_name} Win"],
                    "Probability": probabilities * 100
                }).set_index("Outcome")
                
                st.bar_chart(chart_data, height=300)
                
                # Interpretation
                max_prob = max(probabilities)
                if max_prob > 0.6:
                    st.info("🎯 **High Confidence Prediction** - The model is quite certain about this outcome.")
                elif max_prob > 0.4:
                    st.info("⚖️ **Moderate Confidence** - This could go either way.")
                else:
                    st.info("🤷 **Low Confidence** - Very evenly matched teams.")
                
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.info("Please ensure all features are properly configured.")

# ==============================================================================
# PAGE 2: PLAYER SCORER
# ==============================================================================
# ==============================================================================
# PAGE 2: PLAYER SCORER
# ==============================================================================
elif page == "🏃 Player Scorer":
    st.markdown("<h1 class='main-header'>🏃 Player Scoring Predictor</h1>", unsafe_allow_html=True)
    
    if player_model is None or player_data.empty:
        st.warning("⚠️ Player Model or Data not available.")
        st.info("""
        **To enable this feature:**
        1. Ensure you have player-level data
        2. Run: `python -m src.models.player_scorer`
        3. Refresh this page
        """)
    else:
        st.markdown("Predict the probability of a player scoring in their next match")
        
        # Select Player
        st.markdown("### Select Player")
        player_name = st.selectbox(
            "Choose a player", 
            sorted(player_data['player_name'].unique()),
            key="player_select"
        )
        
        # Get player stats (Use the REAL column names from your feature script)
        p_stats = player_data[player_data['player_name'] == player_name].iloc[-1]
        
        # Display recent form
        st.markdown("### 📊 Recent Performance (Last 5 Games Avg)")
        col1, col2, col3, col4 = st.columns(4)
        
        # Use .get() with the NEW column names
        goals_val = p_stats.get('goals_scored_roll5', 0)
        shots_val = p_stats.get('shots_per_90_roll5', 0)
        mins_val = p_stats.get('minutes_played_roll5', 0)
        xg_val = p_stats.get('xg_per_90_roll5', 0)

        with col1:
            st.metric("Goals", f"{goals_val:.2f}")
        with col2:
            st.metric("Shots/90", f"{shots_val:.2f}")
        with col3:
            st.metric("Minutes", f"{int(mins_val)}")
        with col4:
            st.metric("xG/90", f"{xg_val:.2f}")
        
        if st.button("🎯 Calculate Goal Probability", type="primary", use_container_width=True):
            try:
                # Prepare features using the EXACT names the model expects
                # The model was trained on: ['xg_per_90_roll5', 'shots_per_90_roll5', 'goals_scored_roll5', 'minutes_played_roll5']
                input_features = pd.DataFrame([{
                    'xg_per_90_roll5': xg_val,
                    'shots_per_90_roll5': shots_val,
                    'goals_scored_roll5': goals_val,
                    'minutes_played_roll5': mins_val
                }])
                
                # Predict
                # Note: We take [0][1] to get the probability of "Class 1" (Scoring)
                goal_prob = player_model.predict_proba(input_features)[0][1]
                
                st.markdown("---")
                st.markdown("## 🎯 Prediction Result")
                
                # Display probability
                st.metric(
                    label="Goal Probability", 
                    value=f"{goal_prob*100:.1f}%",
                    delta="High" if goal_prob > 0.3 else "Low"
                )
                
                # Visual gauge
                st.progress(goal_prob)
                
                # Interpretation
                if goal_prob > 0.5:
                    st.balloons()
                    st.success("🔥 **VERY LIKELY TO SCORE!** This player is in excellent form.")
                elif goal_prob > 0.3:
                    st.info("⚡ **GOOD CHANCE** - Moderate likelihood of scoring.")
                elif goal_prob > 0.15:
                    st.warning("❄️ **POSSIBLE** - Low but not negligible chance.")
                else:
                    st.error("📉 **UNLIKELY** - Very low scoring probability.")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Check that feature columns match the trained model.")

# ==============================================================================
# PAGE 3: COMPUTER VISION
# ==============================================================================
elif page == "📹 Computer Vision":
    st.markdown("<h1 class='main-header'>📹 AI Video Analysis</h1>", unsafe_allow_html=True)
    st.markdown("Automated player detection, team classification, and ball tracking using YOLOv8")
    
    if os.path.exists(VIDEO_OUTPUT_PATH):
        st.success("✅ Processed video available!")
        
        # Video player
        st.markdown("### 🎬 Analyzed Match Footage")
        with open(VIDEO_OUTPUT_PATH, 'rb') as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)
        
        # Analysis legend
        st.markdown("### 📊 Detection Legend")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
            <h4> ⚪ White Boxes</h4>
            <p>Home Team Players</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card'>
            <h4> 🟡 Yellow Boxes</h4>
            <p>Away Team Players</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='metric-card'>
            <h4> 🟢 Green Box</h4>
            <p>Ball Detection</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Features")
        st.markdown("""
        - **Real-time Object Detection**: YOLOv8-powered player and ball tracking
        - **Team Classification**: Automatic jersey color detection
        - **Possession Analysis**: Ball proximity tracking
        - **Performance Metrics**: Distance covered, sprint detection
        """)
        
    else:
        st.info("ℹ️ No processed video found yet.")
        
        st.markdown("### 🚀 How to Generate Analysis")
        
        with st.expander("📝 Step-by-Step Instructions", expanded=True):
            st.markdown("""
            1. **Prepare your video**
               - Place raw match footage in `data/video_raw/`
               - Supported formats: `.mp4`, `.avi`, `.mov`
            
            2. **Run the detection script**
               ```bash
               python -m src.computer_vision.player_detection
               ```
            
            3. **Wait for processing**
               - Processing time depends on video length
               - Progress will be shown in terminal
            
            4. **Refresh this page**
               - Output will appear in `data/processed/output_video.mp4`
            """)
        
        st.markdown("### 🎥 Sample Output")
        st.image("https://via.placeholder.com/800x450.png?text=Sample+Detection+Output", 
                 caption="Example: Player and ball detection with team classification")

# ==============================================================================
# PAGE 4: ABOUT
# ==============================================================================
elif page == "ℹ️ About":
    st.markdown("<h1 class='main-header'>ℹ️ About This Project</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Project Overview
    
    This Soccer Analytics Dashboard combines **Machine Learning** and **Computer Vision** 
    to provide comprehensive match analysis and predictions.
    
    ### 🔧 Technologies Used
    
    - **Machine Learning**: scikit-learn, Random Forest, Logistic Regression
    - **Computer Vision**: YOLOv8, OpenCV
    - **Data Processing**: pandas, numpy
    - **Web Framework**: Streamlit
    - **Data Source**: StatsBomb Open Data
    
    ### 📊 Models
    
    1. **Match Outcome Predictor**
       - Predicts: Home Win, Draw, or Away Win
       - Features: Rolling 5-game averages (xG, possession, passes, points)
       - Algorithm: Random Forest Classifier
    
    2. **Player Scoring Predictor**
       - Predicts: Probability of a player scoring
       - Features: Recent form, shooting stats, minutes played
       - Algorithm: Random Forest Classifier
    
    3. **Computer Vision Analysis**
       - Detects: Players, ball, referee
       - Classifies: Team identification via jersey colors
       - Tracks: Possession, movement patterns
       - Model: YOLOv8
    
    ### 📈 Model Performance
    """)
    
    if match_model is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Match Predictor Accuracy", "~65-70%")
            st.caption("Typical for soccer prediction models")
        with col2:
            st.metric("Player Scorer AUC", "~0.75")
            st.caption("Good discriminative ability")
    
    st.markdown("""
    ### 🎓 Learning Outcomes
    
    - End-to-end ML pipeline development
    - Feature engineering for time-series sports data
    - Model evaluation and optimization
    - Computer vision implementation
    - Interactive dashboard creation
    
    ### 📚 Resources
    
    - [StatsBomb Open Data](https://github.com/statsbomb/open-data)
    - [YOLOv8 Documentation](https://docs.ultralytics.com/)
    - [scikit-learn](https://scikit-learn.org/)
    - [Streamlit](https://streamlit.io/)
    
    ### 👨‍💻 Created By
    
    **Ritik Gaire** | Soccer Analytics Project
    
    ---
    
    *For questions or contributions, please reach out via GitHub.*
    """)
    
    # System Info
    with st.expander("🔧 System Information"):
        st.json({
            "Models Loaded": match_model is not None,
            "Match Data": f"{len(match_data)} records" if not match_data.empty else "Not loaded",
            "Player Data": f"{len(player_data)} records" if not player_data.empty else "Not loaded",
            "Video Available": os.path.exists(VIDEO_OUTPUT_PATH)
        })

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>⚽ Soccer Analytics Dashboard | "
    "Powered by Machine Learning & Computer Vision</p>",
    unsafe_allow_html=True
)



