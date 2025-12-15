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
        color: black;
    }
    .metric-card h4 {
        color: black;
        margin-bottom: 0.5rem;
    }
    .metric-card p {
        color: black;
        margin: 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
    }
    .warning-box {
        background-color: #fff4e5;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
    }
    .stat-comparison {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #4CAF50 50%, #f44336 50%);
        border-radius: 0.5rem;
        color: white;
        font-weight: bold;
    }
            
    .insight-box {
    background-color: #e8f4f8;
    border-left: 4px solid #2196F3;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 0.3rem;
    color: #1a1a1a !important;
}

.insight-box * {
    color: #1a1a1a !important;
}

.warning-box {
    background-color: #fff4e5;
    border-left: 4px solid #ff9800;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 0.3rem;
    color: #1a1a1a !important;
}

.warning-box * {
    color: #1a1a1a !important;
}

.success-box {
    background-color: #e8f5e9;
    border-left: 4px solid #4caf50;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 0.3rem;
    color: #1a1a1a !important;
}

.success-box * {
    color: #1a1a1a !important;
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

def get_team_stats(team_name, match_data, prefer_home=True):
    """Get most recent stats for a team, with better fallback logic"""
    if prefer_home:
        primary = match_data[match_data['home_team'] == team_name]
        secondary = match_data[match_data['away_team'] == team_name]
    else:
        primary = match_data[match_data['away_team'] == team_name]
        secondary = match_data[match_data['home_team'] == team_name]
    
    if not primary.empty:
        return primary.iloc[-1], prefer_home
    elif not secondary.empty:
        return secondary.iloc[-1], not prefer_home
    else:
        return None, None

def generate_insights(input_data, probabilities, home_team, away_team):
    """Generate intelligent insights about the prediction"""
    insights = []
    warnings = []
    
    # Extract key metrics
    home_xg_diff = input_data['home_xg_for_roll5'] - input_data['home_xg_against_roll5']
    away_xg_diff = input_data['away_xg_for_roll5'] - input_data['away_xg_against_roll5']
    
    home_form = input_data['home_points_roll5']
    away_form = input_data['away_points_roll5']
    
    poss_diff = input_data['home_possession_roll5'] - input_data['away_possession_roll5']
    
    # Form analysis
    if home_form >= 2.0 and away_form < 1.5:
        insights.append(f"🔥 **{home_team}** is in excellent form ({home_form:.1f} pts/game) while **{away_team}** struggles ({away_form:.1f} pts/game)")
    elif away_form >= 2.0 and home_form < 1.5:
        insights.append(f"🔥 **{away_team}** is in excellent form ({away_form:.1f} pts/game) while **{home_team}** struggles ({home_form:.1f} pts/game)")
    elif abs(home_form - away_form) < 0.3:
        insights.append(f"⚖️ Both teams show similar form ({home_form:.1f} vs {away_form:.1f} pts/game)")
    
    # Poor form warning
    if home_form < 1.0 and away_form < 1.0:
        warnings.append(f"⚠️ **Both teams struggling** - Combined averaging only {home_form + away_form:.1f} pts/game")
    
    # Attack vs Defense
    if home_xg_diff > 0.5:
        insights.append(f"⚔️ **{home_team}** has a strong attacking advantage (xG diff: +{home_xg_diff:.2f})")
    elif home_xg_diff < -0.3:
        warnings.append(f"🛡️ **{home_team}** defense has been leaky (conceding {input_data['home_xg_against_roll5']:.2f} xG/game vs creating {input_data['home_xg_for_roll5']:.2f})")
    
    if away_xg_diff > 0.5:
        insights.append(f"⚔️ **{away_team}** has a strong attacking advantage (xG diff: +{away_xg_diff:.2f})")
    elif away_xg_diff < -0.3:
        warnings.append(f"🛡️ **{away_team}** defense has been leaky (conceding {input_data['away_xg_against_roll5']:.2f} xG/game vs creating {input_data['away_xg_for_roll5']:.2f})")
    
    # Possession style
    if abs(poss_diff) > 15:
        dom_team = home_team if poss_diff > 0 else away_team
        insights.append(f"📊 **{dom_team}** dominates possession ({max(input_data['home_possession_roll5'], input_data['away_possession_roll5']):.1f}%)")
    elif abs(poss_diff) < 5:
        insights.append(f"⚖️ **Even possession battle** - Both teams around 50% ball control")
    
    # Prediction confidence analysis
    max_prob = max(probabilities)
    if max_prob < 0.4:
        warnings.append("⚠️ **Low confidence prediction** - These teams are very evenly matched, result could go any way")
    elif max_prob > 0.65:
        winner = [away_team, "Draw", home_team][np.argmax(probabilities)]
        insights.append(f"✅ **High confidence** in {winner} outcome ({max_prob*100:.1f}%)")
    
    # Head-to-head style clash
    home_attacking = input_data['home_xg_for_roll5'] > 1.3
    away_attacking = input_data['away_xg_for_roll5'] > 1.3
    
    if home_attacking and away_attacking:
        insights.append("🎯 **High-scoring game expected** - Both teams create chances regularly")
    elif not home_attacking and not away_attacking:
        warnings.append("🔒 **Low-scoring game likely** - Both teams struggle to create clear chances")
    
    # Form vs Quality mismatch
    if abs(home_xg_diff - away_xg_diff) > 0.6:
        better_team = home_team if home_xg_diff > away_xg_diff else away_team
        insights.append(f"💎 **{better_team}** has significantly better underlying stats despite current form")
    
    return insights, warnings

def generate_player_insights(xg_val, shots_val, goals_val, mins_val, goal_prob):
    """Generate detailed insights for player scoring prediction"""
    insights = []
    warnings = []
    
    # Efficiency analysis
    if goals_val > 0 and shots_val > 0:
        conversion_rate = (goals_val / (shots_val * (mins_val / 90))) * 100
        if conversion_rate > 20:
            insights.append(f"🎯 **Elite finisher** - Converting {conversion_rate:.1f}% of attempts")
        elif conversion_rate > 15:
            insights.append(f"✅ **Good conversion rate** - {conversion_rate:.1f}% success rate")
        elif conversion_rate < 8:
            warnings.append(f"⚠️ **Low conversion** - Only {conversion_rate:.1f}% of shots becoming goals")
    
    # Volume shooter
    if shots_val > 3.5:
        insights.append(f"🔫 **High volume shooter** - Averaging {shots_val:.1f} shots per 90 minutes")
    elif shots_val < 1.5:
        warnings.append(f"📉 **Limited shooting** - Only {shots_val:.1f} shots per 90 minutes")
    
    # xG analysis
    if xg_val > 0.5:
        insights.append(f"⭐ **Quality chances** - Getting {xg_val:.2f} xG per 90 (excellent positioning)")
    elif xg_val < 0.2:
        warnings.append(f"🚫 **Poor chance quality** - Only {xg_val:.2f} xG per 90 (low-quality attempts)")
    
    # Over/underperforming xG
    if goals_val > 0 and xg_val > 0:
        xg_diff = goals_val - (xg_val * (mins_val / 90))
        if xg_diff > 1.0:
            insights.append(f"🔥 **Hot streak** - Scoring {xg_diff:.1f} more goals than expected!")
        elif xg_diff < -1.0:
            warnings.append(f"❄️ **Cold streak** - Underperforming xG by {abs(xg_diff):.1f} goals")
    
    # Playing time
    if mins_val > 80:
        insights.append(f"💪 **Regular starter** - Playing {mins_val:.0f} mins per game")
    elif mins_val < 45:
        warnings.append(f"⏱️ **Limited minutes** - Only {mins_val:.0f} mins per game (impact sub role)")
    
    # Form analysis
    if goals_val > 0.8:
        insights.append(f"🌟 **In form** - Scoring {goals_val:.2f} goals per game recently")
    elif goals_val == 0:
        warnings.append("📭 **Goal drought** - No goals in last 5 games")
    
    # Probability interpretation
    if goal_prob > 0.5:
        insights.append("🎰 **Better than 50/50** - Model expects a goal more likely than not")
    elif goal_prob > 0.3:
        insights.append("⚡ **Decent chance** - About 1 in 3 probability of scoring")
    elif goal_prob < 0.15:
        warnings.append("📉 **Unlikely to score** - Would need things to go very right")
    
    return insights, warnings

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
    all_teams = pd.concat([match_data['home_team'], match_data['away_team']]).unique()
    st.sidebar.metric("Teams", len(all_teams))
if not player_data.empty:
    st.sidebar.metric("Players Tracked", len(player_data))

# ==============================================================================
# PAGE 1: MATCH PREDICTOR (IMPROVED)
# ==============================================================================
if page == "🏆 Match Predictor":
    st.markdown("<h1 class='main-header'>🏆 Match Outcome Predictor</h1>", unsafe_allow_html=True)
    st.markdown("Predict match results based on recent team form (last 5 games)")
    
    if match_model is None:
        st.stop()

    st.markdown("### Prediction Mode")
    data_source = st.radio(
        "Choose your input method:",
        ["📊 Automatic (Real Data)", "🎮 Manual (Simulator)"],
        horizontal=True
    )
    
    input_data = {}
    home_team_name = "Home Team"
    away_team_name = "Away Team"

    # # ========== AUTOMATIC MODE (IMPROVED) ==========
    # if data_source == "📊 Automatic (Real Data)":
    #     if match_data.empty:
    #         st.error("No match data available.")
    #         st.stop()
        
    #     st.markdown("### Select Teams")
    #     col1, col2 = st.columns(2)
        
    #     all_teams = sorted(pd.concat([
    #         match_data['home_team'], 
    #         match_data['away_team']
    #     ]).unique())
        
    #     with col1:
    #         home_team = st.selectbox("🏠 Home Team", all_teams, key="home")
    #         home_team_name = home_team
        
    #     with col2:
    #         away_teams = [team for team in all_teams if team != home_team]
    #         if not away_teams:
    #             st.error("No away teams available")
    #             st.stop()
            
    #         away_team = st.selectbox("✈️ Away Team", away_teams, key="away")
    #         away_team_name = away_team

    #     h_stats, h_is_home = get_team_stats(home_team, match_data, prefer_home=True)
    #     a_stats, a_is_home = get_team_stats(away_team, match_data, prefer_home=False)
        
    #     if h_stats is None:
    #         st.error(f"No historical data found for {home_team}")
    #         st.stop()
    #     if a_stats is None:
    #         st.error(f"No historical data found for {away_team}")
    #         st.stop()

    #     h_prefix = 'home' if h_is_home else 'away'
    #     a_prefix = 'away' if not a_is_home else 'home'
        
    #     try:
    #         input_data = {
    #             'home_xg_for_roll5':         float(h_stats.get(f'{h_prefix}_xg_for_roll5', 1.2)),
    #             'home_xg_against_roll5':     float(h_stats.get(f'{h_prefix}_xg_against_roll5', 1.0)),
    #             'home_passes_for_roll5':     float(h_stats.get(f'{h_prefix}_passes_for_roll5', 400)),
    #             'home_passes_against_roll5': float(h_stats.get(f'{h_prefix}_passes_against_roll5', 400)),
    #             'home_possession_roll5':     float(h_stats.get(f'{h_prefix}_possession_roll5', 50)),
    #             'home_points_roll5':         float(h_stats.get(f'{h_prefix}_points_roll5', 1.5)),
                
    #             'away_xg_for_roll5':         float(a_stats.get(f'{a_prefix}_xg_for_roll5', 1.0)),
    #             'away_xg_against_roll5':     float(a_stats.get(f'{a_prefix}_xg_against_roll5', 1.2)),
    #             'away_passes_for_roll5':     float(a_stats.get(f'{a_prefix}_passes_for_roll5', 400)),
    #             'away_passes_against_roll5': float(a_stats.get(f'{a_prefix}_passes_against_roll5', 400)),
    #             'away_possession_roll5':     float(a_stats.get(f'{a_prefix}_possession_roll5', 50)),
    #             'away_points_roll5':         float(a_stats.get(f'{a_prefix}_points_roll5', 1.2))
    #         }
            
    #         # FIX: Ensure possession values are reasonable
    #         if input_data['home_possession_roll5'] < 10:
    #             input_data['home_possession_roll5'] = 50
    #         if input_data['away_possession_roll5'] < 10:
    #             input_data['away_possession_roll5'] = 50
                
    #     except Exception as e:
    #         st.error(f"Error extracting features: {e}")
    #         with st.expander("🔍 Debug Info"):
    #             st.write("Home stats columns:", h_stats.index.tolist())
    #             st.write("Away stats columns:", a_stats.index.tolist())
    #         st.stop()
        
    #     # Display matchup info
    #     st.markdown("### 📊 Team Comparison")
        
    #     # Stats comparison grid
    #     col1, col2, col3 = st.columns([2, 1, 2])
        
    #     with col1:
    #         st.markdown(f"#### 🏠 {home_team}")
    #         st.metric("Attack (xG/game)", f"{input_data['home_xg_for_roll5']:.2f}")
    #         st.metric("Defense (xG conceded)", f"{input_data['home_xg_against_roll5']:.2f}")
    #         st.metric("Possession", f"{input_data['home_possession_roll5']:.1f}%")
    #         st.metric("Form (pts/game)", f"{input_data['home_points_roll5']:.2f}")
        
    #     with col2:
    #         st.markdown("#### ⚖️ vs")
            
    #         # xG Advantage
    #         xg_diff = input_data['home_xg_for_roll5'] - input_data['away_xg_for_roll5']
    #         if abs(xg_diff) > 0.15:
    #             adv_team = "🏠" if xg_diff > 0 else "✈️"
    #             st.metric("Attack", adv_team, f"{abs(xg_diff):.2f}")
    #         else:
    #             st.metric("Attack", "⚖️", "Even")
            
    #         # Defense Advantage (lower is better)
    #         def_diff = input_data['away_xg_against_roll5'] - input_data['home_xg_against_roll5']
    #         if abs(def_diff) > 0.15:
    #             def_team = "🏠" if def_diff > 0 else "✈️"
    #             st.metric("Defense", def_team, f"{abs(def_diff):.2f}")
    #         else:
    #             st.metric("Defense", "⚖️", "Even")
            
    #         # Form
    #         form_diff = input_data['home_points_roll5'] - input_data['away_points_roll5']
    #         if abs(form_diff) > 0.3:
    #             form_team = "🏠" if form_diff > 0 else "✈️"
    #             st.metric("Form", form_team, f"{abs(form_diff):.2f}")
    #         else:
    #             st.metric("Form", "⚖️", "Even")
        
    #     with col3:
    #         st.markdown(f"#### ✈️ {away_team}")
    #         st.metric("Attack (xG/game)", f"{input_data['away_xg_for_roll5']:.2f}")
    #         st.metric("Defense (xG conceded)", f"{input_data['away_xg_against_roll5']:.2f}")
    #         st.metric("Possession", f"{input_data['away_possession_roll5']:.1f}%")
    #         st.metric("Form (pts/game)", f"{input_data['away_points_roll5']:.2f}")

    if data_source == "📊 Automatic (Real Data)":
        if match_data.empty:
            st.error("No match data available.")
            st.stop()
        
        st.markdown("### Select Teams")
        col1, col2 = st.columns(2)
        
        all_teams = sorted(pd.concat([
            match_data['home_team'], 
            match_data['away_team']
        ]).unique())
        
        with col1:
            home_team = st.selectbox("🏠 Home Team", all_teams, key="home")
            home_team_name = home_team
        
        with col2:
            away_teams = [team for team in all_teams if team != home_team]
            if not away_teams:
                st.error("No away teams available")
                st.stop()
            
            away_team = st.selectbox("✈️ Away Team", away_teams, key="away")
            away_team_name = away_team

        # Get latest stats for each team (home + away context aware)
        h_stats, h_is_home = get_team_stats(home_team, match_data, prefer_home=True)
        a_stats, a_is_home = get_team_stats(away_team, match_data, prefer_home=False)
        
        if h_stats is None:
            st.error(f"No historical data found for {home_team}")
            st.stop()
        if a_stats is None:
            st.error(f"No historical data found for {away_team}")
            st.stop()

        # Decide which prefix to use based on whether the team
        # appears as home or away in the latest row
        h_prefix = 'home' if h_is_home else 'away'
        a_prefix = 'away' if not a_is_home else 'home'
        
        try:
            # NOTE: possession in CSV is 0–1 (fraction), keep that for the model
            input_data = {
                'home_xg_for_roll5':         float(h_stats.get(f'{h_prefix}_xg_for_roll5', 1.2)),
                'home_xg_against_roll5':     float(h_stats.get(f'{h_prefix}_xg_against_roll5', 1.0)),
                'home_passes_for_roll5':     float(h_stats.get(f'{h_prefix}_passes_for_roll5', 400)),
                'home_passes_against_roll5': float(h_stats.get(f'{h_prefix}_passes_against_roll5', 400)),
                'home_possession_roll5':     float(h_stats.get(f'{h_prefix}_possession_roll5', 0.5)),  # 0–1
                'home_points_roll5':         float(h_stats.get(f'{h_prefix}_points_roll5', 1.5)),
                
                'away_xg_for_roll5':         float(a_stats.get(f'{a_prefix}_xg_for_roll5', 1.0)),
                'away_xg_against_roll5':     float(a_stats.get(f'{a_prefix}_xg_against_roll5', 1.2)),
                'away_passes_for_roll5':     float(a_stats.get(f'{a_prefix}_passes_for_roll5', 400)),
                'away_passes_against_roll5': float(a_stats.get(f'{a_prefix}_passes_against_roll5', 400)),
                'away_possession_roll5':     float(a_stats.get(f'{a_prefix}_possession_roll5', 0.5)),  # 0–1
                'away_points_roll5':         float(a_stats.get(f'{a_prefix}_points_roll5', 1.2))
            }

            # --- UI-only possession in % (model still uses 0–1) ---
            home_poss_pct = (
                input_data['home_possession_roll5'] * 100 
                if input_data['home_possession_roll5'] <= 1 
                else input_data['home_possession_roll5']
            )
            away_poss_pct = (
                input_data['away_possession_roll5'] * 100 
                if input_data['away_possession_roll5'] <= 1 
                else input_data['away_possession_roll5']
            )

        except Exception as e:
            st.error(f"Error extracting features: {e}")
            with st.expander("🔍 Debug Info"):
                st.write("Home stats columns:", h_stats.index.tolist())
                st.write("Away stats columns:", a_stats.index.tolist())
            st.stop()
        
        # Display matchup info
        st.markdown("### 📊 Team Comparison")
        
        # Stats comparison grid
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.markdown(f"#### 🏠 {home_team}")
            st.metric("Attack (xG/game)", f"{input_data['home_xg_for_roll5']:.2f}")
            st.metric("Defense (xG conceded)", f"{input_data['home_xg_against_roll5']:.2f}")
            st.metric("Possession", f"{home_poss_pct:.1f}%")
            st.metric("Form (pts/game)", f"{input_data['home_points_roll5']:.2f}")
        
        with col2:
            st.markdown("#### ⚖️ vs")
            
            # xG Advantage
            xg_diff = input_data['home_xg_for_roll5'] - input_data['away_xg_for_roll5']
            if abs(xg_diff) > 0.15:
                adv_team = "🏠" if xg_diff > 0 else "✈️"
                st.metric("Attack", adv_team, f"{abs(xg_diff):.2f}")
            else:
                st.metric("Attack", "⚖️", "Even")
            
            # Defense Advantage (lower is better)
            def_diff = input_data['away_xg_against_roll5'] - input_data['home_xg_against_roll5']
            if abs(def_diff) > 0.15:
                def_team = "🏠" if def_diff > 0 else "✈️"
                st.metric("Defense", def_team, f"{abs(def_diff):.2f}")
            else:
                st.metric("Defense", "⚖️", "Even")
            
            # Form
            form_diff = input_data['home_points_roll5'] - input_data['away_points_roll5']
            if abs(form_diff) > 0.3:
                form_team = "🏠" if form_diff > 0 else "✈️"
                st.metric("Form", form_team, f"{abs(form_diff):.2f}")
            else:
                st.metric("Form", "⚖️", "Even")
        
        with col3:
            st.markdown(f"#### ✈️ {away_team}")
            st.metric("Attack (xG/game)", f"{input_data['away_xg_for_roll5']:.2f}")
            st.metric("Defense (xG conceded)", f"{input_data['away_xg_against_roll5']:.2f}")
            st.metric("Possession", f"{away_poss_pct:.1f}%")
            st.metric("Form (pts/game)", f"{input_data['away_points_roll5']:.2f}")

    # ========== MANUAL MODE ==========
    else:
        st.markdown("### 🎮 Configure Match Simulation")
        st.info("💡 Stats are cross-linked: what one team does affects what the opponent faces")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏠 Home Team")
            home_team_name = st.text_input("Team Name", "Home FC", key="h_name")
            h_xg = st.slider("Attack (xG/game)", 0.0, 4.0, 1.5, 0.1, key="h_xg")
            h_poss = st.slider("Possession %", 20, 80, 55, 5, key="h_poss")
            h_points = st.slider("Recent Form (pts/game)", 0.0, 3.0, 1.5, 0.1, key="h_pts")
            
            h_passes = int(h_poss * 9.5)
            st.caption(f"Implied passes: {h_passes}")

        with col2:
            st.markdown("#### ✈️ Away Team")
            away_team_name = st.text_input("Team Name", "Away United", key="a_name")
            a_xg = st.slider("Attack (xG/game)", 0.0, 4.0, 1.2, 0.1, key="a_xg")
            
            # a_poss = 100 - h_poss
            a_poss = st.slider("Possession %", 20, 80, 55, 5, key="a_poss")
            a_passes = int(a_poss * 9.5)
            
            
            st.caption(f"Implied passes: {a_passes}")
            
            a_points = st.slider("Recent Form (pts/game)", 0.0, 3.0, 1.2, 0.1, key="a_pts")

        input_data = {
            'home_xg_for_roll5':         h_xg,
            'home_xg_against_roll5':     a_xg,
            'home_passes_for_roll5':     h_passes,
            'home_passes_against_roll5': a_passes,
            'home_possession_roll5':     h_poss,
            'home_points_roll5':         h_points,
            
            'away_xg_for_roll5':         a_xg,
            'away_xg_against_roll5':     h_xg,
            'away_passes_for_roll5':     a_passes,
            'away_passes_against_roll5': h_passes,
            'away_possession_roll5':     a_poss,
            'away_points_roll5':         a_points
        }

    # ========== PREDICTION WITH INSIGHTS ==========
    st.markdown("---")
    
    if st.button("🔮 Predict Match Result", type="primary", use_container_width=True):
        with st.spinner("Analyzing team stats..."):
            try:
                features_df = pd.DataFrame([input_data])
                features_df = features_df[match_features]
                features_scaled = match_scaler.transform(features_df)
                
                prediction_idx = match_model.predict(features_scaled)[0]
                probabilities = match_model.predict_proba(features_scaled)[0]
                
                result_map = {
                    0: f"{away_team_name} Win",
                    1: "Draw",
                    2: f"{home_team_name} Win"
                }
                
                final_result = result_map[prediction_idx]
                
                st.markdown("---")
                st.markdown("## 🎯 Prediction Results")
                
                if prediction_idx == 2:
                    st.success(f"### 🏆 Predicted: **{final_result}**")
                elif prediction_idx == 0:
                    st.error(f"### 🏆 Predicted: **{final_result}**")
                else:
                    st.warning(f"### ⚖️ Predicted: **{final_result}**")
                
                # Confidence breakdown
                st.markdown("### 📊 Confidence Breakdown")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(f"{away_team_name} Win", f"{probabilities[0]*100:.1f}%")
                with col2:
                    st.metric("Draw", f"{probabilities[1]*100:.1f}%")
                with col3:
                    st.metric(f"{home_team_name} Win", f"{probabilities[2]*100:.1f}%")
                
                # Visual
                chart_data = pd.DataFrame({
                    "Outcome": [f"{away_team_name} Win", "Draw", f"{home_team_name} Win"],
                    "Probability": probabilities * 100
                }).set_index("Outcome")
                st.bar_chart(chart_data, height=300)
                
                # INSIGHTS
                st.markdown("### 💡 Key Insights")
                insights, warnings = generate_insights(input_data, probabilities, home_team_name, away_team_name)
                
                for insight in insights:
                    st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)
                
                for warning in warnings:
                    st.markdown(f"<div class='warning-box'>{warning}</div>", unsafe_allow_html=True)
                
                # Feature importance explanation
                with st.expander("🔍 What Drives This Prediction?"):
                    st.markdown("""
                    The model considers:
                    - **Recent Form** (points per game over last 5 matches)
                    - **Attack Strength** (expected goals created)
                    - **Defensive Stability** (expected goals conceded)
                    - **Possession Control** (% of ball control)
                    - **Passing Volume** (build-up play quality)
                    
                    Teams with better attack, stronger defense, and good form are favored.
                    """)
                    
                    st.dataframe(pd.DataFrame([input_data]).T.rename(columns={0: "Value"}), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.exception(e)

# ==============================================================================
# PAGE 2: PLAYER SCORER (ENHANCED)
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
        st.markdown("Predict the probability of a player scoring in their next match based on recent performance")
        
        st.markdown("### Select Player")
        
        # Add team filter for easier navigation
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if 'team' in player_data.columns:
                teams = sorted(player_data['team'].unique())
                selected_team = st.selectbox("Filter by team (optional)", ["All Teams"] + teams)
                
                if selected_team != "All Teams":
                    filtered_players = player_data[player_data['team'] == selected_team]['player_name'].unique()
                else:
                    filtered_players = player_data['player_name'].unique()
            else:
                filtered_players = player_data['player_name'].unique()
        
        with col2:
            player_name = st.selectbox(
                "Choose a player", 
                sorted(filtered_players),
                key="player_select"
            )
        
        p_stats = player_data[player_data['player_name'] == player_name].iloc[-1]
        
        st.markdown("### 📊 Recent Performance (Last 5 Games)")
        
        goals_val = float(p_stats.get('goals_scored_roll5', 0))
        shots_val = float(p_stats.get('shots_per_90_roll5', 0))
        mins_val = float(p_stats.get('minutes_played_roll5', 0))
        xg_val = float(p_stats.get('xg_per_90_roll5', 0))

        # Enhanced metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Goals/Game", f"{goals_val:.2f}", 
                     "Hot" if goals_val > 0.5 else "Cold" if goals_val == 0 else "Average")
        with col2:
            st.metric("Shots/90", f"{shots_val:.2f}",
                     "High" if shots_val > 3.5 else "Low" if shots_val < 2 else "Normal")
        with col3:
            st.metric("Minutes/Game", f"{int(mins_val)}",
                     "Starter" if mins_val > 70 else "Sub")
        with col4:
            st.metric("xG/90", f"{xg_val:.2f}",
                     "Elite" if xg_val > 0.5 else "Poor" if xg_val < 0.2 else "Average")
        
        # Additional context
        if goals_val > 0 and shots_val > 0:
            shots_per_goal = (shots_val * (mins_val / 90)) / goals_val if goals_val > 0 else 0
            st.caption(f"⚽ Current streak: {shots_per_goal:.1f} shots per goal")
        
        if st.button("🎯 Calculate Goal Probability", type="primary", use_container_width=True):
            try:
                input_features = pd.DataFrame([{
                    'xg_per_90_roll5': xg_val,
                    'shots_per_90_roll5': shots_val,
                    'goals_scored_roll5': goals_val,
                    'minutes_played_roll5': mins_val
                }])
                
                goal_prob = player_model.predict_proba(input_features)[0][1]
                
                st.markdown("---")
                st.markdown("## 🎯 Prediction Result")
                
                # Main probability display
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    st.metric(
                        label="Goal Probability", 
                        value=f"{goal_prob*100:.1f}%",
                        delta="High Confidence" if goal_prob > 0.4 else "Low Confidence"
                    )
                    
                    st.progress(min(goal_prob, 1.0))
                    
                    # Odds conversion
                    if goal_prob > 0.01:
                        odds = 1 / goal_prob
                        st.caption(f"📊 Equivalent to {odds:.1f}/1 odds")
                
                # Visual interpretation
                if goal_prob > 0.5:
                    st.balloons()
                    st.markdown("<div class='success-box'><strong>🔥 VERY LIKELY TO SCORE!</strong><br>This player is in excellent form and the model expects a goal.</div>", unsafe_allow_html=True)
                elif goal_prob > 0.35:
                    st.markdown("<div class='insight-box'><strong>⚡ GOOD CHANCE</strong><br>Strong possibility of scoring based on recent performances.</div>", unsafe_allow_html=True)
                elif goal_prob > 0.2:
                    st.markdown("<div class='warning-box'><strong>❄️ MODERATE CHANCE</strong><br>Possible but not highly likely based on current form.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='warning-box'><strong>📉 UNLIKELY</strong><br>Low probability - would need significant improvement or luck.</div>", unsafe_allow_html=True)
                
                # DETAILED INSIGHTS
                st.markdown("### 💡 Performance Analysis")
                insights, warnings = generate_player_insights(xg_val, shots_val, goals_val, mins_val, goal_prob)
                
                if insights:
                    st.markdown("#### ✅ Strengths")
                    for insight in insights:
                        st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)
                
                if warnings:
                    st.markdown("#### ⚠️ Concerns")
                    for warning in warnings:
                        st.markdown(f"<div class='warning-box'>{warning}</div>", unsafe_allow_html=True)
                
                # Comparison to benchmarks
                with st.expander("📈 Compare to Benchmarks"):
                    st.markdown("""
                    **Elite Scorer Benchmarks:**
                    - Goals per game: > 0.7
                    - Shots per 90: > 4.0
                    - xG per 90: > 0.6
                    - Conversion rate: > 20%
                    
                    **Average Benchmarks:**
                    - Goals per game: 0.3 - 0.5
                    - Shots per 90: 2.5 - 3.5
                    - xG per 90: 0.3 - 0.5
                    - Conversion rate: 12-18%
                    """)
                    
                    # Player vs benchmarks
                    benchmark_data = pd.DataFrame({
                        "Metric": ["Goals/Game", "Shots/90", "xG/90"],
                        "Player": [goals_val, shots_val, xg_val],
                        "Elite": [0.7, 4.0, 0.6],
                        "Average": [0.4, 3.0, 0.4]
                    })
                    st.dataframe(benchmark_data, use_container_width=True)
                
                # What would improve chances
                with st.expander("🎯 How to Increase Scoring Probability"):
                    improvements = []
                    
                    if shots_val < 3.0:
                        improvements.append("📈 **Take more shots** - Currently below average volume")
                    if xg_val < 0.4:
                        improvements.append("🎯 **Get into better positions** - Shot quality is low")
                    if mins_val < 60:
                        improvements.append("⏱️ **Play more minutes** - Limited time on pitch")
                    if goals_val < 0.3 and shots_val > 0:
                        improvements.append("🎪 **Improve finishing** - Conversion rate below expectations")
                    
                    if improvements:
                        for improvement in improvements:
                            st.markdown(f"- {improvement}")
                    else:
                        st.success("✅ Player is performing well across all metrics!")
                
                # Feature breakdown
                with st.expander("🔬 Model Input Details"):
                    st.markdown("""
                    The model evaluates:
                    1. **Expected Goals (xG)** - Quality of chances
                    2. **Shot Volume** - Frequency of attempts
                    3. **Recent Goals** - Actual scoring form
                    4. **Playing Time** - Minutes on the pitch
                    """)
                    
                    feature_df = pd.DataFrame([{
                        "Feature": ["xG per 90", "Shots per 90", "Goals (recent)", "Minutes played"],
                        "Value": [f"{xg_val:.2f}", f"{shots_val:.2f}", f"{goals_val:.2f}", f"{mins_val:.0f}"],
                        "Impact": ["High", "High", "Very High", "Medium"]
                    }])
                    st.table(feature_df)
                    
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
        
        st.markdown("### 🎬 Analyzed Match Footage")
        with open(VIDEO_OUTPUT_PATH, 'rb') as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)
        
        st.markdown("### 📊 Detection Legend")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
            <h4>⚪ White Boxes</h4>
            <p>Home Team Players</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card'>
            <h4>🟤 Brown Ball</h4>
            <p>Away Team Players</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='metric-card'>
            <h4>🟡 Yellow Boxes</h4>
            <p>Ball Detection</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Features")
        st.markdown("""
        - **Real-time Object Detection**: YOLOv8-powered tracking
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