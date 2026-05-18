# ═══════════════════════════════════════════════════════════════
# NFL Monte Carlo Simulation — Streamlit Dashboard
# Author: Colin McBane
# Date: 2026
# ═══════════════════════════════════════════════════════════════

from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import nfl_data_py as nfl

BASE_DIR = Path(__file__).resolve().parent

# ── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="NFL Monte Carlo Simulation",
    page_icon="🏈",
    layout="wide"
)

# ── Title & Description ───────────────────────────────────────
st.title("🏈 NFL 2024 Regular Season Monte Carlo Simulation")
st.subheader("Madden 25 Ratings vs Vegas Odds — Which Predicts Better?")
st.markdown("""
This app simulates the 2024 NFL regular season 10,000 times using two methods:
- **Vegas Odds** — Preseason Super Bowl odds from Sports Illustrated (July 2024)
- **Madden 25 Ratings** — Team overall ratings at launch (August 2024)

We compare both methods against actual 2024 win totals to determine 
which is a more accurate predictor of NFL team success.
""")

st.divider()

# ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    """
    Cache data so it doesn't reload every time user interacts
    with the dashboard — makes the app much faster
    """
    df_path = BASE_DIR / 'nfl_data.xlsx'
    df = pd.read_excel(df_path)
    return df

@st.cache_data
def load_schedule():
    """
    Pull and cache the 2024 NFL regular season schedule
    Maps team abbreviations to full names to match Excel sheet
    """
    schedule = nfl.import_schedules([2024])
    schedule = schedule[schedule['game_type'] == 'REG']
    schedule = schedule[['week', 'home_team', 'away_team']]

    team_map = {
        'KC': 'Kansas City Chiefs', 'SF': 'San Francisco 49ers',
        'BAL': 'Baltimore Ravens', 'DET': 'Detroit Lions',
        'PHI': 'Philadelphia Eagles', 'HOU': 'Houston Texans',
        'CIN': 'Cincinnati Bengals', 'BUF': 'Buffalo Bills',
        'DAL': 'Dallas Cowboys', 'NYJ': 'New York Jets',
        'GB': 'Green Bay Packers', 'MIA': 'Miami Dolphins',
        'ATL': 'Atlanta Falcons', 'LA': 'Los Angeles Rams',
        'CLE': 'Cleveland Browns', 'CHI': 'Chicago Bears',
        'LAC': 'Los Angeles Chargers', 'JAX': 'Jacksonville Jaguars',
        'PIT': 'Pittsburgh Steelers', 'IND': 'Indianapolis Colts',
        'SEA': 'Seattle Seahawks', 'TB': 'Tampa Bay Buccaneers',
        'ARI': 'Arizona Cardinals', 'DEN': 'Denver Broncos',
        'LV': 'Las Vegas Raiders', 'TEN': 'Tennessee Titans',
        'WAS': 'Washington Commanders', 'NO': 'New Orleans Saints',
        'CAR': 'Carolina Panthers', 'NYG': 'New York Giants',
        'NE': 'New England Patriots', 'MIN': 'Minnesota Vikings'
    }

    schedule['home_team'] = schedule['home_team'].map(team_map)
    schedule['away_team'] = schedule['away_team'].map(team_map)
    return schedule

@st.cache_data
def run_simulation(_df, _schedule, n_simulations=10000):
    np.random.seed(42)
    """
    Runs both Vegas and Madden simulations and returns results.
    Cached so it only runs once per session — not on every interaction.
    Uses pre-built dictionary lookups for 120x speed improvement
    over the original iterrows() approach.
    """
    # Calculate Vegas strength from Super Bowl odds
    # Normalize to remove sportsbook vig (house edge)
    _df['Raw_Implied_Prob'] = 100 / (_df['SB_Odds'] + 100)
    _df['Vegas_Strength'] = _df['Raw_Implied_Prob'] / _df['Raw_Implied_Prob'].sum()

    # Calculate Madden strength using softmax normalization
    # Dividing by 100 prevents astronomically large exponents
    madden_exp = np.exp(_df['Madden_Overall'] / 100)
    _df['Madden_Strength'] = madden_exp / madden_exp.sum()

    def simulate(strength_col):
        """
        Simulates full season n_simulations times for one strength method.
        Pre-builds lookup structures for maximum performance.
        """
        # Convert to list once — much faster than iterrows
        games = list(zip(_schedule['home_team'], _schedule['away_team']))
        # Dictionary lookup instead of dataframe search on every game
        strengths = dict(zip(_df['Team'], _df[strength_col]))
        total_wins = {team: 0 for team in _df['Team']}

        for _ in range(n_simulations):
            for home, away in games:
                s1, s2 = strengths[home], strengths[away]
                # Win probability = team strength / combined strength
                if np.random.random() < s1 / (s1 + s2):
                    total_wins[home] += 1
                else:
                    total_wins[away] += 1

        return {team: total_wins[team] / n_simulations for team in total_wins}

    vegas_wins = simulate('Vegas_Strength')
    madden_wins = simulate('Madden_Strength')

    # Build results dataframe with predictions and errors
    results = pd.DataFrame({
        'Team': _df['Team'],
        'SB_Odds': _df['SB_Odds'],
        'Madden_Overall': _df['Madden_Overall'],
        'Vegas_Predicted': [round(vegas_wins[t], 1) for t in _df['Team']],
        'Madden_Predicted': [round(madden_wins[t], 1) for t in _df['Team']],
        'Actual_Wins': _df['Actual_Wins']
    })

    # Absolute error so over and under predictions don't cancel out
    results['Vegas_Error'] = abs(results['Vegas_Predicted'] - results['Actual_Wins'])
    results['Madden_Error'] = abs(results['Madden_Predicted'] - results['Actual_Wins'])

    return results, _df

# ── Run Everything ────────────────────────────────────────────
df = load_data()
schedule = load_schedule()

with st.spinner("Running 10,000 season simulations..."):
    results, df = run_simulation(df, schedule)

results_sorted = results.sort_values('Actual_Wins', ascending=True)

# ── Accuracy Metrics ──────────────────────────────────────────
# Calculate MAE for all teams and extreme teams separately
vegas_mae = results['Vegas_Error'].mean()
madden_mae = results['Madden_Error'].mean()
non_500 = results[~results['Actual_Wins'].between(7, 10)]
vegas_mae_extreme = non_500['Vegas_Error'].mean()
madden_mae_extreme = non_500['Madden_Error'].mean()

# Display as metric cards at the top of the dashboard
col1, col2, col3, col4 = st.columns(4)
col1.metric("Vegas MAE (all teams)", f"{vegas_mae:.2f} wins")
col2.metric("Madden MAE (all teams)", f"{madden_mae:.2f} wins")
col3.metric("Vegas MAE (extreme teams)", f"{vegas_mae_extreme:.2f} wins")
col4.metric("Madden MAE (extreme teams)", f"{madden_mae_extreme:.2f} wins")

st.divider()

# ── Chart 1: Predicted vs Actual ──────────────────────────────
st.subheader("Predicted vs Actual Wins — All 32 Teams")

fig1 = go.Figure()
fig1.add_trace(go.Bar(
    name='Actual Wins',
    x=results_sorted['Team'],
    y=results_sorted['Actual_Wins'],
    marker_color='#2ecc71'
))
fig1.add_trace(go.Bar(
    name='Vegas Predicted',
    x=results_sorted['Team'],
    y=results_sorted['Vegas_Predicted'],
    marker_color='#3498db'
))
fig1.add_trace(go.Bar(
    name='Madden Predicted',
    x=results_sorted['Team'],
    y=results_sorted['Madden_Predicted'],
    marker_color='#e74c3c'
))
fig1.update_layout(
    title='2024 NFL Regular Season: Predicted vs Actual Wins',
    xaxis_title='Team',
    yaxis_title='Wins',
    barmode='group',
    xaxis_tickangle=-45,
    template='plotly_dark',
    height=500
)
st.plotly_chart(fig1, use_container_width=True)

# ── Chart 2: Error by Team ────────────────────────────────────
st.subheader("Prediction Error by Team (Lower = More Accurate)")

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    name='Vegas Error',
    x=results_sorted['Team'],
    y=results_sorted['Vegas_Error'],
    marker_color='#3498db'
))
fig2.add_trace(go.Bar(
    name='Madden Error',
    x=results_sorted['Team'],
    y=results_sorted['Madden_Error'],
    marker_color='#e74c3c'
))
fig2.update_layout(
    title='Prediction Error by Team (Lower = More Accurate)',
    xaxis_title='Team',
    yaxis_title='Absolute Error (Wins)',
    barmode='group',
    xaxis_tickangle=-45,
    template='plotly_dark',
    height=500
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Sensitivity Analysis Charts ───────────────────────────────
st.subheader("Sensitivity Analysis — Where the Real Gap Emerges")
st.markdown("Removing average teams shows where each method truly differs on elite and bad teams.")

non_500_sorted = non_500.sort_values('Actual_Wins', ascending=True)

# Chart 3: Predicted vs Actual (extreme teams only)
fig3 = go.Figure()
fig3.add_trace(go.Bar(
    name='Actual Wins',
    x=non_500_sorted['Team'],
    y=non_500_sorted['Actual_Wins'],
    marker_color='#2ecc71'
))
fig3.add_trace(go.Bar(
    name='Vegas Predicted',
    x=non_500_sorted['Team'],
    y=non_500_sorted['Vegas_Predicted'],
    marker_color='#3498db'
))
fig3.add_trace(go.Bar(
    name='Madden Predicted',
    x=non_500_sorted['Team'],
    y=non_500_sorted['Madden_Predicted'],
    marker_color='#e74c3c'
))
fig3.update_layout(
    title='Elite & Bottom Teams Only: Where Predictions Matter Most (Excluding 7-10 Win Teams)',
    xaxis_title='Team',
    yaxis_title='Wins',
    barmode='group',
    xaxis_tickangle=-45,
    template='plotly_dark',
    height=500
)
st.plotly_chart(fig3, use_container_width=True)

# Chart 4: Error by team (extreme teams only)
fig4 = go.Figure()
fig4.add_trace(go.Bar(
    name='Vegas Error',
    x=non_500_sorted['Team'],
    y=non_500_sorted['Vegas_Error'],
    marker_color='#3498db'
))
fig4.add_trace(go.Bar(
    name='Madden Error',
    x=non_500_sorted['Team'],
    y=non_500_sorted['Madden_Error'],
    marker_color='#e74c3c'
))
fig4.update_layout(
    title=f'Prediction Error on Extreme Teams — Vegas MAE: {vegas_mae_extreme:.2f} vs Madden MAE: {madden_mae_extreme:.2f} (Lower = Better)',
    xaxis_title='Team',
    yaxis_title='Absolute Error (Wins)',
    barmode='group',
    xaxis_tickangle=-45,
    template='plotly_dark',
    height=500
)
st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ── Key Findings & Analysis ───────────────────────────────────
st.subheader("Key Findings & Analysis")

st.markdown("### 1. Main Finding")
st.markdown(f"""
Vegas odds outperformed Madden 25 ratings as a predictor of 2024 NFL regular season wins,
with a Mean Absolute Error of **{vegas_mae:.2f} wins** compared to **{madden_mae:.2f} wins**
for Madden. While the overall margin is small, the gap reveals a meaningful difference
in how each method models team quality.
""")

st.markdown("### 2. Designed to Predict vs Designed to Play")
st.markdown("""
The most fundamental reason Vegas odds outperform Madden ratings is **purpose**.
Vegas odds exist for one reason — to accurately predict outcomes. Sportsbooks employ
analysts, algorithms, and vast amounts of historical data specifically to price the
probability of every team winning. They lose money when they are wrong, creating a
direct financial incentive for accuracy.

Madden ratings exist to make a fun, balanced video game. EA Sports designs overalls
to create competitive gameplay — a 99 overall team playing a 76 overall team every
game would be boring. The compression of ratings from 76-92 is a deliberate design
choice for entertainment, not prediction. Comparing the two as predictive models is
inherently unfair to Madden, but reveals exactly why market-based probability models
outperform subjective rating systems in forecasting real world outcomes.
""")

st.markdown("### 3. Why Madden Underperforms")
st.markdown("""
Madden overall ratings range from 76-92, a compressed scale that doesn't create meaningful
separation between teams. As a result, every team was predicted between 8-9 wins regardless
of actual quality — essentially a 50/50 model. This is a fundamental limitation of using
a video game rating system designed for gameplay balance as a predictive analytics tool.
""")

st.markdown("### 4. The Sensitivity Analysis Finding")
st.markdown(f"""
The overall MAE numbers are close partly because both methods predict average teams similarly
well. When removing teams with 7-10 actual wins, the gap grows dramatically —
Vegas MAE: **{vegas_mae_extreme:.2f}** vs Madden MAE: **{madden_mae_extreme:.2f}**,
a difference of **{madden_mae_extreme - vegas_mae_extreme:.2f} wins per team**.
This confirms Vegas odds are significantly better at identifying truly elite and truly bad teams.
""")

st.markdown("### 5. Where Each Model Failed")
st.markdown("""
Vegas's biggest miss was the **San Francisco 49ers** — predicted 13+ wins but finished 6-11
due to season-ending injuries to Christian McCaffrey, Brandon Aiyuk, and others.
Madden's biggest miss was the **Minnesota Vikings** — rated as an average team but dramatically
overperformed as Sam Darnold played at an elite level while JJ McCarthy's injury forced
a quarterback change that ultimately benefited the team.
""")

st.markdown("### 6. Broader Conclusion")
st.markdown("""
These results support the **Efficient Market Hypothesis** applied to sports betting.
Vegas odds incorporate real financial incentives, injury news, roster analysis, and
collective market wisdom — factors a static preseason video game rating cannot capture.
Neither model accounts for in-season variance, which represents the largest source of
prediction error for both methods.
""")

st.markdown("### 7. Technical Note")
st.markdown("""
The initial simulation using pandas `iterrows()` took over 10 minutes to run 10,000 seasons.
Optimizing with pre-built dictionary lookups and list conversion reduced runtime to under
5 seconds — a **120x performance improvement** demonstrating the importance of computational
efficiency in simulation modeling.
""")

st.divider()

# ── Full Results Table ────────────────────────────────────────
results_display = results.copy()
results_display['SB_Odds'] = '+' + results_display['SB_Odds'].astype(str)
results_display.columns = [
    'Team', 'SB Odds', 'Madden Overall',
    'Vegas Predicted', 'Madden Predicted',
    'Actual Wins', 'Vegas Error', 'Madden Error'
]

st.subheader("Full Results Table")
st.dataframe(
    results_display.sort_values('Actual Wins', ascending=False).reset_index(drop=True),
    use_container_width=True,
    hide_index=True
)

st.divider()

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
**Data Sources:**
- Super Bowl Odds: Sports Illustrated (July 25, 2024)
- Madden 25 Team Overalls: Madden NFL 25 launch ratings (August 2024)
- 2024 NFL Schedule: nfl_data_py (nflverse)
- Actual Win Totals: Pro Football Reference
""")