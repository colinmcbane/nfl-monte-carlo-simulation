# ═══════════════════════════════════════════════════════════════
# NFL Monte Carlo Simulation
# Comparing Madden 25 Ratings vs Vegas Odds as season predictors
# Author: Colin McBane
# Date: 2025
# ═══════════════════════════════════════════════════════════════

# Data Sources:
# Super Bowl Odds: Sports Illustrated (July 25, 2024)
# Madden 25 Overalls: Madden NFL 25 launch ratings (August 2024)
# Actual Wins: Pro Football Reference
# 2024 NFL Schedule: nfl_data_py (https://github.com/nflverse/nfl_data_py)

from pathlib import Path

import numpy as np
import pandas as pd
import nfl_data_py as nfl
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent

# ── Data Loading ──────────────────────────────────────────────────────
# Load team data from Excel spreadsheet containing Madden overalls,
# Vegas odds, and actual 2024 win totals for all 32 NFL teams
df = pd.read_excel(BASE_DIR / 'nfl_data.xlsx')

# ── Vegas Odds Conversion ─────────────────────────────────────────────
# Convert American moneyline odds to raw implied probability
# Formula: 100 / (odds + 100) for positive odds
# Example: +600 odds → 100 / 700 = 14.3% chance of winning Super Bowl
# Sportsbooks build in a house edge so raw probs sum to more than 100%
# Dividing by the total removes the vig — all 32 teams sum to exactly 1.0
df['Raw_Implied_Prob'] = 100 / (df['SB_Odds'] + 100)
df['Vegas_Strength'] = df['Raw_Implied_Prob'] / df['Raw_Implied_Prob'].sum()

# ── Madden Strength Calculation ───────────────────────────────────────
# Madden overalls are ratings (76-92), not probabilities
# We use softmax-style normalization to convert them to win probabilities
# Teams with higher overalls get exponentially more strength
# This mirrors how Madden models team quality gaps
# Dividing by 100 keeps exponents manageable and prevents
# astronomical differences between teams
madden_exp = np.exp(df['Madden_Overall'] / 100)
df['Madden_Strength'] = madden_exp / madden_exp.sum()

# ── Schedule Import & Team Mapping ────────────────────────────────────
# Pull the 2024 regular season schedule using nfl_data_py
# NFL seasons are named by the year they start
# (2024 = September 2024 — January 2025)
schedule = nfl.import_schedules([2024])

# Keep only regular season games and relevant columns
schedule = schedule[schedule['game_type'] == 'REG']
schedule = schedule[['week', 'home_team', 'away_team']]

# Map abbreviations to full team names to match our Excel sheet
# nfl_data_py uses abbreviations (KC, SF) but our dataframe uses full names
team_map = {
    'KC': 'Kansas City Chiefs',
    'SF': 'San Francisco 49ers',
    'BAL': 'Baltimore Ravens',
    'DET': 'Detroit Lions',
    'PHI': 'Philadelphia Eagles',
    'HOU': 'Houston Texans',
    'CIN': 'Cincinnati Bengals',
    'BUF': 'Buffalo Bills',
    'DAL': 'Dallas Cowboys',
    'NYJ': 'New York Jets',
    'GB': 'Green Bay Packers',
    'MIA': 'Miami Dolphins',
    'ATL': 'Atlanta Falcons',
    'LA': 'Los Angeles Rams',
    'CLE': 'Cleveland Browns',
    'CHI': 'Chicago Bears',
    'LAC': 'Los Angeles Chargers',
    'JAX': 'Jacksonville Jaguars',
    'PIT': 'Pittsburgh Steelers',
    'IND': 'Indianapolis Colts',
    'SEA': 'Seattle Seahawks',
    'TB': 'Tampa Bay Buccaneers',
    'ARI': 'Arizona Cardinals',
    'DEN': 'Denver Broncos',
    'LV': 'Las Vegas Raiders',
    'TEN': 'Tennessee Titans',
    'WAS': 'Washington Commanders',
    'NO': 'New Orleans Saints',
    'CAR': 'Carolina Panthers',
    'NYG': 'New York Giants',
    'NE': 'New England Patriots',
    'MIN': 'Minnesota Vikings'
}

# Apply mapping to both home and away team columns
schedule['home_team'] = schedule['home_team'].map(team_map)
schedule['away_team'] = schedule['away_team'].map(team_map)

# ── Game Simulation Function ──────────────────────────────────────────
def simulate_game(team1, team2, strength_col, team_df):
    """
    Simulates a single NFL game between two teams.
    Uses relative team strength to calculate win probability.
    Note: Season simulation inlines this logic directly for
    performance — calling this function 2.72 million times
    adds significant overhead.

    Args:
        team1: home team name
        team2: away team name
        strength_col: 'Vegas_Strength' or 'Madden_Strength'
        team_df: dataframe containing team strengths

    Returns:
        winner: name of the winning team
    """
    # Look up each team's strength from the dataframe
    strength1 = df.loc[df['Team'] == team1, strength_col].values[0]
    strength2 = df.loc[df['Team'] == team2, strength_col].values[0]
    total_strength = strength1 + strength2

    # Win probability = team strength / combined strength
    # Stronger team wins more often but can still lose —
    # this randomness is what makes Monte Carlo valuable
    prob_team1_wins = strength1 / total_strength

    # If random number falls below win probability, team1 wins
    if np.random.random() < prob_team1_wins:
        return team1
    else:
        return team2

# ── Monte Carlo Engine ────────────────────────────────────────────────
def simulate_season(schedule, team_df, strength_col, n_simulations=10000):
    """
    Simulates the full NFL regular season n_simulations times.
    Uses pre-built lookup structures for speed instead of
    searching the dataframe on every game.
    Initial iterrows() approach took 10+ minutes — this runs
    in under 5 seconds (120x performance improvement).

    Args:
        schedule: dataframe of all 272 NFL games
        team_df: dataframe with team strengths
        strength_col: 'Vegas_Strength' or 'Madden_Strength'
        n_simulations: number of seasons to simulate (default 10,000)

    Returns:
        avg_wins: dictionary of average wins per team
    """
    # Convert schedule to list of tuples ONCE before simulating
    # Much faster than calling iterrows() on every simulation
    games = list(zip(schedule['home_team'], schedule['away_team']))

    # Pre-build strength dictionary for instant lookup
    # Avoids searching the full dataframe 2.72 million times
    strengths = dict(zip(team_df['Team'], team_df[strength_col]))

    total_wins = {team: 0 for team in team_df['Team']}

    for sim in range(n_simulations):
        for home, away in games:
            s1 = strengths[home]
            s2 = strengths[away]
            if np.random.random() < s1 / (s1 + s2):
                total_wins[home] += 1
            else:
                total_wins[away] += 1

    # Average wins = total wins / number of simulations
    avg_wins = {team: total_wins[team] / n_simulations
                for team in total_wins}
    return avg_wins

# ── Run Simulations ───────────────────────────────────────────────────
# Run both simulations and compare results against actual 2024 wins
print("\nRunning Vegas simulation (10,000 seasons)...")
vegas_wins = simulate_season(schedule, df, 'Vegas_Strength')

print("Running Madden simulation (10,000 seasons)...")
madden_wins = simulate_season(schedule, df, 'Madden_Strength')
print("Done!")

# ── Results & Accuracy Analysis ───────────────────────────────────────
# Build a dataframe comparing all 32 teams
# Vegas predicted, Madden predicted, and actual wins side by side
results = pd.DataFrame({
    'Team': df['Team'],
    'Vegas_Predicted': [round(vegas_wins[t], 1) for t in df['Team']],
    'Madden_Predicted': [round(madden_wins[t], 1) for t in df['Team']],
    'Actual_Wins': df['Actual_Wins']
})

# Calculate error for each method
# Absolute value so overestimates and underestimates don't cancel out
results['Vegas_Error'] = abs(results['Vegas_Predicted'] - results['Actual_Wins'])
results['Madden_Error'] = abs(results['Madden_Predicted'] - results['Actual_Wins'])

# Sort by actual wins so best teams are at top
results = results.sort_values('Actual_Wins', ascending=False)

# Print full results table
print("\nFULL RESULTS TABLE")
print(results.to_string(index=False))

# Mean Absolute Error (MAE) — average error across all 32 teams
# MAE chosen because it expresses error in same units as outcome (wins)
# making results immediately interpretable. Lower is better.
vegas_mae = results['Vegas_Error'].mean()
madden_mae = results['Madden_Error'].mean()

print(f"\nACCURACY SUMMARY")
print(f"Vegas  Mean Absolute Error: {vegas_mae:.2f} wins")
print(f"Madden Mean Absolute Error: {madden_mae:.2f} wins")

if vegas_mae < madden_mae:
    print(f"\nVerdict: Vegas odds were more accurate by {madden_mae - vegas_mae:.2f} wins per team")
else:
    print(f"\nVerdict: Madden ratings were more accurate by {vegas_mae - madden_mae:.2f} wins per team")

# ── Sensitivity Analysis ──────────────────────────────────────────────
# Remove average teams (7-10 wins) to test accuracy on extremes
# Elite teams (11+ wins) and bad teams (6 or fewer wins) are where
# predictive models matter most — anyone can predict a .500 team
# This also removes the compression bias that narrows the MAE gap
non_500 = results[~results['Actual_Wins'].between(7, 10)]

vegas_mae_extreme = non_500['Vegas_Error'].mean()
madden_mae_extreme = non_500['Madden_Error'].mean()

print(f"\nSENSITIVITY ANALYSIS (excluding 7-10 win teams)")
print(f"Teams included: {len(non_500)}/32")
print(f"Vegas  MAE: {vegas_mae_extreme:.2f} wins")
print(f"Madden MAE: {madden_mae_extreme:.2f} wins")

if vegas_mae_extreme < madden_mae_extreme:
    print(f"Verdict: Vegas more accurate by {madden_mae_extreme - vegas_mae_extreme:.2f} wins")
else:
    print(f"Verdict: Madden more accurate by {vegas_mae_extreme - madden_mae_extreme:.2f} wins")

# ── Visualizations ────────────────────────────────────────────────────
# Plotly chosen for interactive hover tooltips — allows detailed
# exploration of individual team predictions vs static matplotlib charts

# Sort by actual wins for cleaner left-to-right visualization
results_sorted = results.sort_values('Actual_Wins', ascending=True)

# Chart 1: Predicted vs Actual Wins — all 32 teams
# Green = actual, Blue = Vegas predicted, Red = Madden predicted
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
    xaxis_tickangle=90,
    template='plotly_dark'
)

fig1.show()

# Chart 2: Prediction error by team
# Smaller bar = more accurate prediction
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
    xaxis_tickangle=90,
    template='plotly_dark'
)

fig2.show()

# Chart 3: Predicted vs Actual — extreme teams only
# Removing 7-10 win teams shows where each method truly differs
non_500_sorted = non_500.sort_values('Actual_Wins', ascending=True)

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
    xaxis_tickangle=90,
    template='plotly_dark'
)

fig3.show()

# Chart 4: Error by team — extreme teams only
# Shows the true gap between methods on teams that matter most
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
    xaxis_tickangle=90,
    template='plotly_dark'
)

fig4.show()