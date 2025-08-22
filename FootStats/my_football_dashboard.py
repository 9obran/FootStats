# Soccer player stats analyzer

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# for the math stuff
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns

class soccer_data_analyzer:
    """
    this class does all the soccer data stuff
    it makes fake data and does math on it
    """
    
    def __init__(self):
        """start everything up"""
        self.my_data = self.make_fake_data()
        self.important_stats = {}
        self.math_results = {}
        
    def make_fake_data(self):
        """
        makes up soccer player data that looks real
        returns a dataframe with all the player info
        """
        np.random.seed(42)  # so we get same results every time
        
        # make list of fake players
        player_names = []
        for i in range(1, 51):
            if i < 10:
                player_names.append(f"Player_0{i}")
            else:
                player_names.append(f"Player_{i}")
        
        game_numbers = []
        for i in range(1, 21):
            game_numbers.append(i)
        
        player_positions = ['Forward', 'Midfielder', 'Defender', 'Goalkeeper']
        
        all_data = []
        
        # make data for each player
        for player in player_names:
            # pick what position they play
            pos = np.random.choice(player_positions, p=[0.3, 0.4, 0.25, 0.05])
            skill_level = np.random.uniform(60, 90)  # how good they are
            
            # make stats for each game
            for game in game_numbers:
                # different positions score different amounts
                if pos == 'Forward':
                    goals_scored = np.random.poisson(0.5)
                    assists_made = np.random.poisson(0.3)
                    shots_taken = np.random.poisson(3.2)
                elif pos == 'Midfielder':
                    goals_scored = np.random.poisson(0.2)
                    assists_made = np.random.poisson(0.8)
                    shots_taken = np.random.poisson(1.8)
                elif pos == 'Defender':
                    goals_scored = np.random.poisson(0.1)
                    assists_made = np.random.poisson(0.2)
                    shots_taken = np.random.poisson(0.8)
                else:  # goalie
                    goals_scored = 0
                    assists_made = np.random.poisson(0.05)
                    shots_taken = 0
                
                # stuff everyone does
                good_passes = int(np.random.normal(35 + skill_level/3, 10))
                total_passes = int(good_passes / np.random.uniform(0.7, 0.95))
                km_ran = np.random.normal(8.5, 1.5)  # kilometers
                fast_runs = np.random.poisson(12)
                
                # how good they played (out of 10)
                game_rating = np.clip(np.random.normal(skill_level/12, 1.2), 4.0, 10.0)
                
                # put it all together
                game_data = {
                    'player_name': player,
                    'game_num': game,
                    'position': pos,
                    'goals': goals_scored,
                    'assists': assists_made,
                    'shots': shots_taken,
                    'good_passes': max(0, good_passes),
                    'total_passes': max(good_passes, total_passes),
                    'distance_km': max(0, km_ran),
                    'sprints': fast_runs,
                    'rating': round(game_rating, 1),
                    'minutes': np.random.choice([90, 85, 78, 45, 30], p=[0.6, 0.15, 0.1, 0.1, 0.05])
                }
                
                all_data.append(game_data)
        
        # make it into a dataframe
        df = pd.DataFrame(all_data)
        
        # calculate some extra stuff
        df['pass_percent'] = (df['good_passes'] / df['total_passes'] * 100).round(1)
        df['goals_and_assists'] = df['goals'] + df['assists']
        df['shot_percent'] = ((df['goals'] / df['shots'].replace(0, 1)) * 100).round(1)
        df['shot_percent'] = df['shot_percent'].fillna(0)
        
        return df
    
    def get_important_numbers(self):
        """
        figure out the most important stats from our data
        """
        # team totals
        all_goals = self.my_data['goals'].sum()
        all_assists = self.my_data['assists'].sum()
        avg_passes = self.my_data['pass_percent'].mean()
        avg_running = self.my_data['distance_km'].mean()
        
        # best players
        best_scorers = self.my_data.groupby('player_name')['goals'].sum().sort_values(ascending=False).head(5)
        best_assisters = self.my_data.groupby('player_name')['assists'].sum().sort_values(ascending=False).head(5)
        
        # stats by position
        pos_stats = self.my_data.groupby('position').agg({
            'goals': 'mean',
            'assists': 'mean',
            'pass_percent': 'mean',
            'rating': 'mean',
            'distance_km': 'mean'
        }).round(2)
        
        # save everything
        self.important_stats = {
            'total_goals': all_goals,
            'total_assists': all_assists,
            'pass_avg': round(avg_passes, 1),
            'distance_avg': round(avg_running, 1),
            'top_goal_scorers': best_scorers.to_dict(),
            'top_assist_makers': best_assisters.to_dict(),
            'position_numbers': pos_stats
        }
        
        return self.important_stats
    
    def do_math_analysis(self):
        """
        do fancy math to see what makes players good
        """
        # pick which stats to use
        stat_columns = ['goals', 'assists', 'pass_percent', 'distance_km', 'sprints']
        X_data = self.my_data[stat_columns]
        y_data = self.my_data['rating']
        
        # make the math model
        math_model = LinearRegression()
        math_model.fit(X_data, y_data)
        
        # get predictions
        predicted_ratings = math_model.predict(X_data)
        r2_number = r2_score(y_data, predicted_ratings)
        
        # save results
        coeffs = {}
        importance = {}
        for i, col in enumerate(stat_columns):
            coeffs[col] = math_model.coef_[i]
            importance[col] = abs(math_model.coef_[i])
        
        self.math_results = {
            'stat_names': stat_columns,
            'coefficients': coeffs,
            'intercept_val': math_model.intercept_,
            'r2_value': r2_number,
            'importance_scores': importance
        }
        
        return self.math_results
    
    def test_hypotheses(self):
        """
        test some ideas about soccer with statistics
        """
        # idea 1: forwards score more than others
        forward_goals = self.my_data[self.my_data['position'] == 'Forward']['goals']
        other_goals = self.my_data[self.my_data['position'] != 'Forward']['goals']
        
        t_stat, p_val = stats.ttest_ind(forward_goals, other_goals)
        
        # idea 2: good passers get better ratings
        corr_val, corr_p = stats.pearsonr(self.my_data['pass_percent'], self.my_data['rating'])
        
        test_results = {
            'forwards_test': {
                't_number': t_stat,
                'p_number': p_val,
                'is_significant': p_val < 0.05,
                'what_it_means': 'Forwards score way more goals' if p_val < 0.05 else 'No big difference in scoring'
            },
            'passing_test': {
                'correlation': corr_val,
                'p_number': corr_p,
                'is_significant': corr_p < 0.05,
                'what_it_means': f'Pass accuracy {"does" if corr_p < 0.05 else "doesnt"} relate to player rating'
            }
        }
        
        return test_results

# make the analyzer and run everything
my_analyzer = soccer_data_analyzer()
important_nums = my_analyzer.get_important_numbers()
math_stuff = my_analyzer.do_math_analysis()
test_stuff = my_analyzer.test_hypotheses()

# start the dashboard
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "My Soccer Dashboard"

# make the webpage layout
app.layout = dbc.Container([
    # title section
    dbc.Row([
        dbc.Col([
            html.H1("⚽ My Soccer Stats Dashboard", className="text-center mb-4 text-primary"),
            html.P("Looking at fake soccer player data I made", className="text-center text-muted mb-4")
        ])
    ]),
    
    # the main stats boxes
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{important_nums['total_goals']}", className="card-title text-success"),
                    html.P("Total Goals", className="card-text")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{important_nums['total_assists']}", className="card-title text-info"),
                    html.P("Total Assists", className="card-text")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{important_nums['pass_avg']}%", className="card-title text-warning"),
                    html.P("Pass Success Rate", className="card-text")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{important_nums['distance_avg']} km", className="card-title text-danger"),
                    html.P("Average Distance", className="card-text")
                ])
            ])
        ], width=3)
    ], className="mb-4"),
    
    # first row of charts
    dbc.Row([
        # player picker chart
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Individual Player Stats"),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='player-picker',
                        options=[{'label': name, 'value': name} for name in my_analyzer.my_data['player_name'].unique()],
                        value=my_analyzer.my_data['player_name'].unique()[0],
                        className="mb-3"
                    ),
                    dcc.Graph(id='player-chart')
                ])
            ])
        ], width=6),
        
        # position comparison
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🏃 Compare Positions"),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='stat-picker',
                        options=[
                            {'label': 'Goals', 'value': 'goals'},
                            {'label': 'Assists', 'value': 'assists'},
                            {'label': 'Pass %', 'value': 'pass_percent'},
                            {'label': 'Distance', 'value': 'distance_km'}
                        ],
                        value='goals',
                        className="mb-3"
                    ),
                    dcc.Graph(id='position-chart')
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # team trends chart
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📈 Team Performance Over Time"),
                dbc.CardBody([
                    dcc.Graph(id='team-trend-chart')
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # math results section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🔬 Math Analysis Results"),
                dbc.CardBody([
                    html.H5("Regression Stuff", className="text-primary"),
                    html.P(f"R² Score: {math_stuff['r2_value']:.3f}", className="mb-2"),
                    html.P("What matters most:", className="mb-1"),
                    html.Ul([
                        html.Li(f"{stat}: {score:.3f}") 
                        for stat, score in sorted(math_stuff['importance_scores'].items(), key=lambda x: x[1], reverse=True)
                    ]),
                    
                    html.Hr(),
                    
                    html.H5("Hypothesis Tests", className="text-primary"),
                    html.P(test_stuff['forwards_test']['what_it_means']),
                    html.P(f"P-value: {test_stuff['forwards_test']['p_number']:.4f}"),
                    html.P(test_stuff['passing_test']['what_it_means']),
                    html.P(f"Correlation: {test_stuff['passing_test']['correlation']:.3f}")
                ])
            ])
        ], width=12)
    ])
], fluid=True)

# function to update player chart
@app.callback(
    Output('player-chart', 'figure'),
    Input('player-picker', 'value')
)
def make_player_chart(picked_player):
    """make a chart for the selected player"""
    player_info = my_analyzer.my_data[my_analyzer.my_data['player_name'] == picked_player]
    
    fig = go.Figure()
    
    # add rating line
    fig.add_trace(go.Scatter(
        x=player_info['game_num'],
        y=player_info['rating'],
        mode='lines+markers',
        name='Rating',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # add goal bars
    fig.add_trace(go.Bar(
        x=player_info['game_num'],
        y=player_info['goals'],
        name='Goals',
        yaxis='y2',
        opacity=0.7,
        marker_color='green'
    ))
    
    fig.update_layout(
        title=f'{picked_player} Performance',
        xaxis_title='Game Number',
        yaxis=dict(title='Rating', side='left'),
        yaxis2=dict(title='Goals', side='right', overlaying='y'),
        hovermode='x unified',
        height=400
    )
    
    return fig

# function to update position chart
@app.callback(
    Output('position-chart', 'figure'),
    Input('stat-picker', 'value')
)
def make_position_chart(picked_stat):
    """make a chart comparing positions"""
    pos_data = my_analyzer.my_data.groupby('position')[picked_stat].mean().reset_index()
    
    fig = px.bar(
        pos_data,
        x='position',
        y=picked_stat,
        color='position',
        title=f'Average {picked_stat} by Position',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

# function to update team chart
@app.callback(
    Output('team-trend-chart', 'figure'),
    Input('team-trend-chart', 'id')  # dummy input
)
def make_team_chart(_):
    """make the team trends chart"""
    game_stats = my_analyzer.my_data.groupby('game_num').agg({
        'goals': 'sum',
        'assists': 'sum',
        'rating': 'mean',
        'pass_percent': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    # goals line
    fig.add_trace(go.Scatter(
        x=game_stats['game_num'],
        y=game_stats['goals'],
        mode='lines+markers',
        name='Total Goals',
        line=dict(color='green', width=2)
    ))
    
    # assists line
    fig.add_trace(go.Scatter(
        x=game_stats['game_num'],
        y=game_stats['assists'],
        mode='lines+markers',
        name='Total Assists',
        line=dict(color='blue', width=2)
    ))
    
    # average rating line
    fig.add_trace(go.Scatter(
        x=game_stats['game_num'],
        y=game_stats['rating'],
        mode='lines+markers',
        name='Avg Rating',
        yaxis='y2',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title='How Our Team Did Each Game',
        xaxis_title='Game Number',
        yaxis=dict(title='Goals/Assists', side='left'),
        yaxis2=dict(title='Rating', side='right', overlaying='y'),
        hovermode='x unified',
        height=400
    )
    
    return fig

# run everything
if __name__ == '__main__':
    print("🚀 Starting my soccer dashboard...")
    print("📊 Made fake data for 50 players and 20 games")
    print("🔍 Did all the math stuff")
    print("📈 Dashboard is ready at http://127.0.0.1:8050/")
    app.run_server(debug=True)