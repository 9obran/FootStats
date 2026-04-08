# FootStats: Soccer Analytics Dashboard

## The Problem

I play Football Manager and always wondered: **what stats actually matter?** The game shows dozens of metrics per player, but which ones actually predict match performance?

I couldn't find a simple tool that lets me:
- Generate realistic player data
- Test correlations between stats
- Build a predictive model for player ratings
- Visualize everything interactively

So I built one.

## The Approach

**Data Generation**: Instead of using real data (hard to get), I wrote a generator that creates realistic synthetic data. Each position has different distributions:
- Forwards: More goals, fewer tackles
- Midfielders: High pass counts, moderate goals
- Defenders: High tackles, low shots
- Goalkeepers: Specialized stats

**Analysis Pipeline**:
1. **Descriptive stats**: Means, distributions by position
2. **Correlation analysis**: Which stats relate to each other?
3. **Regression**: Can we predict player rating from other stats?
4. **Hypothesis testing**: Do forwards actually score more than midfielders? (spoiler: yes)

**Dashboard**: Built with Plotly Dash + Bootstrap. Interactive filtering by position, player, stat type.

## The Result

A fully interactive dashboard showing:
- Player performance trends over 20 games
- Position comparisons (box plots, bar charts)
- Statistical analysis (correlations, regression, hypothesis tests)
- Player categorization (star/good/average/needs improvement)

**Key Findings from the Data**:
- Pass accuracy correlates with rating (r ≈ 0.4)
- Forwards score significantly more than other positions (p < 0.001, obviously)
- A simple linear regression can predict rating with R² ≈ 0.6 from just 5 stats

## Run It

```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Launch dashboard
python my_football_dashboard.py

# Or run statistical analysis only
python stats_stuff.py
```

Then open http://127.0.0.1:8050 in your browser.

## Project Structure

```text
FootStats/
├── my_football_dashboard.py    # Interactive Dash dashboard (main entry point)
├── stats_stuff.py             # Statistical analysis module
├── requirements.txt           # Dependencies
└── README.md                 # This file
```

## What I Learned

**Statistics**:
- T-tests for comparing groups (forwards vs midfielders)
- ANOVA for comparing multiple groups
- Pearson correlation for linear relationships
- Linear regression for prediction
- Effect size matters, not just p-values

**Dash/Plotly**:
- Callbacks for interactivity
- Layout management with Bootstrap
- Multi-axis charts for different scales

**Data Generation**:
- Poisson distribution for count data (goals, assists)
- Normal distribution for continuous stats (pass %, km run)
- Position-specific parameters make data realistic

## What I'd Do Next

1. **Real Data**: Integrate with an API like Football-data.org or scrape real stats
2. **More Models**: Try Random Forest or XGBoost for better prediction
3. **Time Series**: Add trend analysis (players improving/declining over season)
4. **Team Analysis**: Not just individual players, but team-level patterns
5. **Deployment**: Deploy to Render or similar for public access

## Screenshots

![FootStats Dashboard](dashboard.jpg)

## Requirements

- Python 3.8+
- Dash, Plotly, scikit-learn, scipy

See `requirements.txt` for pinned versions.

## License

Student project - built for learning data science and visualization.