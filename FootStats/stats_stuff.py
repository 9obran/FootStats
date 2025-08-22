# doing math on football data - this is hard but i think i got it

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class soccer_math_analyzer:
    """
    this class does all the hard math stuff for soccer data
    i looked up most of this online but i think i understand it now
    """
    
    def __init__(self, data):
        """
        start the math analyzer with soccer data
        data should be a pandas dataframe
        """
        self.my_data = data
        self.results = {}
        
    def basic_stats(self):
        """
        calculate basic statistics for all the number columns
        like mean, median, etc.
        """
        number_cols = self.my_data.select_dtypes(include=[np.number]).columns
        
        basic_results = {}
        for col in number_cols:
            basic_results[col] = {
                'average': self.my_data[col].mean(),
                'middle_value': self.my_data[col].median(),
                'spread': self.my_data[col].std(),
                'lowest': self.my_data[col].min(),
                'highest': self.my_data[col].max(),
                'skew_number': stats.skew(self.my_data[col]),  # how lopsided
                'kurtosis_number': stats.kurtosis(self.my_data[col])  # how pointy
            }
        
        self.results['basic_numbers'] = basic_results
        return basic_results
    
    def correlation_stuff(self):
        """
        see how different stats relate to each other
        correlation means when one goes up, does the other go up too?
        """
        # pick the important columns to compare
        important_cols = [
            'goals', 'assists', 'shots', 'pass_percent', 
            'distance_km', 'rating', 'goals_and_assists'
        ]
        
        corr_data = self.my_data[important_cols]
        correlation_table = corr_data.corr()
        
        # find the strong relationships (over 0.3)
        strong_relationships = {}
        for i in range(len(correlation_table.columns)):
            for j in range(i+1, len(correlation_table.columns)):
                col1 = correlation_table.columns[i]
                col2 = correlation_table.columns[j]
                corr_value = correlation_table.iloc[i, j]
                
                if abs(corr_value) > 0.3:  # only care about strong ones
                    strong_relationships[f"{col1}_and_{col2}"] = {
                        'correlation_number': corr_value,
                        'strength': self.describe_correlation(abs(corr_value))
                    }
        
        self.results['correlations'] = {
            'full_table': correlation_table,
            'strong_ones': strong_relationships
        }
        
        return self.results['correlations']
    
    def describe_correlation(self, corr_val):
        """helper function to say how strong a correlation is"""
        if corr_val >= 0.7:
            return "Really Strong"
        elif corr_val >= 0.5:
            return "Pretty Strong"
        elif corr_val >= 0.3:
            return "Kind of Weak"
        else:
            return "Super Weak"
    
    def do_regression(self):
        """
        try to predict player rating using other stats
        this is like guessing how good a player is based on their numbers
        """
        # what we want to use to predict
        predictors = ['goals', 'assists', 'pass_percent', 'distance_km', 'shots']
        what_to_predict = 'rating'
        
        # get the data ready
        X_stuff = self.my_data[predictors]
        y_stuff = self.my_data[what_to_predict]
        
        # split into training and testing (learned this in class)
        X_train, X_test, y_train, y_test = train_test_split(
            X_stuff, y_stuff, test_size=0.2, random_state=42
        )
        
        # make the prediction model
        my_model = LinearRegression()
        my_model.fit(X_train, y_train)
        
        # see how good our predictions are
        train_predictions = my_model.predict(X_train)
        test_predictions = my_model.predict(X_test)
        
        # calculate how good we did
        train_score = r2_score(y_train, train_predictions)
        test_score = r2_score(y_test, test_predictions)
        train_error = np.sqrt(mean_squared_error(y_train, train_predictions))
        test_error = np.sqrt(mean_squared_error(y_test, test_predictions))
        
        # figure out which stats matter most
        importance = {}
        coefficients = {}
        for i, predictor in enumerate(predictors):
            coefficients[predictor] = my_model.coef_[i]
            importance[predictor] = abs(my_model.coef_[i])
        
        # sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        # do some basic significance testing (this part is really hard)
        errors = y_train - train_predictions
        error_variance = np.mean(errors**2)
        coef_variance = error_variance * np.diagonal(np.linalg.inv(X_train.T @ X_train))
        t_statistics = my_model.coef_ / np.sqrt(coef_variance)
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_statistics), len(X_train) - len(predictors) - 1))
        
        regression_info = {
            'coefficients': coefficients,
            'y_intercept': my_model.intercept_,
            'importance_ranking': importance,
            'training_score': train_score,
            'testing_score': test_score,
            'training_error': train_error,
            'testing_error': test_error,
            'p_values': dict(zip(predictors, p_values)),
            'significant_predictors': [pred for pred, p in zip(predictors, p_values) if p < 0.05]
        }
        
        self.results['regression_results'] = regression_info
        return regression_info
    
    def position_comparison(self):
        """
        compare how different positions perform using ANOVA
        ANOVA tells us if groups are really different or just random
        """
        positions = self.my_data['position'].unique()
        
        # test different stats across positions
        stats_to_test = ['goals', 'assists', 'pass_percent', 'rating']
        anova_info = {}
        
        for stat in stats_to_test:
            # get data for each position
            position_data = []
            for pos in positions:
                pos_stats = self.my_data[self.my_data['position'] == pos][stat]
                position_data.append(pos_stats)
            
            # do the ANOVA test
            f_number, p_number = stats.f_oneway(*position_data)
            
            anova_info[stat] = {
                'f_statistic': f_number,
                'p_value': p_number,
                'is_different': p_number < 0.05
            }
        
        # get summary stats by position
        position_summary = self.my_data.groupby('position')[stats_to_test].agg(['mean', 'std']).round(2)
        
        self.results['position_comparison'] = {
            'anova_tests': anova_info,
            'position_averages': position_summary
        }
        
        return self.results['position_comparison']
    
    def test_my_ideas(self):
        """
        test some ideas i have about soccer using statistics
        """
        my_tests = {}
        
        # idea 1: forwards score way more goals than midfielders
        forward_goals = self.my_data[self.my_data['position'] == 'Forward']['goals']
        midfielder_goals = self.my_data[self.my_data['position'] == 'Midfielder']['goals']
        
        t_stat1, p_val1 = stats.ttest_ind(forward_goals, midfielder_goals)
        my_tests['forwards_vs_midfielders'] = {
            'test_name': 'T-test for two groups',
            'null_idea': 'Forwards and midfielders score the same amount',
            't_number': t_stat1,
            'p_number': p_val1,
            'is_significant': p_val1 < 0.05,
            'effect_size': (forward_goals.mean() - midfielder_goals.mean()) / 
                          np.sqrt((forward_goals.var() + midfielder_goals.var()) / 2),
            'what_it_means': self.explain_ttest(p_val1, forward_goals.mean(), 
                                              midfielder_goals.mean(), 'goals')
        }
        
        # idea 2: good passers get better ratings
        pass_rating_corr, p_val2 = stats.pearsonr(self.my_data['pass_percent'], 
                                                   self.my_data['rating'])
        my_tests['passing_and_rating'] = {
            'test_name': 'Correlation test',
            'null_idea': 'Pass accuracy and rating are not related',
            'correlation_number': pass_rating_corr,
            'p_number': p_val2,
            'is_significant': p_val2 < 0.05,
            'what_it_means': f"Pass accuracy {'is' if p_val2 < 0.05 else 'is not'} related to player rating (r={pass_rating_corr:.3f})"
        }
        
        # idea 3: are ratings normally distributed? (bell curve shaped)
        sample_size = min(5000, len(self.my_data))
        rating_sample = self.my_data['rating'].sample(sample_size)
        shapiro_stat, p_val3 = stats.shapiro(rating_sample)
        my_tests['rating_distribution'] = {
            'test_name': 'Shapiro-Wilk normality test',
            'null_idea': 'Ratings follow a normal bell curve',
            'test_number': shapiro_stat,
            'p_number': p_val3,
            'is_significant': p_val3 < 0.05,
            'what_it_means': f"Ratings {'do not' if p_val3 < 0.05 else 'do'} follow a bell curve shape"
        }
        
        # idea 4: players with more assists are rated higher
        median_assists = self.my_data['assists'].median()
        high_assist_ratings = self.my_data[self.my_data['assists'] > median_assists]['rating']
        low_assist_ratings = self.my_data[self.my_data['assists'] <= median_assists]['rating']
        
        t_stat4, p_val4 = stats.ttest_ind(high_assist_ratings, low_assist_ratings)
        my_tests['assists_and_rating'] = {
            'test_name': 'T-test comparing high vs low assist players',
            'null_idea': 'High and low assist players have same ratings',
            't_number': t_stat4,
            'p_number': p_val4,
            'is_significant': p_val4 < 0.05,
            'what_it_means': self.explain_ttest(p_val4, high_assist_ratings.mean(), 
                                              low_assist_ratings.mean(), 'rating')
        }
        
        self.results['my_hypothesis_tests'] = my_tests
        return my_tests
    
    def explain_ttest(self, p_value, mean1, mean2, stat_name):
        """helper function to explain what t-test results mean"""
        if p_value < 0.05:
            if mean1 > mean2:
                direction = "higher"
            else:
                direction = "lower"
            return f"Big difference found: Group 1 has {direction} {stat_name} (p={p_value:.4f})"
        else:
            return f"No big difference in {stat_name} between groups (p={p_value:.4f})"
    
    def trend_analysis(self):
        """
        look at how performance changes over time (across games)
        """
        # calculate team performance by game
        game_trends = self.my_data.groupby('game_num').agg({
            'goals': ['sum', 'mean'],
            'assists': ['sum', 'mean'],
            'rating': 'mean',
            'pass_percent': 'mean',
            'distance_km': 'mean'
        }).round(2)
        
        # flatten column names (this is confusing but necessary)
        game_trends.columns = ['_'.join(col).strip() for col in game_trends.columns.values]
        game_trends = game_trends.reset_index()
        
        # see if there are trends over time
        games = game_trends['game_num']
        
        trends = {}
        for col in game_trends.columns:
            if col != 'game_num':
                # calculate correlation between game number and stat
                corr, p_val = stats.pearsonr(games, game_trends[col])
                trends[col] = {
                    'correlation_with_time': corr,
                    'p_value': p_val,
                    'is_trending': p_val < 0.05,
                    'trend_direction': 'improving' if corr > 0 else 'declining' if corr < 0 else 'stable'
                }
        
        self.results['time_trends'] = {
            'game_by_game_data': game_trends,
            'trend_analysis': trends
        }
        
        return self.results['time_trends']
    
    def player_performance_categories(self):
        """
        group players into categories based on their overall performance
        like 'star players', 'average players', etc.
        """
        # calculate overall stats per player
        player_totals = self.my_data.groupby('player_name').agg({
            'goals': 'sum',
            'assists': 'sum',
            'rating': 'mean',
            'pass_percent': 'mean',
            'distance_km': 'mean',
            'game_num': 'count'  # games played
        }).round(2)
        
        player_totals.columns = ['total_goals', 'total_assists', 'avg_rating', 
                                'avg_pass_percent', 'avg_distance', 'games_played']
        
        # create a simple overall score
        # normalize each stat to 0-1 scale then average them
        for col in ['total_goals', 'total_assists', 'avg_rating']:
            min_val = player_totals[col].min()
            max_val = player_totals[col].max()
            player_totals[f'{col}_normalized'] = (player_totals[col] - min_val) / (max_val - min_val)
        
        # calculate overall score
        player_totals['overall_score'] = (
            player_totals['total_goals_normalized'] * 0.3 +
            player_totals['total_assists_normalized'] * 0.3 +
            player_totals['avg_rating_normalized'] * 0.4
        )
        
        # categorize players
        def categorize_player(score):
            if score >= 0.8:
                return 'Star Player'
            elif score >= 0.6:
                return 'Good Player'
            elif score >= 0.4:
                return 'Average Player'
            else:
                return 'Needs Improvement'
        
        player_totals['category'] = player_totals['overall_score'].apply(categorize_player)
        
        # count how many in each category
        category_counts = player_totals['category'].value_counts()
        
        self.results['player_categories'] = {
            'player_stats': player_totals,
            'category_distribution': category_counts.to_dict()
        }
        
        return self.results['player_categories']
    
    def generate_report(self):
        """
        make a simple text report of all the findings
        """
        if not self.results:
            return "No analysis has been run yet. Please run the analysis methods first."
        
        report = []
        report.append("=== MY SOCCER DATA ANALYSIS REPORT ===\n")
        
        # basic stats summary
        if 'basic_numbers' in self.results:
            report.append("BASIC STATISTICS:")
            report.append("- Goals: avg={:.2f}, max={}".format(
                self.results['basic_numbers']['goals']['average'],
                self.results['basic_numbers']['goals']['highest']
            ))
            report.append("- Assists: avg={:.2f}, max={}".format(
                self.results['basic_numbers']['assists']['average'],
                self.results['basic_numbers']['assists']['highest']
            ))
            report.append("- Rating: avg={:.2f}".format(
                self.results['basic_numbers']['rating']['average']
            ))
            report.append("")
        
        # correlation findings
        if 'correlations' in self.results:
            report.append("STRONG RELATIONSHIPS FOUND:")
            for relationship, info in self.results['correlations']['strong_ones'].items():
                report.append("- {}: {:.3f} ({})".format(
                    relationship.replace('_and_', ' vs '),
                    info['correlation_number'],
                    info['strength']
                ))
            report.append("")
        
        # regression results
        if 'regression_results' in self.results:
            report.append("PREDICTION MODEL RESULTS:")
            report.append("- Model accuracy (R²): {:.3f}".format(
                self.results['regression_results']['testing_score']
            ))
            report.append("- Most important factors:")
            for factor, importance in list(self.results['regression_results']['importance_ranking'].items())[:3]:
                report.append("  * {}: {:.3f}".format(factor, importance))
            report.append("")
        
        # hypothesis test results
        if 'my_hypothesis_tests' in self.results:
            report.append("HYPOTHESIS TEST RESULTS:")
            for test_name, test_info in self.results['my_hypothesis_tests'].items():
                report.append("- {}: {}".format(
                    test_name.replace('_', ' ').title(),
                    test_info['what_it_means']
                ))
            report.append("")
        
        # player categories
        if 'player_categories' in self.results:
            report.append("PLAYER PERFORMANCE CATEGORIES:")
            for category, count in self.results['player_categories']['category_distribution'].items():
                report.append("- {}: {} players".format(category, count))
            report.append("")
        
        report.append("=== END OF REPORT ===")
        
        return "\n".join(report)

# example of how to use this class
def run_full_analysis(data):
    """
    run all the analysis functions on soccer data
    """
    analyzer = soccer_math_analyzer(data)
    
    print("Running basic statistics...")
    basic = analyzer.basic_stats()
    
    print("Analyzing correlations...")
    correlations = analyzer.correlation_stuff()
    
    print("Doing regression analysis...")
    regression = analyzer.do_regression()
    
    print("Comparing positions...")
    positions = analyzer.position_comparison()
    
    print("Testing hypotheses...")
    hypotheses = analyzer.test_my_ideas()
    
    print("Analyzing trends...")
    trends = analyzer.trend_analysis()
    
    print("Categorizing players...")
    categories = analyzer.player_performance_categories()
    
    print("Generating report...")
    report = analyzer.generate_report()
    
    return analyzer, report

# if running this file directly, do a demo
if __name__ == "__main__":
    print("This is the statistics analysis module for soccer data")
    print("Import this file and use the soccer_math_analyzer class")
    print("Example: analyzer = soccer_math_analyzer(your_data)")
    print("Then call methods like analyzer.basic_stats()")