"""
Backtesting Metrics
Calculate and track prediction accuracy metrics
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class BacktestMetrics:
    """Calculate and store backtesting performance metrics"""
    
    def __init__(self):
        """Initialize metrics tracker"""
        self.predictions = []
        self.actuals = []
        self.dates = []
        self.players = []
    
    def add_prediction(self, actual, predicted, date=None, player=None):
        """
        Add a single prediction result
        
        Args:
            actual (float): Actual value
            predicted (float): Predicted value
            date (str): Game date (optional)
            player (str): Player name (optional)
        """
        self.predictions.append(predicted)
        self.actuals.append(actual)
        self.dates.append(date)
        self.players.append(player)
    
    def add_predictions_batch(self, actuals, predictions, dates=None, players=None):
        """
        Add multiple predictions at once
        
        Args:
            actuals (array-like): Actual values
            predictions (array-like): Predicted values
            dates (array-like): Game dates (optional)
            players (array-like): Player names (optional)
        """
        self.predictions.extend(predictions)
        self.actuals.extend(actuals)
        
        if dates is not None:
            self.dates.extend(dates)
        else:
            self.dates.extend([None] * len(predictions))
        
        if players is not None:
            self.players.extend(players)
        else:
            self.players.extend([None] * len(predictions))
    
    def calculate_metrics(self):
        """
        Calculate all performance metrics
        
        Returns:
            dict: Dictionary of metrics
        """
        if len(self.predictions) == 0:
            return {}
        
        actuals = np.array(self.actuals)
        predictions = np.array(self.predictions)
        
        # Basic metrics
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)
        
        # Error statistics
        errors = predictions - actuals
        mean_error = np.mean(errors)  # Bias
        std_error = np.std(errors)
        
        # Accuracy within thresholds
        abs_errors = np.abs(errors)
        acc_within_1 = np.mean(abs_errors <= 1)
        acc_within_2 = np.mean(abs_errors <= 2)
        acc_within_3 = np.mean(abs_errors <= 3)
        acc_within_5 = np.mean(abs_errors <= 5)
        
        # Over/under prediction rates
        over_predictions = np.mean(errors > 0)
        under_predictions = np.mean(errors < 0)
        
        # Percentiles
        error_25th = np.percentile(abs_errors, 25)
        error_50th = np.percentile(abs_errors, 50)
        error_75th = np.percentile(abs_errors, 75)
        error_90th = np.percentile(abs_errors, 90)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mean_error': mean_error,
            'std_error': std_error,
            'accuracy_within_1': acc_within_1,
            'accuracy_within_2': acc_within_2,
            'accuracy_within_3': acc_within_3,
            'accuracy_within_5': acc_within_5,
            'over_prediction_rate': over_predictions,
            'under_prediction_rate': under_predictions,
            'error_25th_percentile': error_25th,
            'error_median': error_50th,
            'error_75th_percentile': error_75th,
            'error_90th_percentile': error_90th,
            'total_predictions': len(predictions),
            'mean_actual': np.mean(actuals),
            'mean_predicted': np.mean(predictions)
        }
        
        return metrics
    
    def get_results_dataframe(self):
        """
        Get all results as a DataFrame
        
        Returns:
            pd.DataFrame: Results with predictions and actuals
        """
        df = pd.DataFrame({
            'actual': self.actuals,
            'predicted': self.predictions,
            'error': np.array(self.predictions) - np.array(self.actuals),
            'abs_error': np.abs(np.array(self.predictions) - np.array(self.actuals)),
            'date': self.dates,
            'player': self.players
        })
        
        return df
    
    def print_metrics(self, stat_name=""):
        """
        Print metrics in a formatted way
        
        Args:
            stat_name (str): Name of the statistic being evaluated
        """
        metrics = self.calculate_metrics()
        
        if not metrics:
            print("No predictions to evaluate")
            return
        
        print(f"\n{'='*60}")
        print(f"Backtest Metrics - {stat_name}")
        print(f"{'='*60}")
        
        print(f"\n📊 Overall Performance:")
        print(f"  Total Predictions:    {metrics['total_predictions']:,}")
        print(f"  MAE:                  {metrics['mae']:.3f}")
        print(f"  RMSE:                 {metrics['rmse']:.3f}")
        print(f"  R² Score:             {metrics['r2']:.3f}")
        
        print(f"\n🎯 Accuracy Within Thresholds:")
        print(f"  Within 1:             {metrics['accuracy_within_1']*100:.1f}%")
        print(f"  Within 2:             {metrics['accuracy_within_2']*100:.1f}%")
        print(f"  Within 3:             {metrics['accuracy_within_3']*100:.1f}%")
        print(f"  Within 5:             {metrics['accuracy_within_5']*100:.1f}%")
        
        print(f"\n📈 Prediction Bias:")
        print(f"  Mean Error:           {metrics['mean_error']:.3f}")
        print(f"  Std Error:            {metrics['std_error']:.3f}")
        print(f"  Over-predictions:     {metrics['over_prediction_rate']*100:.1f}%")
        print(f"  Under-predictions:    {metrics['under_prediction_rate']*100:.1f}%")
        
        print(f"\n📉 Error Distribution:")
        print(f"  25th Percentile:      {metrics['error_25th_percentile']:.3f}")
        print(f"  Median (50th):        {metrics['error_median']:.3f}")
        print(f"  75th Percentile:      {metrics['error_75th_percentile']:.3f}")
        print(f"  90th Percentile:      {metrics['error_90th_percentile']:.3f}")
        
        print(f"\n🔢 Averages:")
        print(f"  Mean Actual:          {metrics['mean_actual']:.2f}")
        print(f"  Mean Predicted:       {metrics['mean_predicted']:.2f}")
        print(f"{'='*60}\n")
    
    def get_worst_predictions(self, n=10):
        """
        Get worst predictions by absolute error
        
        Args:
            n (int): Number of worst predictions to return
            
        Returns:
            pd.DataFrame: Worst predictions
        """
        df = self.get_results_dataframe()
        return df.nlargest(n, 'abs_error')
    
    def get_best_predictions(self, n=10):
        """
        Get best predictions by absolute error
        
        Args:
            n (int): Number of best predictions to return
            
        Returns:
            pd.DataFrame: Best predictions
        """
        df = self.get_results_dataframe()
        return df.nsmallest(n, 'abs_error')
    
    def get_metrics_by_player(self):
        """
        Calculate metrics for each player
        
        Returns:
            pd.DataFrame: Metrics per player
        """
        df = self.get_results_dataframe()
        
        if df['player'].isna().all():
            return None
        
        player_metrics = df.groupby('player').apply(
            lambda x: pd.Series({
                'predictions': len(x),
                'mae': mean_absolute_error(x['actual'], x['predicted']),
                'mean_error': x['error'].mean(),
                'mean_actual': x['actual'].mean(),
                'mean_predicted': x['predicted'].mean()
            })
        ).reset_index()
        
        return player_metrics.sort_values('mae')
    
    def clear(self):
        """Clear all stored predictions"""
        self.predictions = []
        self.actuals = []
        self.dates = []
        self.players = []