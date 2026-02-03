"""
Backtest Engine
Run backtests on historical data to validate model performance
"""

import pandas as pd
import numpy as np
import yaml
import os
import pickle
from src.backtesting.metrics import BacktestMetrics


class BacktestEngine:
    """Engine for running backtests on NBA prediction models"""
    
    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize backtest engine
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models_dir = self.config['paths']['models_dir']
        self.processed_dir = self.config['paths']['processed_dir']
        self.results_dir = self.config['paths']['results_dir']
        self.target_stats = self.config['target_stats']
        
        self.models = {}
        self.metrics = {}
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_models(self):
        """Load all trained models"""
        print("\n" + "="*60)
        print("LOADING TRAINED MODELS")
        print("="*60)
        
        for stat in self.target_stats:
            model_path = f"{self.models_dir}/{stat.lower()}_model.pkl"
            
            if not os.path.exists(model_path):
                print(f"⚠️  Model not found: {model_path}")
                continue
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models[stat] = model_data
            print(f"✓ Loaded {stat} model")
        
        print(f"\nTotal models loaded: {len(self.models)}")
    
    def load_test_data(self):
        """
        Load test data for backtesting
        
        Returns:
            pd.DataFrame: Test dataset
        """
        print("\n" + "="*60)
        print("LOADING TEST DATA")
        print("="*60)
        
        test_path = f"{self.processed_dir}/test_features.csv"
        
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        test_df = pd.read_csv(test_path)
        
        # Convert date if exists
        if 'GAME_DATE' in test_df.columns:
            test_df['GAME_DATE'] = pd.to_datetime(test_df['GAME_DATE'])
        
        print(f"✓ Test data loaded: {test_df.shape}")
        print(f"  Games: {len(test_df):,}")
        print(f"  Players: {test_df['PLAYER_NAME'].nunique()}")
        if 'GAME_DATE' in test_df.columns:
            print(f"  Date range: {test_df['GAME_DATE'].min()} to {test_df['GAME_DATE'].max()}")
        
        return test_df
    
    def prepare_features(self, df, feature_names):
        """
        Prepare features for prediction
        
        Args:
            df (pd.DataFrame): Data with all columns
            feature_names (list): List of feature column names
            
        Returns:
            pd.DataFrame: Features only
        """
        # Get available features (some might be missing)
        available_features = [f for f in feature_names if f in df.columns]
        
        if len(available_features) < len(feature_names):
            missing = set(feature_names) - set(available_features)
            print(f"⚠️  Missing {len(missing)} features from model")
        
        return df[available_features]
    
    def run_backtest_for_stat(self, stat, test_df):
        """
        Run backtest for a specific statistic
        
        Args:
            stat (str): Target statistic (PTS, REB, AST)
            test_df (pd.DataFrame): Test data
            
        Returns:
            BacktestMetrics: Metrics object with results
        """
        print(f"\n{'='*60}")
        print(f"BACKTESTING: {stat}")
        print(f"{'='*60}")
        
        # Load model
        if stat not in self.models:
            print(f"⚠️  Model not loaded for {stat}")
            return None
        
        model_data = self.models[stat]
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        # Prepare data
        X_test = self.prepare_features(test_df, feature_names)
        y_test = test_df[stat]
        
        print(f"Making predictions on {len(X_test):,} games...")
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Create metrics tracker
        metrics = BacktestMetrics()
        
        # Add predictions with metadata
        dates = test_df['GAME_DATE'].values if 'GAME_DATE' in test_df.columns else None
        players = test_df['PLAYER_NAME'].values if 'PLAYER_NAME' in test_df.columns else None
        
        metrics.add_predictions_batch(
            actuals=y_test,
            predictions=predictions,
            dates=dates,
            players=players
        )
        
        # Print metrics
        metrics.print_metrics(stat_name=stat)
        
        return metrics
    
    def run_all_backtests(self):
        """
        Run backtests for all statistics
        
        Returns:
            dict: Dictionary of BacktestMetrics for each stat
        """
        print("\n" + "="*70)
        print(" "*20 + "BACKTESTING PIPELINE")
        print("="*70)
        
        # Load models
        self.load_models()
        
        if len(self.models) == 0:
            print("\n❌ No models found! Please train models first.")
            return {}
        
        # Load test data
        test_df = self.load_test_data()
        
        # Run backtest for each stat
        all_metrics = {}
        
        for stat in self.target_stats:
            if stat not in self.models:
                continue
            
            metrics = self.run_backtest_for_stat(stat, test_df)
            
            if metrics:
                all_metrics[stat] = metrics
                
                # Save results
                results_df = metrics.get_results_dataframe()
                results_path = f"{self.results_dir}/backtest_{stat.lower()}_results.csv"
                results_df.to_csv(results_path, index=False)
                print(f"✓ Results saved: {results_path}")
        
        # Print summary
        self.print_backtest_summary(all_metrics)
        
        # Save summary metrics
        self.save_metrics_summary(all_metrics)
        
        return all_metrics
    
    def print_backtest_summary(self, all_metrics):
        """
        Print summary comparison across all stats
        
        Args:
            all_metrics (dict): Dictionary of BacktestMetrics
        """
        print("\n" + "="*70)
        print(" "*22 + "BACKTEST SUMMARY")
        print("="*70)
        
        print("\n📊 Performance Comparison:\n")
        print(f"{'Stat':<10} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'Within 3':<12} {'Predictions':<12}")
        print("-" * 70)
        
        for stat, metrics in all_metrics.items():
            metrics_dict = metrics.calculate_metrics()
            
            mae = metrics_dict['mae']
            rmse = metrics_dict['rmse']
            r2 = metrics_dict['r2']
            acc3 = metrics_dict['accuracy_within_3'] * 100
            total = metrics_dict['total_predictions']
            
            print(f"{stat:<10} {mae:<10.3f} {rmse:<10.3f} {r2:<10.3f} {acc3:<11.1f}% {total:<12,}")
        
        print("\n✅ Backtesting complete!")
        print(f"📁 Results saved to: {self.results_dir}/")
    
    def save_metrics_summary(self, all_metrics):
        """
        Save metrics summary to JSON
        
        Args:
            all_metrics (dict): Dictionary of BacktestMetrics
        """
        import json
        
        summary = {}
        
        for stat, metrics in all_metrics.items():
            summary[stat] = metrics.calculate_metrics()
            
            # Convert numpy types to Python types for JSON
            for key, value in summary[stat].items():
                if isinstance(value, (np.integer, np.floating)):
                    summary[stat][key] = float(value)
        
        summary_path = f"{self.results_dir}/backtest_metrics_summary.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Metrics summary saved: {summary_path}")
    
    def analyze_player_performance(self, stat, top_n=10):
        """
        Analyze which players have best/worst prediction accuracy
        
        Args:
            stat (str): Statistic to analyze
            top_n (int): Number of players to show
            
        Returns:
            tuple: (best_players, worst_players)
        """
        if stat not in self.metrics:
            print(f"No metrics found for {stat}")
            return None, None
        
        metrics = self.metrics[stat]
        player_metrics = metrics.get_metrics_by_player()
        
        if player_metrics is None:
            print("Player information not available")
            return None, None
        
        # Filter players with enough predictions
        player_metrics = player_metrics[player_metrics['predictions'] >= 10]
        
        best_players = player_metrics.nsmallest(top_n, 'mae')
        worst_players = player_metrics.nlargest(top_n, 'mae')
        
        print(f"\n{'='*60}")
        print(f"Top {top_n} Most Predictable Players - {stat}")
        print(f"{'='*60}")
        print(best_players[['player', 'mae', 'predictions']].to_string(index=False))
        
        print(f"\n{'='*60}")
        print(f"Top {top_n} Least Predictable Players - {stat}")
        print(f"{'='*60}")
        print(worst_players[['player', 'mae', 'predictions']].to_string(index=False))
        
        return best_players, worst_players


if __name__ == "__main__":
    engine = BacktestEngine()
    metrics = engine.run_all_backtests()