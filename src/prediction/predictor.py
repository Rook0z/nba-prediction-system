import pandas as pd
import numpy as np
import pickle
import os
import yaml
from datetime import datetime, timedelta


class NBAPredictor:
    """Generate predictions for NBA player performance"""
    
    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize predictor
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models_dir = self.config['paths']['models_dir']
        self.processed_dir = self.config['paths']['processed_dir']
        self.target_stats = self.config['target_stats']
        
        self.models = {}
        self.feature_names = {}
        
    def load_models(self):
        """Load all trained models"""
        print("\n" + "="*60)
        print("LOADING PREDICTION MODELS")
        print("="*60)
        
        for stat in self.target_stats:
            model_path = f"{self.models_dir}/{stat.lower()}_model.pkl"
            
            if not os.path.exists(model_path):
                print(f"⚠️  Model not found: {model_path}")
                continue
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models[stat] = model_data['model']
            self.feature_names[stat] = model_data['feature_names']
            
            training_date = model_data.get('training_date')
            if training_date:
                print(f"✓ Loaded {stat} model (trained: {training_date.strftime('%Y-%m-%d')})")
            else:
                print(f"✓ Loaded {stat} model")
        
        print(f"\nTotal models loaded: {len(self.models)}")
        
        if len(self.models) == 0:
            raise ValueError("No models found! Please train models first.")
    
    def load_latest_data(self):
        """
        Load the most recent data for making predictions
        
        Returns:
            pd.DataFrame: Latest game data with features
        """
        print("\n" + "="*60)
        print("LOADING LATEST DATA")
        print("="*60)
        
        # Try to load test data (most recent)
        test_path = f"{self.processed_dir}/test_features.csv"
        
        if os.path.exists(test_path):
            df = pd.read_csv(test_path)
            print(f"✓ Loaded test data: {df.shape}")
        else:
            # Fallback to training data
            train_path = f"{self.processed_dir}/train_features.csv"
            if os.path.exists(train_path):
                df = pd.read_csv(train_path)
                print(f"✓ Loaded training data: {df.shape}")
            else:
                raise FileNotFoundError("No feature data found!")
        
        # Convert date
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        
        return df
    
    def get_latest_player_data(self, df, lookback_days=30):
        """
        Get the most recent data for each player
        
        Args:
            df (pd.DataFrame): Full dataset
            lookback_days (int): How many days to look back
            
        Returns:
            pd.DataFrame: Latest data per player
        """
        if 'GAME_DATE' not in df.columns:
            return df.groupby('PLAYER_ID').last().reset_index()
        
        # Get recent games
        max_date = df['GAME_DATE'].max()
        cutoff_date = max_date - timedelta(days=lookback_days)
        
        recent_df = df[df['GAME_DATE'] >= cutoff_date].copy()
        
        # Get most recent game for each player
        latest = recent_df.groupby('PLAYER_ID').last().reset_index()
        
        return latest
    
    def prepare_features(self, df, stat):
        """
        Prepare features for prediction
        
        Args:
            df (pd.DataFrame): Data with features
            stat (str): Target statistic
            
        Returns:
            pd.DataFrame: Features only
        """
        feature_names = self.feature_names[stat]
        
        # Get available features
        available_features = [f for f in feature_names if f in df.columns]
        
        if len(available_features) < len(feature_names):
            missing = set(feature_names) - set(available_features)
            print(f"⚠️  Warning: {len(missing)} features missing for {stat}")
        
        return df[available_features]
    
    def predict_for_players(self, df, players=None, min_minutes=None):
        """
        Generate predictions for specified players
        
        Args:
            df (pd.DataFrame): Player data with features
            players (list): List of player names (None = all)
            min_minutes (float): Minimum minutes filter
            
        Returns:
            pd.DataFrame: Predictions for each player and stat
        """
        print("\n" + "="*60)
        print("GENERATING PREDICTIONS")
        print("="*60)
        
        # Filter by players if specified
        if players:
            df = df[df['PLAYER_NAME'].isin(players)].copy()
            print(f"Filtering to {len(players)} specified players")
        
        # Filter by minutes if specified
        if min_minutes and 'MIN' in df.columns:
            df = df[df['MIN'] >= min_minutes].copy()
            print(f"Filtering to players with {min_minutes}+ minutes")
        
        if len(df) == 0:
            print("⚠️  No players match criteria!")
            return None
        
        print(f"\nGenerating predictions for {len(df)} players...")
        
        # Prepare results dataframe
        results = df[['PLAYER_NAME', 'TEAM_ABBREVIATION']].copy()
        
        # Add current averages if available
        for stat in self.target_stats:
            if stat in df.columns:
                results[f'{stat}_CURRENT'] = df[stat]
            
            if f'{stat}_L10' in df.columns:
                results[f'{stat}_L10_AVG'] = df[f'{stat}_L10']
        
        # Generate predictions for each stat
        for stat in self.target_stats:
            if stat not in self.models:
                print(f"⚠️  No model for {stat}")
                continue
            
            X = self.prepare_features(df, stat)
            predictions = self.models[stat].predict(X)
            
            results[f'{stat}_PREDICTED'] = predictions
            
            print(f"✓ {stat} predictions generated")
        
        return results
    
    def predict_today(self, min_minutes=20):
        """
        Generate predictions using most recent player data
        
        Args:
            min_minutes (float): Minimum minutes to include player
            
        Returns:
            pd.DataFrame: Predictions for all active players
        """
        print("\n" + "="*70)
        print(" "*20 + "TODAY'S PREDICTIONS")
        print("="*70)
        
        # Load models
        self.load_models()
        
        # Load data
        df = self.load_latest_data()
        
        # Get latest player data
        latest_df = self.get_latest_player_data(df, lookback_days=30)
        
        print(f"\nLatest data for {len(latest_df)} players")
        
        # Generate predictions
        predictions = self.predict_for_players(
            latest_df,
            min_minutes=min_minutes
        )
        
        if predictions is None:
            return None
        
        # Sort by predicted points (most interesting players first)
        if 'PTS_PREDICTED' in predictions.columns:
            predictions = predictions.sort_values('PTS_PREDICTED', ascending=False)
        
        return predictions
    
    def format_predictions(self, predictions, top_n=50):
        """
        Format predictions for display
        
        Args:
            predictions (pd.DataFrame): Prediction results
            top_n (int): Number of top players to show
            
        Returns:
            str: Formatted output
        """
        if predictions is None or len(predictions) == 0:
            return "No predictions available"
        
        output = []
        output.append("\n" + "="*100)
        output.append(" "*35 + "PLAYER PREDICTIONS")
        output.append("="*100)
        
        # Show top N players
        top_predictions = predictions.head(top_n)
        
        output.append(f"\n{'Player':<25} {'Team':<6} {'PTS':<8} {'REB':<8} {'AST':<8}")
        output.append("-" * 100)
        
        for idx, row in top_predictions.iterrows():
            player = row['PLAYER_NAME'][:24]
            team = row.get('TEAM_ABBREVIATION', 'N/A')[:5]
            
            pts = row.get('PTS_PREDICTED', 0)
            reb = row.get('REB_PREDICTED', 0)
            ast = row.get('AST_PREDICTED', 0)
            
            output.append(f"{player:<25} {team:<6} {pts:>6.1f}   {reb:>6.1f}   {ast:>6.1f}")
        
        output.append("\n" + "="*100)
        output.append(f"Total players: {len(predictions)}")
        output.append(f"Showing top {len(top_predictions)}")
        
        return "\n".join(output)
    
    def save_predictions(self, predictions, filename=None):
        """
        Save predictions to CSV
        
        Args:
            predictions (pd.DataFrame): Predictions
            filename (str): Output filename (auto-generated if None)
            
        Returns:
            str: Path to saved file
        """
        if filename is None:
            today = datetime.now().strftime('%Y%m%d')
            filename = f"predictions_{today}.csv"
        
        results_dir = self.config['paths']['results_dir']
        filepath = os.path.join(results_dir, filename)
        
        predictions.to_csv(filepath, index=False)
        
        print(f"\n✓ Predictions saved: {filepath}")
        
        return filepath
    
    def get_player_prediction(self, player_name):
        """
        Get prediction for a specific player
        
        Args:
            player_name (str): Player name
            
        Returns:
            dict: Predictions for the player
        """
        self.load_models()
        df = self.load_latest_data()
        latest_df = self.get_latest_player_data(df)
        
        predictions = self.predict_for_players(latest_df, players=[player_name])
        
        if predictions is None or len(predictions) == 0:
            return None
        
        return predictions.iloc[0].to_dict()


if __name__ == "__main__":
    predictor = NBAPredictor()
    predictions = predictor.predict_today(min_minutes=15)
    
    if predictions is not None:
        print(predictor.format_predictions(predictions, top_n=30))
        predictor.save_predictions(predictions)
