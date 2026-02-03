"""
Feature Pipeline
Orchestrates the complete feature engineering process
"""

import pandas as pd
import numpy as np
import yaml
import os
from src.features.player_features import PlayerFeatureEngineer
from src.features.matchup_features import MatchupFeatureEngineer


class FeaturePipeline:
    """Complete feature engineering pipeline"""
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize pipeline with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.processed_dir = self.config['paths']['processed_dir']
        self.rolling_windows = self.config['features']['rolling_windows']
        
        # Initialize feature engineers
        self.player_engineer = PlayerFeatureEngineer(rolling_windows=self.rolling_windows)
        self.matchup_engineer = MatchupFeatureEngineer()
    
    def load_processed_data(self):
        """
        Load processed game logs
        
        Returns:
            pd.DataFrame: Processed data
        """
        print("\n" + "="*60)
        print("LOADING PROCESSED DATA")
        print("="*60)
        
        file_path = f"{self.processed_dir}/processed_game_logs.csv"
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Processed data not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        
        print(f"✓ Loaded: {file_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Players: {df['PLAYER_NAME'].nunique()}")
        print(f"  Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
        
        return df
    
    def remove_early_games(self, df, min_games=5):
        """
        Remove early games where we don't have enough history for features
        
        Args:
            df (pd.DataFrame): Data with features
            min_games (int): Minimum games needed for reliable features
            
        Returns:
            pd.DataFrame: Filtered data
        """
        print(f"\n" + "="*60)
        print(f"FILTERING EARLY GAMES (need {min_games}+ history)")
        print("="*60)
        
        initial_rows = len(df)
        
        # Calculate game number for each player in each season
        df['PLAYER_GAME_NUM'] = df.groupby(['PLAYER_ID', 'SEASON']).cumcount() + 1
        
        # Keep only games where player has played at least min_games
        df = df[df['PLAYER_GAME_NUM'] > min_games].copy()
        
        # Drop the helper column
        df = df.drop('PLAYER_GAME_NUM', axis=1)
        
        rows_removed = initial_rows - len(df)
        print(f"✓ Removed {rows_removed:,} early games ({rows_removed/initial_rows*100:.1f}%)")
        print(f"✓ Remaining: {len(df):,} games")
        
        return df
    
    def handle_missing_values(self, df):
        """
        Handle any remaining missing values in features
        
        Args:
            df (pd.DataFrame): Data with features
            
        Returns:
            pd.DataFrame: Data with no missing values in feature columns
        """
        print("\n" + "="*60)
        print("HANDLING MISSING VALUES")
        print("="*60)
        
        # Identify feature columns (exclude identifiers and targets)
        id_cols = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'GAME_ID', 
                   'GAME_DATE', 'MATCHUP', 'OPPONENT', 'SEASON']
        target_cols = ['PTS', 'REB', 'AST']
        
        feature_cols = [col for col in df.columns if col not in id_cols + target_cols]
        
        # Check for missing values
        missing = df[feature_cols].isnull().sum()
        missing = missing[missing > 0]
        
        if len(missing) > 0:
            print(f"Found missing values in {len(missing)} columns:")
            for col, count in missing.items():
                print(f"  - {col}: {count} ({count/len(df)*100:.1f}%)")
            
            # Fill missing values with forward fill, then backward fill, then 0
            df[feature_cols] = df.groupby('PLAYER_ID')[feature_cols].fillna(method='ffill')
            df[feature_cols] = df.groupby('PLAYER_ID')[feature_cols].fillna(method='bfill')
            df[feature_cols] = df[feature_cols].fillna(0)
            
            print("\n✓ Filled missing values using forward/backward fill + zero")
        else:
            print("✓ No missing values found!")
        
        return df
    
    def create_training_sets(self, df):
        """
        Split data into training and testing sets based on seasons
        
        Args:
            df (pd.DataFrame): Complete feature dataset
            
        Returns:
            tuple: (train_df, test_df)
        """
        print("\n" + "="*60)
        print("CREATING TRAINING/TESTING SPLITS")
        print("="*60)
        
        train_seasons = self.config['data']['train_seasons']
        test_season = self.config['data']['test_season']
        
        # Training data
        train_df = df[df['SEASON'].isin(train_seasons)].copy()
        print(f"\nTraining Set:")
        print(f"  Seasons: {', '.join(train_seasons)}")
        print(f"  Games: {len(train_df):,}")
        print(f"  Players: {train_df['PLAYER_NAME'].nunique()}")
        
        # Testing data
        test_df = df[df['SEASON'] == test_season].copy()
        print(f"\nTesting Set:")
        print(f"  Season: {test_season}")
        print(f"  Games: {len(test_df):,}")
        print(f"  Players: {test_df['PLAYER_NAME'].nunique()}")
        
        return train_df, test_df
    
    def get_feature_names(self, df):
        """
        Get list of feature column names
        
        Args:
            df (pd.DataFrame): Data with features
            
        Returns:
            list: Feature column names
        """
        # Columns to exclude from features
        exclude_cols = [
            'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'GAME_ID',
            'GAME_DATE', 'MATCHUP', 'OPPONENT', 'SEASON',
            'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',  # Raw stats
            'FG_PCT', 'FG3_PCT', 'FT_PCT', 'PLUS_MINUS'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return feature_cols
    
    def run_pipeline(self):
        """
        Run the complete feature engineering pipeline
        
        Returns:
            tuple: (train_df, test_df, feature_names)
        """
        print("\n" + "="*70)
        print(" "*20 + "FEATURE ENGINEERING PIPELINE")
        print("="*70)
        
        # Load data
        df = self.load_processed_data()
        
        # Create player features
        df = self.player_engineer.create_all_player_features(df)
        
        # Create matchup features
        df = self.matchup_engineer.create_all_matchup_features(df)
        
        # Remove early games without enough history
        df = self.remove_early_games(df, min_games=5)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Get feature names
        feature_names = self.get_feature_names(df)
        
        # Create training/testing splits
        train_df, test_df = self.create_training_sets(df)
        
        # Save datasets
        print("\n" + "="*60)
        print("SAVING FEATURE DATASETS")
        print("="*60)
        
        train_path = f"{self.processed_dir}/train_features.csv"
        test_path = f"{self.processed_dir}/test_features.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"✓ Training set saved: {train_path}")
        print(f"✓ Testing set saved: {test_path}")
        
        # Save feature names
        feature_list_path = f"{self.processed_dir}/feature_names.txt"
        with open(feature_list_path, 'w') as f:
            for feature in feature_names:
                f.write(f"{feature}\n")
        
        print(f"✓ Feature names saved: {feature_list_path}")
        
        # Final summary
        print("\n" + "="*70)
        print(" "*20 + "FEATURE ENGINEERING COMPLETE!")
        print("="*70)
        print(f"\n📊 Dataset Summary:")
        print(f"  Total features: {len(feature_names)}")
        print(f"  Training games: {len(train_df):,}")
        print(f"  Testing games: {len(test_df):,}")
        print(f"  Total games: {len(train_df) + len(test_df):,}")
        
        print(f"\n🎯 Target Variables:")
        for target in ['PTS', 'REB', 'AST']:
            print(f"  {target}:")
            print(f"    Train mean: {train_df[target].mean():.2f}")
            print(f"    Test mean: {test_df[target].mean():.2f}")
        
        print("\n✅ Ready for model training!")
        
        return train_df, test_df, feature_names


if __name__ == "__main__":
    pipeline = FeaturePipeline()
    train_df, test_df, feature_names = pipeline.run_pipeline()