"""
Data Processor
Cleans, filters, and prepares NBA data for feature engineering
"""

import pandas as pd
import numpy as np
import yaml
import os
from datetime import datetime


class NBADataProcessor:
    """Processes raw NBA data into clean, usable format"""
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize processor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_dir = self.config['paths']['raw_dir']
        self.processed_dir = self.config['paths']['processed_dir']
        self.min_games = self.config['data']['min_games']
        self.min_minutes = self.config['data']['min_minutes']
        
        self._create_directories()
    
    def _create_directories(self):
        """Create processed data directory"""
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def load_raw_data(self):
        """
        Load all raw data files
        
        Returns:
            dict: Dictionary containing game_logs, team_stats, schedules
        """
        print("\n" + "="*60)
        print("LOADING RAW DATA")
        print("="*60)
        
        data = {}
        
        # Load game logs
        game_log_files = [f"{self.raw_dir}/game_logs/player_logs_{season}.csv" 
                         for season in self.config['data']['seasons']]
        game_logs = []
        for file in game_log_files:
            if os.path.exists(file):
                df = pd.read_csv(file)
                game_logs.append(df)
                print(f"✓ Loaded: {file}")
        
        if game_logs:
            data['game_logs'] = pd.concat(game_logs, ignore_index=True)
            print(f"\nTotal game logs loaded: {len(data['game_logs'])}")
        
        # Load team stats
        team_stat_files = [f"{self.raw_dir}/team_stats/team_stats_{season}.csv" 
                          for season in self.config['data']['seasons']]
        team_stats = []
        for file in team_stat_files:
            if os.path.exists(file):
                df = pd.read_csv(file)
                team_stats.append(df)
        
        if team_stats:
            data['team_stats'] = pd.concat(team_stats, ignore_index=True)
        
        # Load schedules
        schedule_files = [f"{self.raw_dir}/schedules/schedule_{season}.csv" 
                         for season in self.config['data']['seasons']]
        schedules = []
        for file in schedule_files:
            if os.path.exists(file):
                df = pd.read_csv(file)
                schedules.append(df)
        
        if schedules:
            data['schedules'] = pd.concat(schedules, ignore_index=True)
        
        return data
    
    def clean_game_logs(self, df):
        """
        Clean and filter player game logs
        
        Args:
            df (pd.DataFrame): Raw game logs
            
        Returns:
            pd.DataFrame: Cleaned game logs
        """
        print("\n" + "="*60)
        print("CLEANING GAME LOGS")
        print("="*60)
        
        print(f"Initial records: {len(df)}")
        
        # Convert date to datetime
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        
        # Sort by player and date
        df = df.sort_values(['PLAYER_NAME', 'GAME_DATE']).reset_index(drop=True)
        
        # Remove duplicate games (sometimes API returns duplicates)
        df = df.drop_duplicates(subset=['PLAYER_ID', 'GAME_ID'], keep='first')
        print(f"After removing duplicates: {len(df)}")
        
        # Filter by minimum minutes
        df = df[df['MIN'] >= self.min_minutes].copy()
        print(f"After min minutes filter ({self.min_minutes}+ mins): {len(df)}")
        
        # Calculate games played per player
        games_per_player = df.groupby('PLAYER_NAME').size()
        valid_players = games_per_player[games_per_player >= self.min_games].index
        df = df[df['PLAYER_NAME'].isin(valid_players)].copy()
        print(f"After min games filter ({self.min_games}+ games): {len(df)}")
        print(f"Unique players remaining: {df['PLAYER_NAME'].nunique()}")
        
        # Extract home/away from matchup
        df['IS_HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
        
        # Extract opponent team
        df['OPPONENT'] = df['MATCHUP'].apply(self._extract_opponent)
        
        # Add season column
        df['SEASON'] = df['SEASON_YEAR']
        
        # Select and rename key columns
        key_columns = [
            'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'GAME_ID', 
            'GAME_DATE', 'MATCHUP', 'OPPONENT', 'IS_HOME', 'MIN',
            'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 
            'FG3_PCT', 'FT_PCT', 'PLUS_MINUS', 'SEASON'
        ]
        
        df = df[key_columns].copy()
        
        print(f"\n✓ Cleaning complete!")
        print(f"Final dataset: {len(df)} games, {df['PLAYER_NAME'].nunique()} players")
        
        return df
    
    def _extract_opponent(self, matchup):
        """Extract opponent team from matchup string"""
        if 'vs.' in matchup:
            return matchup.split('vs. ')[1]
        elif '@' in matchup:
            return matchup.split('@ ')[1]
        return None
    
    def add_rest_days(self, df):
        """
        Calculate rest days between games for each player
        
        Args:
            df (pd.DataFrame): Game logs
            
        Returns:
            pd.DataFrame: Game logs with rest_days column
        """
        print("\nCalculating rest days between games...")
        
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
        
        # Calculate days since last game
        df['PREV_GAME_DATE'] = df.groupby('PLAYER_ID')['GAME_DATE'].shift(1)
        df['REST_DAYS'] = (df['GAME_DATE'] - df['PREV_GAME_DATE']).dt.days
        
        # Fill first game of season with average rest (2 days)
        df['REST_DAYS'] = df['REST_DAYS'].fillna(2)
        
        # Identify back-to-back games
        df['IS_BACK_TO_BACK'] = (df['REST_DAYS'] <= 1).astype(int)
        
        df = df.drop('PREV_GAME_DATE', axis=1)
        
        print(f"✓ Rest days calculated")
        print(f"  - Back-to-back games: {df['IS_BACK_TO_BACK'].sum()}")
        print(f"  - Average rest days: {df['REST_DAYS'].mean():.2f}")
        
        return df
    
    def merge_opponent_stats(self, game_logs, team_stats):
        """
        Merge opponent defensive statistics
        
        Args:
            game_logs (pd.DataFrame): Player game logs
            team_stats (pd.DataFrame): Team statistics
            
        Returns:
            pd.DataFrame: Game logs with opponent stats
        """
        print("\nMerging opponent statistics...")
        
        # Rename team stats columns for opponent
        opponent_stats = team_stats.copy()
        opponent_stats.columns = ['OPP_' + col if col != 'TEAM_NAME' else 'OPPONENT' 
                                  for col in opponent_stats.columns]
        
        # Merge opponent stats
        df = game_logs.merge(opponent_stats, on='OPPONENT', how='left')
        
        print(f"✓ Opponent stats merged")
        
        return df
    
    def process_all_data(self):
        """
        Complete data processing pipeline
        
        Returns:
            pd.DataFrame: Fully processed dataset
        """
        print("\n" + "="*60)
        print("STARTING DATA PROCESSING PIPELINE")
        print("="*60)
        
        # Load raw data
        raw_data = self.load_raw_data()
        
        if 'game_logs' not in raw_data:
            print("✗ No game logs found!")
            return None
        
        # Clean game logs
        clean_logs = self.clean_game_logs(raw_data['game_logs'])
        
        # Add rest days
        clean_logs = self.add_rest_days(clean_logs)
        
        # Merge opponent stats if available
        if 'team_stats' in raw_data:
            clean_logs = self.merge_opponent_stats(clean_logs, raw_data['team_stats'])
        
        # Save processed data
        output_file = f"{self.processed_dir}/processed_game_logs.csv"
        clean_logs.to_csv(output_file, index=False)
        
        print("\n" + "="*60)
        print("DATA PROCESSING COMPLETE!")
        print("="*60)
        print(f"✓ Processed data saved to: {output_file}")
        print(f"✓ Total records: {len(clean_logs)}")
        print(f"✓ Players: {clean_logs['PLAYER_NAME'].nunique()}")
        print(f"✓ Date range: {clean_logs['GAME_DATE'].min()} to {clean_logs['GAME_DATE'].max()}")
        
        # Show sample stats
        print("\nTarget Stats Summary:")
        for stat in self.config['target_stats']:
            stat_col = stat.upper()
            if stat_col in clean_logs.columns:
                print(f"  {stat.upper()}:")
                print(f"    Mean: {clean_logs[stat_col].mean():.2f}")
                print(f"    Median: {clean_logs[stat_col].median():.2f}")
                print(f"    Std: {clean_logs[stat_col].std():.2f}")
        
        return clean_logs
    
    def get_data_quality_report(self, df):
        """
        Generate data quality report
        
        Args:
            df (pd.DataFrame): Processed data
        """
        print("\n" + "="*60)
        print("DATA QUALITY REPORT")
        print("="*60)
        
        print(f"\nDataset Shape: {df.shape}")
        print(f"\nMissing Values:")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) == 0:
            print("  ✓ No missing values!")
        else:
            print(missing)
        
        print(f"\nDuplicate Rows: {df.duplicated().sum()}")
        
        print(f"\nDate Range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
        
        print(f"\nTop 10 Players by Games:")
        top_players = df.groupby('PLAYER_NAME').size().sort_values(ascending=False).head(10)
        for player, games in top_players.items():
            print(f"  {player}: {games} games")


# Usage example
if __name__ == "__main__":
    processor = NBADataProcessor()
    
    # Process all data
    processed_data = processor.process_all_data()
    
    # Generate quality report
    if processed_data is not None:
        processor.get_data_quality_report(processed_data)