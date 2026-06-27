import pandas as pd
import numpy as np


class PlayerFeatureEngineer:
    """Generate player-level statistical features"""
    
    def __init__(self, rolling_windows=[5, 10, 15]):
        """
        Initialize feature engineer
        
        Args:
            rolling_windows (list): Windows for rolling averages
        """
        self.rolling_windows = rolling_windows
        self.target_stats = ['PTS', 'REB', 'AST']
        self.additional_stats = ['MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'PLUS_MINUS']
    
    def create_rolling_averages(self, df):
        """
        Create rolling average features for all target statistics
        
        Args:
            df (pd.DataFrame): Game logs sorted by player and date
            
        Returns:
            pd.DataFrame: Data with rolling average features
        """
        print("\n" + "="*60)
        print("CREATING ROLLING AVERAGES")
        print("="*60)
        
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
        r
        all_stats = self.target_stats + self.additional_stats
        
        for window in self.rolling_windows:
            print(f"\nCalculating {window}-game rolling averages...")
            
            for stat in all_stats:
                if stat in df.columns:
                    col_name = f'{stat}_L{window}'
                    
                    # Calculate rolling average (excluding current game)
                    df[col_name] = (
                        df.groupby('PLAYER_ID')[stat]
                        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
                    )
                    
            print(f"  ✓ Created {len(all_stats)} features for {window}-game window")
        
        total_features = len(all_stats) * len(self.rolling_windows)
        print(f"\n✅ Total rolling average features created: {total_features}")
        
        return df
    
    def create_trend_features(self, df):
        """
        Create features showing performance trends
        
        Args:
            df (pd.DataFrame): Data with rolling averages
            
        Returns:
            pd.DataFrame: Data with trend features
        """
        print("\n" + "="*60)
        print("CREATING TREND FEATURES")
        print("="*60)
        
        for stat in self.target_stats:
            # Trend: difference between short-term and long-term average
            if f'{stat}_L5' in df.columns and f'{stat}_L15' in df.columns:
                df[f'{stat}_TREND'] = df[f'{stat}_L5'] - df[f'{stat}_L15']
            
            # Recent form: last 5 games vs season average
            if f'{stat}_L5' in df.columns:
                season_avg = df.groupby(['PLAYER_ID', 'SEASON'])[stat].transform('mean')
                df[f'{stat}_RECENT_FORM'] = df[f'{stat}_L5'] - season_avg
        
        print("✓ Trend features (short vs long term)")
        print("✓ Recent form features (recent vs season average)")
        print(f"✅ Created {len(self.target_stats) * 2} trend features")
        
        return df
    
    def create_consistency_features(self, df):
        """
        Create features measuring player consistency
        
        Args:
            df (pd.DataFrame): Game logs
            
        Returns:
            pd.DataFrame: Data with consistency features
        """
        print("\n" + "="*60)
        print("CREATING CONSISTENCY FEATURES")
        print("="*60)
        
        for stat in self.target_stats:
            # Standard deviation over last 10 games
            df[f'{stat}_STD_L10'] = (
                df.groupby('PLAYER_ID')[stat]
                .transform(lambda x: x.shift(1).rolling(window=10, min_periods=3).std())
            )
            
            # Coefficient of variation (std/mean) - relative consistency
            if f'{stat}_L10' in df.columns:
                df[f'{stat}_CV_L10'] = df[f'{stat}_STD_L10'] / (df[f'{stat}_L10'] + 0.1)
        
        print("✓ Standard deviation features")
        print("✓ Coefficient of variation (relative consistency)")
        print(f"✅ Created {len(self.target_stats) * 2} consistency features")
        
        return df
    
    def create_home_away_features(self, df):
        """
        Create home/away performance features
        
        Args:
            df (pd.DataFrame): Game logs
            
        Returns:
            pd.DataFrame: Data with home/away features
        """
        print("\n" + "="*60)
        print("CREATING HOME/AWAY FEATURES")
        print("="*60)
        
        for stat in self.target_stats:
            # Home average (last 10 home games)
            df[f'{stat}_HOME_L10'] = (
                df[df['IS_HOME'] == 1].groupby('PLAYER_ID')[stat]
                .transform(lambda x: x.shift(1).rolling(window=10, min_periods=3).mean())
            )
            
            # Away average (last 10 away games)
            df[f'{stat}_AWAY_L10'] = (
                df[df['IS_HOME'] == 0].groupby('PLAYER_ID')[stat]
                .transform(lambda x: x.shift(1).rolling(window=10, min_periods=3).mean())
            )
            
            # Fill NaN with overall average for players without enough home/away games
            df[f'{stat}_HOME_L10'] = df[f'{stat}_HOME_L10'].fillna(df[f'{stat}_L10'])
            df[f'{stat}_AWAY_L10'] = df[f'{stat}_AWAY_L10'].fillna(df[f'{stat}_L10'])
        
        print("✓ Home performance averages")
        print("✓ Away performance averages")
        print(f"✅ Created {len(self.target_stats) * 2} home/away features")
        
        return df
    
    def create_rest_features(self, df):
        """
        Create features related to rest and fatigue
        
        Args:
            df (pd.DataFrame): Game logs with REST_DAYS
            
        Returns:
            pd.DataFrame: Data with rest-related features
        """
        print("\n" + "="*60)
        print("CREATING REST/FATIGUE FEATURES")
        print("="*60)
        
        # Rest days categories
        df['REST_0_1'] = (df['REST_DAYS'] <= 1).astype(int)  # Back-to-back or 1 day
        df['REST_2_3'] = ((df['REST_DAYS'] >= 2) & (df['REST_DAYS'] <= 3)).astype(int)
        df['REST_4_PLUS'] = (df['REST_DAYS'] >= 4).astype(int)
        
        # Performance on back-to-backs (if player has played them)
        for stat in self.target_stats:
            df[f'{stat}_ON_B2B'] = (
                df[df['IS_BACK_TO_BACK'] == 1].groupby('PLAYER_ID')[stat]
                .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
            )
            # Fill with regular average if no B2B history
            df[f'{stat}_ON_B2B'] = df[f'{stat}_ON_B2B'].fillna(df[f'{stat}_L10'])
        
        # Games in last 7 days (schedule density)
        # Calculate manually since time-based rolling doesn't work in groupby
        df['GAMES_IN_L7'] = 0
        for player_id in df['PLAYER_ID'].unique():
            player_mask = df['PLAYER_ID'] == player_id
            player_dates = df.loc[player_mask, 'GAME_DATE'].values
            
            games_in_7d = []
            for i, date in enumerate(player_dates):
                if i == 0:
                    games_in_7d.append(0)
                else:
                    # Count games in previous 7 days
                    prev_dates = player_dates[:i]
                    date_diffs = (date - prev_dates) / pd.Timedelta(days=1)
                    count = np.sum(date_diffs <= 7)
                    games_in_7d.append(count)
            
            df.loc[player_mask, 'GAMES_IN_L7'] = games_in_7d
        
        print("✓ Rest day categories")
        print("✓ Back-to-back performance")
        print("✓ Schedule density (games in last 7 days)")
        print(f"✅ Created {3 + len(self.target_stats) + 1} rest/fatigue features")
        
        return df
    
    def create_season_position_features(self, df):
        """
        Create features based on position in season
        
        Args:
            df (pd.DataFrame): Game logs
            
        Returns:
            pd.DataFrame: Data with season position features
        """
        print("\n" + "="*60)
        print("CREATING SEASON POSITION FEATURES")
        print("="*60)
        
        # Game number in season for each player
        df['GAME_NUM_SEASON'] = df.groupby(['PLAYER_ID', 'SEASON']).cumcount() + 1
        
        # Early season indicator (first 10 games)
        df['IS_EARLY_SEASON'] = (df['GAME_NUM_SEASON'] <= 10).astype(int)
        
        # Late season indicator (last 15 games)
        season_game_counts = df.groupby(['PLAYER_ID', 'SEASON'])['GAME_NUM_SEASON'].transform('max')
        df['IS_LATE_SEASON'] = (df['GAME_NUM_SEASON'] > (season_game_counts - 15)).astype(int)
        
        print("✓ Game number in season")
        print("✓ Early season indicator")
        print("✓ Late season indicator")
        print("✅ Created 3 season position features")
        
        return df
    
    def create_all_player_features(self, df):
        """
        Create all player-level features
        
        Args:
            df (pd.DataFrame): Processed game logs
            
        Returns:
            pd.DataFrame: Data with all player features
        """
        print("\n" + "="*60)
        print("PLAYER FEATURE ENGINEERING PIPELINE")
        print("="*60)
        print(f"Input shape: {df.shape}")
        print(f"Players: {df['PLAYER_NAME'].nunique()}")
        
        # Create features in order
        df = self.create_rolling_averages(df)
        df = self.create_trend_features(df)
        df = self.create_consistency_features(df)
        df = self.create_home_away_features(df)
        df = self.create_rest_features(df)
        df = self.create_season_position_features(df)
        
        # Count total features created
        original_cols = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'GAME_ID', 
                        'GAME_DATE', 'MATCHUP', 'OPPONENT', 'IS_HOME', 'MIN',
                        'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 
                        'FG3_PCT', 'FT_PCT', 'PLUS_MINUS', 'SEASON', 'REST_DAYS',
                        'IS_BACK_TO_BACK']
        
        new_features = [col for col in df.columns if col not in original_cols]
        
        print("\n" + "="*60)
        print("PLAYER FEATURE ENGINEERING COMPLETE")
        print("="*60)
        print(f"Output shape: {df.shape}")
        print(f"New features created: {len(new_features)}")
        print(f"\nFeature categories:")
        print(f"  - Rolling averages: {len(self.target_stats + self.additional_stats) * len(self.rolling_windows)}")
        print(f"  - Trends: {len(self.target_stats) * 2}")
        print(f"  - Consistency: {len(self.target_stats) * 2}")
        print(f"  - Home/Away: {len(self.target_stats) * 2}")
        print(f"  - Rest/Fatigue: {3 + len(self.target_stats) + 1}")
        print(f"  - Season position: 3")
        
        return df
