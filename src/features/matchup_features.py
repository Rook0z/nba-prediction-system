import pandas as pd
import numpy as np


class MatchupFeatureEngineer:
    """Generate matchup and opponent-specific features"""
    
    def __init__(self):
        """Initialize matchup feature engineer"""
        self.target_stats = ['PTS', 'REB', 'AST']
    
    def create_opponent_defense_features(self, df):
        """
        Create opponent defensive strength features
        
        Args:
            df (pd.DataFrame): Game logs with opponent stats
            
        Returns:
            pd.DataFrame: Data with opponent defensive features
        """
        print("\n" + "="*60)
        print("CREATING OPPONENT DEFENSIVE FEATURES")
        print("="*60)
        
        # Check if opponent stats exist
        opp_stat_cols = [col for col in df.columns if col.startswith('OPP_')]
        
        if not opp_stat_cols:
            print("⚠️  No opponent stats found, skipping opponent features")
            return df
        
        # Rename opponent columns for clarity
        opponent_features = []
        
        if 'OPP_AVG_PTS' in df.columns:
            df['OPP_DEF_RATING_PTS'] = df['OPP_AVG_PTS']  # Points allowed
            opponent_features.append('Points defense')
        
        if 'OPP_AVG_REB' in df.columns:
            df['OPP_DEF_RATING_REB'] = df['OPP_AVG_REB']  # Rebounds allowed
            opponent_features.append('Rebounds defense')
        
        if 'OPP_AVG_AST' in df.columns:
            df['OPP_DEF_RATING_AST'] = df['OPP_AVG_AST']  # Assists allowed
            opponent_features.append('Assists defense')
        
        print(f"✓ Created {len(opponent_features)} opponent defensive features")
        for feature in opponent_features:
            print(f"  - {feature}")
        
        return df
    
    def create_head_to_head_features(self, df):
        """
        Create head-to-head performance features vs specific opponents
        
        Args:
            df (pd.DataFrame): Game logs
            
        Returns:
            pd.DataFrame: Data with head-to-head features
        """
        print("\n" + "="*60)
        print("CREATING HEAD-TO-HEAD FEATURES")
        print("="*60)
        
        for stat in self.target_stats:
            df[f'{stat}_VS_OPP_L5'] = (
                df.groupby(['PLAYER_ID', 'OPPONENT'])[stat]
                .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
            )
            
            # Fill missing values with overall average
            df[f'{stat}_VS_OPP_L5'] = df[f'{stat}_VS_OPP_L5'].fillna(df[f'{stat}_L10'])
        
        print("✓ Performance vs specific opponents (last 5 matchups)")
        print(f"✅ Created {len(self.target_stats)} head-to-head features")
        
        return df
    
    def create_matchup_difficulty_features(self, df):
        """
        Create features for matchup difficulty
        
        Args:
            df (pd.DataFrame): Game logs with opponent stats
            
        Returns:
            pd.DataFrame: Data with matchup difficulty features
        """
        print("\n" + "="*60)
        print("CREATING MATCHUP DIFFICULTY FEATURES")
        print("="*60)
        
        features_created = 0
        
        # For each stat, compare opponent's defensive rating to league average
        for stat in self.target_stats:
            opp_col = f'OPP_DEF_RATING_{stat}'
            
            if opp_col in df.columns:
                df[f'{stat}_MATCHUP_DIFFICULTY'] = league_avg - df[opp_col]
                features_created += 1
        
        if features_created > 0:
            print(f"✓ Matchup difficulty (opponent vs league average)")
            print(f"✅ Created {features_created} matchup difficulty features")
        else:
            print("⚠️  No opponent stats available for matchup difficulty")
        
        return df
    
    def create_pace_features(self, df):
        """
        Create game pace features (if opponent pace data available)
        
        Args:
            df (pd.DataFrame): Game logs
            
        Returns:
            pd.DataFrame: Data with pace features
        """
        print("\n" + "="*60)
        print("CREATING PACE FEATURES")
        print("="*60)
        
        # Check if plus/minus exists as proxy for pace
        if 'OPP_AVG_PLUS_MINUS' in df.columns:
            df['OPP_PACE_PROXY'] = df['OPP_AVG_PLUS_MINUS']
            print("✓ Opponent pace proxy (plus/minus)")
            print("✅ Created 1 pace feature")
        else:
            print("⚠️  No pace data available")
        
        return df
    
    def create_recent_opponent_form(self, df):
        """
        Create features for opponent's recent form
        
        Args:
            df (pd.DataFrame): Game logs
            
        Returns:
            pd.DataFrame: Data with opponent form features
        """
        print("\n" + "="*60)
        print("CREATING OPPONENT RECENT FORM FEATURES")
        print("="*60)
        
        features_created = 0
        
        for stat in self.target_stats:
            opp_col = f'OPP_DEF_RATING_{stat}'
            
            if opp_col in df.columns:
                df[f'{stat}_OPP_DEF_TREND'] = (
                    df.groupby(['OPPONENT', 'SEASON'])[opp_col]
                    .transform(lambda x: x.rolling(window=5, min_periods=1).mean() - 
                                        x.rolling(window=15, min_periods=3).mean())
                )
                features_created += 1
        
        if features_created > 0:
            print(f"✓ Opponent defensive trends")
            print(f"✅ Created {features_created} opponent form features")
        else:
            print("⚠️  No opponent stats available for form features")
        
        return df
    
    def create_all_matchup_features(self, df):
        """
        Create all matchup-related features
        
        Args:
            df (pd.DataFrame): Game logs with opponent info
            
        Returns:
            pd.DataFrame: Data with all matchup features
        """
        print("\n" + "="*60)
        print("MATCHUP FEATURE ENGINEERING PIPELINE")
        print("="*60)
        print(f"Input shape: {df.shape}")
        
        initial_cols = df.shape[1]
        
        # Create features
        df = self.create_opponent_defense_features(df)
        df = self.create_head_to_head_features(df)
        df = self.create_matchup_difficulty_features(df)
        df = self.create_pace_features(df)
        df = self.create_recent_opponent_form(df)
        
        # Calculate new features
        new_features = df.shape[1] - initial_cols
        
        print("\n" + "="*60)
        print("MATCHUP FEATURE ENGINEERING COMPLETE")
        print("="*60)
        print(f"Output shape: {df.shape}")
        print(f"New matchup features created: {new_features}")
        
        return df
