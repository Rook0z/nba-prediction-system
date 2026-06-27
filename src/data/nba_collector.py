import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelogs, leaguegamefinder, teamgamelogs
from nba_api.stats.static import players, teams
import time
import os
from datetime import datetime
import yaml


class NBADataCollector:
    """Collects NBA player and team data for specified seasons"""
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize collector with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.seasons = self.config['data']['seasons']
        self.raw_dir = self.config['paths']['raw_dir']
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        dirs = [
            f"{self.raw_dir}/game_logs",
            f"{self.raw_dir}/team_stats",
            f"{self.raw_dir}/schedules"
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def _season_to_api_format(self, season):
        """Convert season format: '2023-24' -> '2023-24'"""
        return season
    
    def collect_player_game_logs(self, season):
        """
        Collect all player game logs for a season
        
        Args:
            season (str): Season in format '2023-24'
            
        Returns:
            pd.DataFrame: Player game logs
        """
        print(f"\n{'='*50}")
        print(f"Collecting player game logs for {season}...")
        print(f"{'='*50}")
        
        try:
            # Fetch game logs using NBA API
            game_logs = playergamelogs.PlayerGameLogs(
                season_nullable=season,
                season_type_nullable='Regular Season'
            )
            
            df = game_logs.get_data_frames()[0]
            
            # Save raw data
            filename = f"{self.raw_dir}/game_logs/player_logs_{season}.csv"
            df.to_csv(filename, index=False)
            
            print(f"✓ Collected {len(df)} game logs")
            print(f"✓ Saved to: {filename}")
            print(f"✓ Unique players: {df['PLAYER_NAME'].nunique()}")
            
            return df
            
        except Exception as e:
            print(f"✗ Error collecting game logs for {season}: {str(e)}")
            return None
    
    def collect_team_stats(self, season):
        """
        Collect team statistics for a season
        
        Args:
            season (str): Season in format '2023-24'
            
        Returns:
            pd.DataFrame: Team statistics
        """
        print(f"\nCollecting team stats for {season}...")
        
        try:
            # Fetch team game logs
            team_logs = teamgamelogs.TeamGameLogs(
                season_nullable=season,
                season_type_nullable='Regular Season'
            )
            
            df = team_logs.get_data_frames()[0]
            
            # Calculate team averages and defensive metrics
            team_stats = df.groupby('TEAM_NAME').agg({
                'PTS': 'mean',
                'REB': 'mean',
                'AST': 'mean',
                'FG_PCT': 'mean',
                'FG3_PCT': 'mean',
                'FT_PCT': 'mean',
                'PLUS_MINUS': 'mean'
            }).reset_index()
            
            team_stats.columns = ['TEAM_NAME', 'AVG_PTS', 'AVG_REB', 'AVG_AST', 
                                 'AVG_FG_PCT', 'AVG_FG3_PCT', 'AVG_FT_PCT', 'AVG_PLUS_MINUS']
            
            # Save team stats
            filename = f"{self.raw_dir}/team_stats/team_stats_{season}.csv"
            team_stats.to_csv(filename, index=False)
            
            print(f"✓ Collected stats for {len(team_stats)} teams")
            print(f"✓ Saved to: {filename}")
            
            return team_stats
            
        except Exception as e:
            print(f"✗ Error collecting team stats for {season}: {str(e)}")
            return None
    
    def collect_schedules(self, season):
        """
        Collect game schedules for calculating rest days
        
        Args:
            season (str): Season in format '2023-24'
            
        Returns:
            pd.DataFrame: Game schedules
        """
        print(f"\nCollecting schedules for {season}...")
        
        try:
            # Use LeagueGameFinder to get all games
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable='Regular Season'
            )
            
            games = gamefinder.get_data_frames()[0]
            
            # Select relevant columns
            schedule = games[['GAME_ID', 'GAME_DATE', 'TEAM_NAME', 'MATCHUP']].copy()
            schedule['GAME_DATE'] = pd.to_datetime(schedule['GAME_DATE'])
            
            # Save schedules
            filename = f"{self.raw_dir}/schedules/schedule_{season}.csv"
            schedule.to_csv(filename, index=False)
            
            print(f"✓ Collected {len(schedule)} games")
            print(f"✓ Saved to: {filename}")
            
            return schedule
            
        except Exception as e:
            print(f"✗ Error collecting schedules for {season}: {str(e)}")
            return None
    
    def collect_all_seasons(self):
        """
        Collect all data for all configured seasons
        
        Returns:
            dict: Dictionary containing all collected dataframes
        """
        print("\n" + "="*60)
        print("NBA DATA COLLECTION STARTING")
        print("="*60)
        print(f"Seasons to collect: {', '.join(self.seasons)}")
        print(f"Target stats: {', '.join(self.config['target_stats'])}")
        
        all_data = {
            'game_logs': [],
            'team_stats': [],
            'schedules': []
        }
        
        for season in self.seasons:
            print(f"\n{'='*60}")
            print(f"Processing Season: {season}")
            print(f"{'='*60}")
            
            # Collect player game logs
            game_logs = self.collect_player_game_logs(season)
            if game_logs is not None:
                all_data['game_logs'].append(game_logs)
            
            # Wait to avoid rate limiting
            time.sleep(2)
            
            # Collect team stats
            team_stats = self.collect_team_stats(season)
            if team_stats is not None:
                all_data['team_stats'].append(team_stats)
            
            time.sleep(2)
            
            # Collect schedules
            schedules = self.collect_schedules(season)
            if schedules is not None:
                all_data['schedules'].append(schedules)
            
            time.sleep(2)
        
        # Combine all seasons
        print("\n" + "="*60)
        print("COMBINING ALL SEASONS")
        print("="*60)
        
        combined_data = {}
        
        if all_data['game_logs']:
            combined_data['game_logs'] = pd.concat(all_data['game_logs'], ignore_index=True)
            print(f"✓ Total game logs: {len(combined_data['game_logs'])}")
            
        if all_data['team_stats']:
            combined_data['team_stats'] = pd.concat(all_data['team_stats'], ignore_index=True)
            print(f"✓ Total team stats records: {len(combined_data['team_stats'])}")
            
        if all_data['schedules']:
            combined_data['schedules'] = pd.concat(all_data['schedules'], ignore_index=True)
            print(f"✓ Total schedule records: {len(combined_data['schedules'])}")
        
        print("\n" + "="*60)
        print("DATA COLLECTION COMPLETE!")
        print("="*60)
        
        return combined_data
    
    def get_summary_stats(self):
        """Print summary statistics of collected data"""
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        
        for season in self.seasons:
            game_log_file = f"{self.raw_dir}/game_logs/player_logs_{season}.csv"
            if os.path.exists(game_log_file):
                df = pd.read_csv(game_log_file)
                print(f"\n{season}:")
                print(f"  - Total games: {len(df)}")
                print(f"  - Unique players: {df['PLAYER_NAME'].nunique()}")
                print(f"  - Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")


if __name__ == "__main__":
    collector = NBADataCollector()
    
    # Collect all data
    data = collector.collect_all_seasons()
    
    # Show summary
    collector.get_summary_stats()
