import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nba_api.stats.endpoints import scoreboardv2, playergamelog
from nba_api.stats.static import players
import glob
import time
import os


class PredictionValidator:
    """Validates predictions against actual game results"""
    
    def __init__(self, predictions_dir='./results', results_dir='./results'):
        """
        Initialize validator
        
        Args:
            predictions_dir (str): Directory containing prediction files
            results_dir (str): Directory to save validation results
        """
        self.predictions_dir = predictions_dir
        self.results_dir = results_dir
        
        os.makedirs(results_dir, exist_ok=True)
    
    def load_predictions(self, prediction_date=None):
        """
        Load predictions for a specific date
        
        Args:
            prediction_date (str): Date in YYYYMMDD format (default: today)
            
        Returns:
            pd.DataFrame: Predictions
        """
        if prediction_date is None:
            prediction_date = datetime.now().strftime('%Y%m%d')
        
        print("\n" + "="*60)
        print("LOADING PREDICTIONS")
        print("="*60)
        
        prediction_file = f"{self.predictions_dir}/predictions_{prediction_date}.csv"
        
        if not os.path.exists(prediction_file):
            print(f"✗ Prediction file not found: {prediction_file}")
            print("\nTrying to find most recent prediction file...")
            
            files = glob.glob(f"{self.predictions_dir}/predictions_*.csv")
            if files:
                prediction_file = max(files, key=os.path.getctime)
                print(f"✓ Found: {prediction_file}")
            else:
                raise FileNotFoundError("No prediction files found!")
        
        df = pd.read_csv(prediction_file)
        print(f"✓ Loaded {len(df)} predictions")
        print(f"  Date: {prediction_date}")
        print(f"  Players: {len(df)}")
        
        return df
    
    def get_games_for_date(self, game_date):
        """
        Get all games for a specific date
        
        Args:
            game_date (str): Date in YYYY-MM-DD format
            
        Returns:
            list: Game IDs for that date
        """
        print("\n" + "="*60)
        print(f"FETCHING GAMES FOR {game_date}")
        print("="*60)
        
        try:
            # Get scoreboard for date
            scoreboard = scoreboardv2.ScoreboardV2(game_date=game_date)
            games = scoreboard.get_data_frames()[0]
            
            if len(games) == 0:
                print(f"⚠️  No games found for {game_date}")
                print("   Note: Games may not be completed yet, or date format may be wrong")
                return []
            
            print(f"✓ Found {len(games)} games")
            
            return games['GAME_ID'].unique().tolist()
            
        except Exception as e:
            print(f"✗ Error fetching games: {e}")
            return []
    
    def get_player_stats_for_game(self, player_name, game_date):
        """
        Get actual stats for a player on a specific date
        
        Args:
            player_name (str): Player name
            game_date (str): Game date in YYYY-MM-DD format
            
        Returns:
            dict: Player stats or None
        """
        try:
            # Find player ID
            all_players = players.get_players()
            player = [p for p in all_players if p['full_name'].lower() == player_name.lower()]
            
            if not player:
                return None
            
            player_id = player[0]['id']
            
            # Get player game log
            game_log = playergamelog.PlayerGameLog(
                player_id=player_id,
                season='2024-25'
            )
            
            games_df = game_log.get_data_frames()[0]
            games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
            
            # Filter for specific date
            target_date = pd.to_datetime(game_date)
            game = games_df[games_df['GAME_DATE'] == target_date]
            
            if len(game) == 0:
                return None
            
            game = game.iloc[0]
            
            return {
                'PTS': game['PTS'],
                'REB': game['REB'],
                'AST': game['AST'],
                'MIN': game['MIN'],
                'FGM': game['FGM'],
                'FGA': game['FGA'],
                'MATCHUP': game['MATCHUP']
            }
            
        except Exception as e:
            print(f"  ✗ Error fetching stats for {player_name}: {e}")
            return None
    
    def fetch_actual_results(self, predictions_df, game_date):
        """
        Fetch actual results for all predicted players
        
        Args:
            predictions_df (pd.DataFrame): Predictions
            game_date (str): Date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Predictions with actual results
        """
        print("\n" + "="*60)
        print("FETCHING ACTUAL RESULTS FROM NBA API")
        print("="*60)
        print("⏳ This may take 2-5 minutes due to API rate limits...")
        
        results = []
        
        for idx, row in predictions_df.iterrows():
            player_name = row['Player']
            
            print(f"\n[{idx+1}/{len(predictions_df)}] Fetching: {player_name}")
            
            # Get actual stats
            actual_stats = self.get_player_stats_for_game(player_name, game_date)
            
            if actual_stats:
                results.append({
                    'Player': player_name,
                    'Team': row['Team'],
                    'Matchup': actual_stats.get('MATCHUP', 'N/A'),
                    'Minutes': actual_stats.get('MIN', 0),
                    
                    'PTS_PRED': row['PTS'],
                    'PTS_ACTUAL': actual_stats['PTS'],
                    'PTS_ERROR': abs(row['PTS'] - actual_stats['PTS']),
                    
                    'REB_PRED': row['REB'],
                    'REB_ACTUAL': actual_stats['REB'],
                    'REB_ERROR': abs(row['REB'] - actual_stats['REB']),
                    
                    'AST_PRED': row['AST'],
                    'AST_ACTUAL': actual_stats['AST'],
                    'AST_ERROR': abs(row['AST'] - actual_stats['AST'])
                })
                
                print(f"  ✓ PTS: {row['PTS']:.1f} → {actual_stats['PTS']} (error: {abs(row['PTS'] - actual_stats['PTS']):.1f})")
            else:
                print(f"  ✗ No game data found (DNP or not played yet)")
            
            time.sleep(0.6)  # NBA API rate limit
        
        if not results:
            print("\n⚠️  No actual results found!")
            print("   Possible reasons:")
            print("   - Games haven't been played yet")
            print("   - Games are still in progress")
            print("   - Players didn't play (DNP)")
            return None
        
        results_df = pd.DataFrame(results)
        print(f"\n✓ Successfully fetched results for {len(results_df)} players")
        
        return results_df
    
    def calculate_validation_metrics(self, results_df):
        """
        Calculate validation metrics
        
        Args:
            results_df (pd.DataFrame): Results with predictions and actuals
            
        Returns:
            dict: Validation metrics
        """
        print("\n" + "="*60)
        print("CALCULATING VALIDATION METRICS")
        print("="*60)
        
        metrics = {}
        
        for stat in ['PTS', 'REB', 'AST']:
            pred_col = f'{stat}_PRED'
            actual_col = f'{stat}_ACTUAL'
            error_col = f'{stat}_ERROR'
            
            predictions = results_df[pred_col].values
            actuals = results_df[actual_col].values
            errors = results_df[error_col].values
            
            mae = np.mean(errors)
            rmse = np.sqrt(np.mean((predictions - actuals)**2))
            
            within_1 = np.mean(errors <= 1)
            within_2 = np.mean(errors <= 2)
            within_3 = np.mean(errors <= 3)
            within_5 = np.mean(errors <= 5)
            
            bias = np.mean(predictions - actuals)
            
            metrics[stat] = {
                'mae': mae,
                'rmse': rmse,
                'within_1': within_1,
                'within_2': within_2,
                'within_3': within_3,
                'within_5': within_5,
                'bias': bias,
                'mean_pred': np.mean(predictions),
                'mean_actual': np.mean(actuals)
            }
        
        return metrics
    
    def print_validation_report(self, results_df, metrics):
        """
        Print validation report
        
        Args:
            results_df (pd.DataFrame): Results
            metrics (dict): Validation metrics
        """
        print("\n" + "="*70)
        print(" "*20 + "VALIDATION REPORT")
        print("="*70)
        
        # Overall metrics
        print("\n📊 OVERALL PERFORMANCE\n")
        print(f"{'Stat':<6} {'MAE':<8} {'RMSE':<8} {'Within 3':<12} {'Bias':<8}")
        print("-" * 70)
        
        for stat, m in metrics.items():
            print(f"{stat:<6} {m['mae']:<8.3f} {m['rmse']:<8.3f} "
                  f"{m['within_3']*100:<12.1f}% {m['bias']:<8.2f}")
        
        # Best predictions
        print("\n🎯 BEST PREDICTIONS (Lowest Total Error)\n")
        results_df['TOTAL_ERROR'] = (results_df['PTS_ERROR'] + 
                                     results_df['REB_ERROR'] + 
                                     results_df['AST_ERROR'])
        
        best = results_df.nsmallest(10, 'TOTAL_ERROR')
        print(f"{'Player':<25} {'PTS':<15} {'REB':<15} {'AST':<15}")
        print("-" * 70)
        
        for idx, row in best.iterrows():
            pts_str = f"{row['PTS_PRED']:.1f}→{row['PTS_ACTUAL']}"
            reb_str = f"{row['REB_PRED']:.1f}→{row['REB_ACTUAL']}"
            ast_str = f"{row['AST_PRED']:.1f}→{row['AST_ACTUAL']}"
            print(f"{row['Player']:<25} {pts_str:<15} {reb_str:<15} {ast_str:<15}")
        
        # Worst predictions
        print("\n⚠️  WORST PREDICTIONS (Highest Total Error)\n")
        worst = results_df.nlargest(10, 'TOTAL_ERROR')
        print(f"{'Player':<25} {'PTS':<15} {'REB':<15} {'AST':<15}")
        print("-" * 70)
        
        for idx, row in worst.iterrows():
            pts_str = f"{row['PTS_PRED']:.1f}→{row['PTS_ACTUAL']}"
            reb_str = f"{row['REB_PRED']:.1f}→{row['REB_ACTUAL']}"
            ast_str = f"{row['AST_PRED']:.1f}→{row['AST_ACTUAL']}"
            print(f"{row['Player']:<25} {pts_str:<15} {reb_str:<15} {ast_str:<15}")
        
        # Summary stats
        print("\n📈 ACCURACY BREAKDOWN\n")
        for stat in ['PTS', 'REB', 'AST']:
            m = metrics[stat]
            print(f"{stat}:")
            print(f"  MAE:          {m['mae']:.3f}")
            print(f"  Within 1:     {m['within_1']*100:.1f}%")
            print(f"  Within 2:     {m['within_2']*100:.1f}%")
            print(f"  Within 3:     {m['within_3']*100:.1f}%")
            print(f"  Within 5:     {m['within_5']*100:.1f}%")
            print(f"  Bias:         {m['bias']:+.2f}")
            print()
    
    def save_results(self, results_df, metrics, date):
        """
        Save validation results
        
        Args:
            results_df (pd.DataFrame): Results
            metrics (dict): Metrics
            date (str): Date
        """
        print("\n" + "="*60)
        print("SAVING VALIDATION RESULTS")
        print("="*60)
        
        # Save detailed results
        results_file = f"{self.results_dir}/validation_{date}.csv"
        results_df.to_csv(results_file, index=False)
        print(f"✓ Results saved: {results_file}")
        
        # Save metrics
        metrics_df = pd.DataFrame(metrics).T
        metrics_file = f"{self.results_dir}/validation_metrics_{date}.csv"
        metrics_df.to_csv(metrics_file)
        print(f"✓ Metrics saved: {metrics_file}")
    
    def validate_today(self, game_date=None):
        """
        Run validation for today's predictions
        
        Args:
            game_date (str): Date in YYYY-MM-DD format (default: today)
        """
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
        
        print("\n" + "="*70)
        print(" "*15 + "LIVE PREDICTION VALIDATION")
        print("="*70)
        print(f"\nValidating predictions for: {game_date}")
        
        # Load predictions
        prediction_date = game_date.replace('-', '')
        predictions_df = self.load_predictions(prediction_date)
        
        # Fetch actual results
        results_df = self.fetch_actual_results(predictions_df, game_date)
        
        if results_df is None or len(results_df) == 0:
            print("\n❌ No results to validate!")
            print("   Make sure games have been completed.")
            return
        
        # Calculate metrics
        metrics = self.calculate_validation_metrics(results_df)
        
        self.print_validation_report(results_df, metrics)
        
        # Save results
        self.save_results(results_df, metrics, prediction_date)
        
        print("\n" + "="*70)
        print("✅ VALIDATION COMPLETE!")
        print("="*70)
        
        return results_df, metrics


if __name__ == "__main__":
    import sys
    
    validator = PredictionValidator()
    
    if len(sys.argv) > 1:
        game_date = sys.argv[1]  # Format: YYYY-MM-DD
        print(f"Validating predictions for: {game_date}")
    else:
        game_date = None  # Use today
        print("Validating predictions for today")
    
    validator.validate_today(game_date)
