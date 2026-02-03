"""
Daily Predictions Script
Generate predictions for today's NBA games
"""

import sys
import os
import argparse
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.prediction.predictor import NBAPredictor


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='Generate NBA player predictions')
    parser.add_argument('--min-minutes', type=float, default=15,
                       help='Minimum minutes to include player (default: 15)')
    parser.add_argument('--top-n', type=int, default=50,
                       help='Number of top players to show (default: 50)')
    parser.add_argument('--player', type=str, default=None,
                       help='Predict for specific player')
    parser.add_argument('--save', action='store_true',
                       help='Save predictions to CSV')
    
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║        NBA PLAYER PERFORMANCE PREDICTION SYSTEM          ║
    ║                  LIVE PREDICTIONS                        ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    print(f"\n📅 Date: {datetime.now().strftime('%B %d, %Y')}")
    print(f"⏰ Time: {datetime.now().strftime('%I:%M %p')}")
    
    try:
        predictor = NBAPredictor()
        
        # Predict for specific player
        if args.player:
            print(f"\n🔍 Looking up player: {args.player}")
            
            prediction = predictor.get_player_prediction(args.player)
            
            if prediction is None:
                print(f"\n❌ Player not found: {args.player}")
                print("\nTips:")
                print("  - Check spelling")
                print("  - Use full name (e.g., 'LeBron James')")
                print("  - Make sure player is active")
                return
            
            print("\n" + "="*60)
            print(f"PREDICTION: {args.player}")
            print("="*60)
            
            print(f"\nTeam: {prediction.get('TEAM_ABBREVIATION', 'N/A')}")
            print(f"\nPredicted Stats:")
            print(f"  Points:    {prediction.get('PTS_PREDICTED', 0):.1f}")
            print(f"  Rebounds:  {prediction.get('REB_PREDICTED', 0):.1f}")
            print(f"  Assists:   {prediction.get('AST_PREDICTED', 0):.1f}")
            
            if 'PTS_L10_AVG' in prediction:
                print(f"\nLast 10 Game Averages:")
                print(f"  Points:    {prediction.get('PTS_L10_AVG', 0):.1f}")
                print(f"  Rebounds:  {prediction.get('REB_L10_AVG', 0):.1f}")
                print(f"  Assists:   {prediction.get('AST_L10_AVG', 0):.1f}")
            
            print("="*60)
            
        else:
            # Predict for all players
            predictions = predictor.predict_today(min_minutes=args.min_minutes)
            
            if predictions is None:
                print("\n❌ No predictions generated")
                return
            
            # Display predictions
            print(predictor.format_predictions(predictions, top_n=args.top_n))
            
            # Save if requested
            if args.save:
                filepath = predictor.save_predictions(predictions)
                print(f"\n💾 Predictions saved!")
            
            # Show summary stats
            print("\n" + "="*60)
            print("PREDICTION SUMMARY")
            print("="*60)
            
            for stat in ['PTS', 'REB', 'AST']:
                pred_col = f'{stat}_PREDICTED'
                if pred_col in predictions.columns:
                    mean_pred = predictions[pred_col].mean()
                    max_pred = predictions[pred_col].max()
                    max_player = predictions.loc[predictions[pred_col].idxmax(), 'PLAYER_NAME']
                    
                    print(f"\n{stat}:")
                    print(f"  Average prediction: {mean_pred:.1f}")
                    print(f"  Highest: {max_pred:.1f} ({max_player})")
            
            print("\n" + "="*60)
            print("🎯 NEXT STEPS")
            print("="*60)
            print("\n1. Compare predictions to Vegas lines")
            print("2. Identify value opportunities")
            print("3. Track accuracy over time")
            print("4. Refine and improve models")
            
            print("\n💡 TIP: Use --save flag to save predictions to CSV")
            print("   Example: python scripts/predict_today.py --save")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print("  1. Models are trained: python scripts/train_models.py")
        print("  2. Feature data exists: data/processed/test_features.csv")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting:")
        print("  1. Verify all previous steps completed successfully")
        print("  2. Check config.yaml paths are correct")
        print("  3. Ensure models and data files exist")


if __name__ == "__main__":
    main()