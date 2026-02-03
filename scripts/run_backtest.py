"""
Backtest Script
Run backtests to validate model performance on historical data
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.backtesting.backtest_engine import BacktestEngine


def main():
    """Main execution function"""
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║        NBA PLAYER PERFORMANCE PREDICTION SYSTEM          ║
    ║                      BACKTESTING                         ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Initialize backtest engine
        engine = BacktestEngine()
        
        # Run all backtests
        all_metrics = engine.run_all_backtests()
        
        if not all_metrics:
            print("\n❌ Backtesting failed - no models found")
            print("\nPlease train models first:")
            print("  python scripts/train_models.py")
            return
        
        # Analyze player performance
        print("\n" + "="*70)
        print(" "*20 + "PLAYER-LEVEL ANALYSIS")
        print("="*70)
        
        for stat in all_metrics.keys():
            engine.analyze_player_performance(stat, top_n=5)
        
        print("\n" + "="*70)
        print("🎉 BACKTESTING COMPLETE!")
        print("="*70)
        
        print("\n📊 Results Summary:")
        for stat, metrics in all_metrics.items():
            metrics_dict = metrics.calculate_metrics()
            print(f"\n{stat}:")
            print(f"  MAE:              {metrics_dict['mae']:.3f}")
            print(f"  Accuracy (±3):    {metrics_dict['accuracy_within_3']*100:.1f}%")
            print(f"  Total Games:      {metrics_dict['total_predictions']:,}")
        
        print("\n📁 Files Generated:")
        print("  ✓ results/backtest_pts_results.csv")
        print("  ✓ results/backtest_reb_results.csv")
        print("  ✓ results/backtest_ast_results.csv")
        print("  ✓ results/backtest_metrics_summary.json")
        
        print("\n🎯 Next Steps:")
        print("  1. Review prediction accuracy in results files")
        print("  2. Analyze which players are most/least predictable")
        print("  3. Generate live predictions: python scripts/predict_today.py")
        
        print("\n💡 Tips:")
        print("  - Check backtest_metrics_summary.json for detailed stats")
        print("  - Look at individual game predictions in CSV files")
        print("  - Compare predicted vs actual to identify patterns")
        
        print("\n" + "="*70)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print("  1. Models are trained: python scripts/train_models.py")
        print("  2. Test data exists: data/processed/test_features.csv")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting tips:")
        print("  1. Verify models were trained successfully")
        print("  2. Check test_features.csv has all required columns")
        print("  3. Ensure config.yaml paths are correct")


if __name__ == "__main__":
    main()