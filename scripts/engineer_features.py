import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.features.feature_pipeline import FeaturePipeline


def main():
    """Main execution function"""
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║        NBA PLAYER PERFORMANCE PREDICTION SYSTEM          ║
    ║                 FEATURE ENGINEERING                      ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        
        pipeline = FeaturePipeline()
        train_df, test_df, feature_names = pipeline.run_pipeline()
        
        print("\n" + "="*60)
        print("FEATURE BREAKDOWN")
        print("="*60)
        
        feature_types = {
            'Rolling Averages': [f for f in feature_names if '_L' in f and '_VS_' not in f],
            'Trends': [f for f in feature_names if '_TREND' in f or '_RECENT_FORM' in f],
            'Consistency': [f for f in feature_names if '_STD_' in f or '_CV_' in f],
            'Home/Away': [f for f in feature_names if '_HOME_' in f or '_AWAY_' in f],
            'Rest/Fatigue': [f for f in feature_names if 'REST_' in f or 'B2B' in f or 'GAMES_IN_' in f],
            'Season Position': [f for f in feature_names if 'SEASON' in f],
            'Head-to-Head': [f for f in feature_names if '_VS_OPP_' in f],
            'Opponent Defense': [f for f in feature_names if 'OPP_DEF_' in f or 'MATCHUP_DIFFICULTY' in f],
        }
        
        for category, features in feature_types.items():
            if features:
                print(f"\n{category}: {len(features)}")
                for feature in features[:5]:  # Show first 5
                    print(f"  - {feature}")
                if len(features) > 5:
                    print(f"  ... and {len(features) - 5} more")
        
        print("\n" + "="*60)
        print("🎉 FEATURE ENGINEERING SUCCESSFUL!")
        print("="*60)
        print("\nGenerated files:")
        print("  ✓ data/processed/train_features.csv")
        print("  ✓ data/processed/test_features.csv")
        print("  ✓ data/processed/feature_names.txt")
        
        print("\nNext steps:")
        print("  1. Review features in train_features.csv")
        print("  2. Train models: python scripts/train_models.py")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run data collection first:")
        print("  python scripts/collect_data.py")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting tips:")
        print("  1. Check that data collection completed successfully")
        print("  2. Verify config.yaml settings")
        print("  3. Check for any data quality issues")


if __name__ == "__main__":
    main()
