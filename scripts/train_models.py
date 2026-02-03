"""
Model Training Script
Run this to train XGBoost models for all target statistics
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.trainer import ModelTrainer


def main():
    """Main execution function"""
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║        NBA PLAYER PERFORMANCE PREDICTION SYSTEM          ║
    ║                    MODEL TRAINING                        ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Train all models
        models = trainer.train_all_models(use_validation=True)
        
        print("\n" + "="*70)
        print("🎉 MODEL TRAINING COMPLETE!")
        print("="*70)
        
        print("\n✅ Successfully trained models:")
        for stat in models.keys():
            print(f"  ✓ {stat} Model")
        
        print(f"\n📁 Models saved in: ./models/")
        print("  - points_model.pkl")
        print("  - rebounds_model.pkl")
        print("  - assists_model.pkl")
        
        print("\n📊 Performance Summary:")
        for stat, model in models.items():
            if 'validation' in model.metrics:
                val_mae = model.metrics['validation']['mae']
                val_r2 = model.metrics['validation']['r2']
                print(f"  {stat}:")
                print(f"    MAE: {val_mae:.3f}")
                print(f"    R²:  {val_r2:.3f}")
        
        print("\n🎯 Next Steps:")
        print("  1. Review feature importance in the output above")
        print("  2. Run backtesting to validate performance")
        print("  3. Generate predictions for upcoming games")
        
        print("\n" + "="*70)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure you've run feature engineering first:")
        print("  python scripts/engineer_features.py")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting tips:")
        print("  1. Verify feature engineering completed successfully")
        print("  2. Check train_features.csv and test_features.csv exist")
        print("  3. Ensure XGBoost is installed: pip install xgboost")


if __name__ == "__main__":
    main()