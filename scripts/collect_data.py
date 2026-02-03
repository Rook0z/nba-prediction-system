"""
Main script to collect and process NBA data
Run this script first to gather all necessary data
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.nba_collector import NBADataCollector
from src.data.data_processor import NBADataProcessor


def main():
    """Main execution function"""
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║        NBA PLAYER PERFORMANCE PREDICTION SYSTEM          ║
    ║                  DATA COLLECTION                         ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Collect raw data
    print("\n📥 STEP 1: COLLECTING RAW DATA FROM NBA API")
    print("-" * 60)
    
    collector = NBADataCollector()
    
    try:
        raw_data = collector.collect_all_seasons()
        print("\n✅ Raw data collection successful!")
        
    except Exception as e:
        print(f"\n❌ Error during data collection: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. NBA API might be rate-limiting - try again in a few minutes")
        print("3. Verify the seasons in config.yaml are valid")
        return
    
    # Step 2: Process and clean data
    print("\n\n🔧 STEP 2: PROCESSING AND CLEANING DATA")
    print("-" * 60)
    
    processor = NBADataProcessor()
    
    try:
        processed_data = processor.process_all_data()
        
        if processed_data is not None:
            print("\n✅ Data processing successful!")
            
            # Generate quality report
            processor.get_data_quality_report(processed_data)
            
            print("\n\n" + "="*60)
            print("🎉 DATA COLLECTION COMPLETE!")
            print("="*60)
            print("\nNext steps:")
            print("1. Review the processed data in: ./data/processed/")
            print("2. Run feature engineering: python scripts/engineer_features.py")
            print("3. Train models: python scripts/train_models.py")
            
        else:
            print("\n❌ Data processing failed!")
            
    except Exception as e:
        print(f"\n❌ Error during data processing: {str(e)}")
        return


if __name__ == "__main__":
    main()