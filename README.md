# NBA Player Performance Prediction System

A machine learning system to predict NBA player performance (Points, Rebounds, Assists) using 3 seasons of historical data.

## 📊 Project Overview

**Target Predictions:**
- Player Points per game
- Player Rebounds per game
- Player Assists per game

**Data Coverage:**
- Seasons: 2022-23, 2023-24, 2024-25
- Training: 2022-23 + 2023-24
- Testing: 2024-25 (current season)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or create project directory
mkdir nba-prediction-system
cd nba-prediction-system

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `config/config.yaml` to customize settings:
- Target statistics
- Seasons to analyze
- Feature engineering parameters
- Model hyperparameters

### 3. Collect Data

```bash
# This will take 5-10 minutes
python scripts/collect_data.py
```

This script will:
- ✅ Fetch player game logs from NBA API
- ✅ Collect team statistics
- ✅ Gather schedule information
- ✅ Clean and process all data
- ✅ Save processed data to `data/processed/`

### 4. Expected Output

After running data collection, you should have:

```
data/
├── raw/
│   ├── game_logs/
│   │   ├── player_logs_2022-23.csv
│   │   ├── player_logs_2023-24.csv
│   │   └── player_logs_2024-25.csv
│   ├── team_stats/
│   │   └── team_stats_*.csv
│   └── schedules/
│       └── schedule_*.csv
└── processed/
    └── processed_game_logs.csv  ← Main dataset for feature engineering
```

## 📁 Project Structure

```
nba-prediction-system/
├── config/               # Configuration files
├── data/                 # Raw and processed data
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── data/            # Data collection and processing
│   ├── features/        # Feature engineering (next step)
│   ├── models/          # Model training (next step)
│   └── utils/           # Helper functions
├── scripts/             # Executable scripts
├── models/              # Saved trained models
└── results/             # Predictions and backtest results
```

## 🔧 Configuration Details

### Minimum Requirements
- **Min Games:** 15 games played
- **Min Minutes:** 20 minutes per game average
- Players below these thresholds are filtered out

### Feature Settings
- **Rolling Windows:** 5, 10, 15 game averages
- **Home/Away:** Separate performance tracking
- **Rest Days:** Days between games
- **Opponent Defense:** Defensive ratings
- **Back-to-Back:** B2B game indicator

## 📊 Data Quality Checks

The processor automatically:
- Removes duplicate games
- Filters by minimum games/minutes
- Validates date ranges
- Calculates rest days
- Merges opponent statistics
- Reports missing values

## ⚠️ Troubleshooting

### NBA API Rate Limiting
If you see errors about rate limiting:
- Wait 5-10 minutes between runs
- The script includes automatic delays
- Consider collecting one season at a time

### Missing Data
If some files are missing:
- Check your internet connection
- Verify seasons in config.yaml are valid
- NBA API occasionally has downtime

### Import Errors
Make sure you're running scripts from the project root:
```bash
# Wrong
cd scripts
python collect_data.py

# Correct
python scripts/collect_data.py
```

## 📝 Next Steps

After successful data collection:

1. **Explore the data** using `notebooks/01_data_exploration.ipynb`
2. **Engineer features** - Coming next!
3. **Train models** - Coming soon!
4. **Run backtests** - Coming soon!

## 🎯 Current Status

- ✅ Project structure created
- ✅ Data collection implemented
- ✅ Data processing implemented
- ⏳ Feature engineering (next)
- ⏳ Model training
- ⏳ Backtesting framework
- ⏳ Live predictions

## 💡 Tips

1. **Start small:** Successfully collect data before moving to features
2. **Check quality:** Review data quality report after collection
3. **Save checkpoints:** Processed data is saved at each step
4. **Monitor progress:** Script provides detailed progress updates

## 🤝 Contributing

This is a personal project for NBA performance prediction. Feel free to adapt and modify for your own use!

---

**Quick note** the program is still under development so you can experience some bugs. 
**Ready to collect data?** Run: `python scripts/collect_data.py`