"""
Constants and mappings for NBA prediction system
"""

# Target statistics
TARGET_STATS = ['points', 'rebounds', 'assists']

# NBA Team abbreviations (30 teams)
NBA_TEAMS = [
    'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 
    'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA',
    'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX',
    'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
]

# Column mappings for consistency
COLUMN_MAPPINGS = {
    'PTS': 'points',
    'REB': 'rebounds', 
    'AST': 'assists',
    'STL': 'steals',
    'BLK': 'blocks',
    'TOV': 'turnovers',
    'FG_PCT': 'fg_pct',
    'FG3_PCT': 'fg3_pct',
    'FT_PCT': 'ft_pct',
    'MIN': 'minutes',
    'PLUS_MINUS': 'plus_minus'
}

# Stat thresholds for filtering
STAT_THRESHOLDS = {
    'points': {'min': 0, 'max': 80},
    'rebounds': {'min': 0, 'max': 30},
    'assists': {'min': 0, 'max': 25},
    'minutes': {'min': 0, 'max': 48}
}

# Feature groups
FEATURE_GROUPS = {
    'player_stats': ['points', 'rebounds', 'assists', 'minutes', 'fg_pct'],
    'situational': ['is_home', 'rest_days', 'is_back_to_back'],
    'opponent': ['opp_def_rating', 'opp_pace', 'opp_pts_allowed']
}

# Model configurations
MODEL_CONFIGS = {
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'lightgbm': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
}

# Seasons info
SEASON_INFO = {
    '2022-23': {
        'start_date': '2022-10-18',
        'end_date': '2023-04-09',
        'games_per_team': 82
    },
    '2023-24': {
        'start_date': '2023-10-24',
        'end_date': '2024-04-14',
        'games_per_team': 82
    },
    '2024-25': {
        'start_date': '2024-10-22',
        'end_date': '2025-04-13',
        'games_per_team': 82
    }
}

# Rolling window configurations
ROLLING_WINDOWS = {
    'short_term': 5,   # Last 5 games
    'medium_term': 10,  # Last 10 games
    'long_term': 15     # Last 15 games
}

# Validation split ratios
VALIDATION_SPLITS = {
    'train': 0.7,
    'validation': 0.15,
    'test': 0.15
}