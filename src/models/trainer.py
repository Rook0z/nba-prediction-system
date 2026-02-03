"""
Model Trainer
Handles training multiple models and managing the training pipeline
"""

import pandas as pd
import numpy as np
import yaml
import os
from sklearn.model_selection import train_test_split
from src.models.points_model import PointsModel
from src.models.rebounds_model import ReboundsModel
from src.models.assists_model import AssistsModel


class ModelTrainer:
    """Trainer for NBA prediction models"""
    
    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize trainer
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.processed_dir = self.config['paths']['processed_dir']
        self.models_dir = self.config['paths']['models_dir']
        self.target_stats = self.config['target_stats']
        
        self.models = {}
        self.feature_names = None
        
    def load_training_data(self):
        """
        Load training and testing data
        
        Returns:
            tuple: (train_df, test_df)
        """
        print("\n" + "="*60)
        print("LOADING TRAINING DATA")
        print("="*60)
        
        train_path = f"{self.processed_dir}/train_features.csv"
        test_path = f"{self.processed_dir}/test_features.csv"
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found: {train_path}")
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"✓ Training data loaded: {train_df.shape}")
        print(f"✓ Testing data loaded: {test_df.shape}")
        
        return train_df, test_df
    
    def load_feature_names(self):
        """
        Load feature names
        
        Returns:
            list: Feature column names
        """
        feature_path = f"{self.processed_dir}/feature_names.txt"
        
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                features = [line.strip() for line in f.readlines()]
            print(f"✓ Loaded {len(features)} features from file")
            return features
        else:
            print("⚠️  feature_names.txt not found, inferring from data")
            return None
    
    def prepare_data_for_stat(self, train_df, test_df, target_stat):
        """
        Prepare features and target for specific stat
        
        Args:
            train_df (pd.DataFrame): Training data
            test_df (pd.DataFrame): Testing data
            target_stat (str): Target statistic (PTS, REB, AST)
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        # Columns to exclude
        exclude_cols = [
            'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'GAME_ID',
            'GAME_DATE', 'MATCHUP', 'OPPONENT', 'SEASON',
            'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
            'FG_PCT', 'FG3_PCT', 'FT_PCT', 'PLUS_MINUS'
        ]
        
        # Get feature columns
        if self.feature_names is None:
            feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        else:
            feature_cols = self.feature_names
        
        # Ensure target exists
        if target_stat not in train_df.columns:
            raise ValueError(f"Target {target_stat} not found in data")
        
        # Split features and target
        X_train = train_df[feature_cols]
        y_train = train_df[target_stat]
        
        X_test = test_df[feature_cols]
        y_test = test_df[target_stat]
        
        print(f"\n{target_stat} Dataset:")
        print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"  Testing: {X_test.shape[0]} samples")
        print(f"  Target mean (train): {y_train.mean():.2f}")
        print(f"  Target mean (test): {y_test.mean():.2f}")
        
        return X_train, y_train, X_test, y_test
    
    def create_validation_split(self, X_train, y_train, val_size=0.2):
        """
        Create validation split from training data
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            val_size (float): Validation split size
            
        Returns:
            tuple: (X_train_sub, X_val, y_train_sub, y_val)
        """
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size,
            random_state=42
        )
        
        return X_train_sub, X_val, y_train_sub, y_val
    
    def get_model_for_stat(self, target_stat):
        """
        Get appropriate model for target statistic
        
        Args:
            target_stat (str): Target statistic
            
        Returns:
            BaseModel: Initialized model
        """
        if target_stat == 'PTS':
            return PointsModel()
        elif target_stat == 'REB':
            return ReboundsModel()
        elif target_stat == 'AST':
            return AssistsModel()
        else:
            raise ValueError(f"Unknown target stat: {target_stat}")
    
    def train_model(self, target_stat, X_train, y_train, X_val=None, y_val=None):
        """
        Train model for specific statistic
        
        Args:
            target_stat (str): Target statistic
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features (optional)
            y_val (pd.Series): Validation target (optional)
            
        Returns:
            BaseModel: Trained model
        """
        print("\n" + "="*60)
        print(f"TRAINING MODEL: {target_stat}")
        print("="*60)
        
        # Initialize model
        model = self.get_model_for_stat(target_stat)
        
        # Train model
        model.train(X_train, y_train, X_val, y_val)
        
        # Print feature importance
        importance_df = model.get_feature_importance(top_n=15)
        if importance_df is not None:
            print(f"\n{'='*60}")
            print(f"Top 15 Most Important Features - {target_stat}")
            print(f"{'='*60}")
            
            for idx, row in importance_df.iterrows():
                bar_length = int(row['importance'] / importance_df['importance'].max() * 40)
                bar = '█' * bar_length
                print(f"{row['feature']:30s} {bar} {row['importance']:.4f}")
            
            print(f"{'='*60}\n")
        
        return model
    
    def train_all_models(self, use_validation=True):
        """
        Train models for all target statistics
        
        Args:
            use_validation (bool): Whether to use validation split
            
        Returns:
            dict: Dictionary of trained models
        """
        print("\n" + "="*70)
        print(" "*20 + "MODEL TRAINING PIPELINE")
        print("="*70)
        
        # Load data
        train_df, test_df = self.load_training_data()
        
        # Load feature names
        self.feature_names = self.load_feature_names()
        
        # Train model for each stat
        for stat in self.target_stats:
            print(f"\n\n{'#'*70}")
            print(f"{'#'*20} {stat} MODEL {'#'*20}")
            print(f"{'#'*70}")
            
            # Prepare data
            X_train, y_train, X_test, y_test = self.prepare_data_for_stat(
                train_df, test_df, stat
            )
            
            # Create validation split
            if use_validation:
                X_train_sub, X_val, y_train_sub, y_val = self.create_validation_split(
                    X_train, y_train, val_size=0.2
                )
                print(f"\nValidation split: {len(X_val)} samples ({len(X_val)/len(X_train)*100:.1f}%)")
            else:
                X_train_sub, X_val, y_train_sub, y_val = X_train, None, y_train, None
            
            # Train model
            model = self.train_model(stat, X_train_sub, y_train_sub, X_val, y_val)
            
            # Evaluate on test set
            print("\n" + "="*60)
            print("EVALUATING ON TEST SET (2024-25 Season)")
            print("="*60)
            test_metrics = model.evaluate(X_test, y_test)
            model.print_metrics(test_metrics)
            
            # Store model
            self.models[stat] = model
            
            # Save model
            model_path = f"{self.models_dir}/{stat.lower()}_model.pkl"
            model.save(model_path)
        
        # Print summary
        self.print_training_summary()
        
        return self.models
    
    def print_training_summary(self):
        """Print summary of all trained models"""
        print("\n" + "="*70)
        print(" "*25 + "TRAINING SUMMARY")
        print("="*70)
        
        print("\n📊 Model Performance Comparison:\n")
        print(f"{'Stat':<10} {'Train MAE':<12} {'Val MAE':<12} {'R² Score':<10}")
        print("-" * 70)
        
        for stat, model in self.models.items():
            train_mae = model.metrics['train']['mae']
            
            if 'validation' in model.metrics:
                val_mae = model.metrics['validation']['mae']
                val_r2 = model.metrics['validation']['r2']
            else:
                val_mae = train_mae
                val_r2 = model.metrics['train']['r2']
            
            print(f"{stat:<10} {train_mae:<12.3f} {val_mae:<12.3f} {val_r2:<10.3f}")
        
        print("\n✅ All models trained and saved!")
        print(f"📁 Models saved to: {self.models_dir}/")
        
        print("\n🎯 Next steps:")
        print("  1. Review model performance metrics")
        print("  2. Run backtesting: python scripts/run_backtest.py")
        print("  3. Generate predictions: python scripts/predict_today.py")


if __name__ == "__main__":
    trainer = ModelTrainer()
    models = trainer.train_all_models(use_validation=True)