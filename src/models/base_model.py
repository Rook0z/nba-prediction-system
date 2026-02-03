"""
Base Model Class
Abstract base for all prediction models
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime


class BaseModel(ABC):
    """Abstract base class for NBA prediction models"""
    
    def __init__(self, target_stat, model_params=None):
        """
        Initialize base model
        
        Args:
            target_stat (str): Target statistic (PTS, REB, or AST)
            model_params (dict): Model hyperparameters
        """
        self.target_stat = target_stat
        self.model_params = model_params or {}
        self.model = None
        self.feature_names = None
        self.training_date = None
        self.metrics = {}
        
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass
    
    def evaluate(self, X, y):
        """
        Evaluate model performance
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): True values
            
        Returns:
            dict: Evaluation metrics
        """
        predictions = self.predict(X)
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        
        # Calculate accuracy within thresholds
        errors = np.abs(predictions - y)
        acc_1 = np.mean(errors <= 1)
        acc_2 = np.mean(errors <= 2)
        acc_3 = np.mean(errors <= 3)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'accuracy_within_1': acc_1,
            'accuracy_within_2': acc_2,
            'accuracy_within_3': acc_3,
            'mean_prediction': np.mean(predictions),
            'mean_actual': np.mean(y)
        }
        
        return metrics
    
    def save(self, filepath):
        """
        Save model to file
        
        Args:
            filepath (str): Path to save model
        """
        model_data = {
            'model': self.model,
            'target_stat': self.target_stat,
            'feature_names': self.feature_names,
            'model_params': self.model_params,
            'training_date': self.training_date,
            'metrics': self.metrics
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved: {filepath}")
    
    def load(self, filepath):
        """
        Load model from file
        
        Args:
            filepath (str): Path to model file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.target_stat = model_data['target_stat']
        self.feature_names = model_data['feature_names']
        self.model_params = model_data['model_params']
        self.training_date = model_data['training_date']
        self.metrics = model_data.get('metrics', {})
        
        print(f"✓ Model loaded: {filepath}")
    
    def get_feature_importance(self, top_n=20):
        """
        Get feature importance
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance
        """
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def print_metrics(self, metrics=None):
        """
        Print model metrics in a formatted way
        
        Args:
            metrics (dict): Metrics to print (uses self.metrics if None)
        """
        if metrics is None:
            metrics = self.metrics
        
        print(f"\n{'='*60}")
        print(f"{self.target_stat} Model Performance")
        print(f"{'='*60}")
        print(f"MAE:              {metrics['mae']:.3f}")
        print(f"RMSE:             {metrics['rmse']:.3f}")
        print(f"R² Score:         {metrics['r2']:.3f}")
        print(f"\nAccuracy:")
        print(f"  Within 1:       {metrics['accuracy_within_1']*100:.1f}%")
        print(f"  Within 2:       {metrics['accuracy_within_2']*100:.1f}%")
        print(f"  Within 3:       {metrics['accuracy_within_3']*100:.1f}%")
        print(f"\nPredictions:")
        print(f"  Mean Predicted: {metrics['mean_prediction']:.2f}")
        print(f"  Mean Actual:    {metrics['mean_actual']:.2f}")
        print(f"{'='*60}\n")