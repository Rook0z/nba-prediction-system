import xgboost as xgb
from src.models.base_model import BaseModel
from datetime import datetime


class PointsModel(BaseModel):
    """XGBoost model for predicting player points (PTS)"""
    
    def __init__(self, model_params=None):
        """
        Initialize Points model
        
        Args:
            model_params (dict): XGBoost hyperparameters
        """
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Update with user params
        if model_params:
            default_params.update(model_params)
        
        super().__init__(target_stat='PTS', model_params=default_params)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train Points prediction model
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training points values
            X_val (pd.DataFrame): Validation features (optional)
            y_val (pd.Series): Validation points values (optional)
        """
        print(f"\n{'='*60}")
        print(f"Training Points Model")
        print(f"{'='*60}")
        print(f"Training samples: {len(X_train)}")
        if X_val is not None:
            print(f"Validation samples: {len(X_val)}")
        print(f"Features: {X_train.shape[1]}")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Initialize model
        self.model = xgb.XGBRegressor(**self.model_params)
        
        # Prepare evaluation set
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        
        # Train model
        print("\nTraining in progress...")
        
        if eval_set:
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            if hasattr(self.model, 'best_iteration'):
                print(f"Best iteration: {self.model.best_iteration}")
        else:
            self.model.fit(X_train, y_train, verbose=False)
        
        self.training_date = datetime.now()
        print("✓ Training complete!")
        
        # Evaluate
        print("\nEvaluating on training set...")
        train_metrics = self.evaluate(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            print("\nEvaluating on validation set...")
            val_metrics = self.evaluate(X_val, y_val)
            self.metrics = {
                'train': train_metrics,
                'validation': val_metrics
            }
            
            print("\n📊 Training Metrics:")
            self.print_metrics(train_metrics)
            
            print("📊 Validation Metrics:")
            self.print_metrics(val_metrics)
        else:
            self.metrics = {'train': train_metrics}
            self.print_metrics(train_metrics)
    
    def predict(self, X):
        """
        Predict player points
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.array: Predicted points
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
