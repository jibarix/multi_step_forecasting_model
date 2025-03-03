"""
Ensemble models for economic forecasting.
"""
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb

logger = logging.getLogger(__name__)

class EnsembleModel:
    """
    Wrapper for ensemble models.
    """
    
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        Initialize an ensemble model.
        
        Args:
            model_type: Type of ensemble model
                      ('random_forest', 'gradient_boosting', 'xgboost')
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.model_params = kwargs
        self.model = None
        self.feature_names = None
        
    def _initialize_model(self):
        """
        Initialize the underlying sklearn model.
        """
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(**self.model_params)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(**self.model_params)
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(**self.model_params)
        else:
            raise ValueError(f"Unknown ensemble model type: {self.model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, feature_names: Optional[List[str]] = None) -> 'EnsembleModel':
        """
        Fit the model to training data.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Optional list of feature names
            
        Returns:
            Self for method chaining
        """
        if self.model is None:
            self._initialize_model()
        
        # Store feature names for later use
        self.feature_names = feature_names or X.columns.tolist()
        
        # Fit the model
        self.model.fit(X, y)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be fitted before evaluation")
        
        predictions = self.predict(X)
        
        metrics = {
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions)
        }
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importances = self.model.feature_importances_
        
        # Create dictionary mapping feature names to importance scores
        if self.feature_names is not None:
            return {name: importance for name, importance in zip(self.feature_names, importances)}
        else:
            return {f"feature_{i}": importance for i, importance in enumerate(importances)}
    
    def tune_hyperparameters(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        scoring: str = 'neg_mean_squared_error'
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using grid search with time series cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            param_grid: Grid of hyperparameters to search
            cv: Number of cross-validation folds (uses TimeSeriesSplit)
            scoring: Scoring metric to use
            
        Returns:
            Dictionary with best parameters and best score
        """
        if self.model is None:
            self._initialize_model()
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=cv)
        
        # Set up grid search
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=tscv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.model_params.update(grid_search.best_params_)
        self.model = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def save(self, filepath: str):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model to
        """
        # Implement model saving logic
        pass
    
    @classmethod
    def load(cls, filepath: str) -> 'EnsembleModel':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model
        """
        # Implement model loading logic
        pass