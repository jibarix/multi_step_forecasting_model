"""
Linear regression models for economic forecasting.
"""
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import statsmodels.api as sm

logger = logging.getLogger(__name__)

class LinearModel:
    """
    Wrapper for linear regression models.
    """
    
    def __init__(self, model_type: str = 'linear_regression', **kwargs):
        """
        Initialize a linear model.
        
        Args:
            model_type: Type of linear model to use 
                       ('linear_regression', 'lasso', 'ridge', 'elastic_net')
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.model_params = kwargs
        self.model = None
        self.stats_model = None  # For statsmodels summary
        self.feature_names = None
        
    def _initialize_model(self):
        """
        Initialize the underlying sklearn model.
        """
        if self.model_type == 'linear_regression':
            self.model = LinearRegression(**self.model_params)
        elif self.model_type == 'lasso':
            self.model = Lasso(**self.model_params)
        elif self.model_type == 'ridge':
            self.model = Ridge(**self.model_params)
        elif self.model_type == 'elastic_net':
            self.model = ElasticNet(**self.model_params)
        else:
            raise ValueError(f"Unknown linear model type: {self.model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, feature_names: Optional[List[str]] = None) -> 'LinearModel':
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
        
        # Fit sklearn model
        self.model.fit(X, y)
        
        # Also fit statsmodels OLS for detailed statistics
        try:
            X_with_const = sm.add_constant(X)
            self.stats_model = sm.OLS(y, X_with_const).fit()
        except Exception as e:
            logger.warning(f"Could not fit statsmodels OLS: {e}")
            self.stats_model = None
        
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
        Get feature importance or coefficients.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if self.model_type == 'linear_regression':
            coefficients = self.model.coef_
        elif self.model_type in ['lasso', 'ridge', 'elastic_net']:
            coefficients = self.model.coef_
        else:
            raise ValueError(f"Feature importance not implemented for {self.model_type}")
        
        # Create dictionary mapping feature names to coefficients
        if self.feature_names is not None:
            return {name: coef for name, coef in zip(self.feature_names, coefficients)}
        else:
            return {f"feature_{i}": coef for i, coef in enumerate(coefficients)}
    
    def get_summary(self) -> str:
        """
        Get summary statistics for the model (if available).
        
        Returns:
            Model summary string
        """
        if self.stats_model is not None:
            return self.stats_model.summary().as_text()
        else:
            return "Detailed model statistics not available. Use statsmodels for summary statistics."
    
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
    def load(cls, filepath: str) -> 'LinearModel':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model
        """
        # Implement model loading logic
        pass