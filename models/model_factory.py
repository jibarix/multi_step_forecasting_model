"""
Model factory for creating various forecasting models.
"""
import logging
from typing import Dict, Any, Optional, List, Union
import pandas as pd

from config.model_config import (
    get_model_config,
    LINEAR_MODELS,
    TIME_SERIES_MODELS,
    ENSEMBLE_MODELS,
    NEURAL_NETWORK_MODELS
)
from models.linear_models import LinearModel
from models.time_series_models import TimeSeriesModel
from models.ensemble_models import EnsembleModel

logger = logging.getLogger(__name__)

def get_model(model_type: str, **kwargs) -> Union[LinearModel, TimeSeriesModel, EnsembleModel]:
    """
    Factory function to create and return a model of the specified type.
    
    Args:
        model_type: Type of model to create (e.g., 'linear_regression', 'arima', 'random_forest')
        **kwargs: Additional parameters to override default configuration
    
    Returns:
        Model instance of the appropriate type
    
    Raises:
        ValueError: If model_type is unknown or not supported
    """
    # Get default configuration for this model type
    try:
        config = get_model_config(model_type)
    except ValueError:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type}. Available models: "
                        f"{LINEAR_MODELS + TIME_SERIES_MODELS + ENSEMBLE_MODELS + NEURAL_NETWORK_MODELS}")
    
    # Override default config with provided kwargs
    config.update(kwargs)
    
    # Create appropriate model instance
    if model_type in LINEAR_MODELS:
        logger.info(f"Creating linear model: {model_type}")
        return LinearModel(model_type=model_type, **config)
    elif model_type in TIME_SERIES_MODELS:
        logger.info(f"Creating time series model: {model_type}")
        return TimeSeriesModel(model_type=model_type, **config)
    elif model_type in ENSEMBLE_MODELS:
        logger.info(f"Creating ensemble model: {model_type}")
        return EnsembleModel(model_type=model_type, **config)
    elif model_type in NEURAL_NETWORK_MODELS:
        logger.error(f"Neural network models not yet implemented: {model_type}")
        raise NotImplementedError(f"Neural network models not yet implemented: {model_type}")
    else:
        logger.error(f"Model type not in any known category: {model_type}")
        raise ValueError(f"Model type {model_type} not in any known category")

def create_model_pipeline(
    model_types: List[str],
    ensemble_method: str = 'average',
    **kwargs
) -> Dict[str, Union[LinearModel, TimeSeriesModel, EnsembleModel]]:
    """
    Create multiple models for an ensemble or comparison pipeline.
    
    Args:
        model_types: List of model types to create
        ensemble_method: Method for combining model predictions (for future implementation)
        **kwargs: Additional parameters for specific model types, format: {model_type: {param: value}}
    
    Returns:
        Dictionary mapping model names to model instances
    """
    models = {}
    
    for model_type in model_types:
        # Get model-specific kwargs if available
        model_kwargs = kwargs.get(model_type, {})
        
        try:
            # Create model
            model = get_model(model_type, **model_kwargs)
            
            # Add to models dictionary with unique name
            model_count = sum(1 for name in models if name.startswith(model_type))
            if model_count > 0:
                model_name = f"{model_type}_{model_count + 1}"
            else:
                model_name = model_type
            
            models[model_name] = model
            logger.info(f"Added model {model_name} to pipeline")
            
        except (ValueError, NotImplementedError) as e:
            logger.warning(f"Skipping model {model_type}: {e}")
    
    if not models:
        logger.error("No valid models were created")
        raise ValueError("No valid models were created")
    
    logger.info(f"Created model pipeline with {len(models)} models: {list(models.keys())}")
    return models

def auto_select_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_types: Optional[List[str]] = None,
    metric: str = 'rmse'
) -> Union[LinearModel, TimeSeriesModel, EnsembleModel]:
    """
    Automatically select the best model based on validation performance.
    
    Args:
        X_train: Training feature matrix
        y_train: Training target vector
        X_val: Validation feature matrix
        y_val: Validation target vector
        model_types: List of model types to try (defaults to a selection of models)
        metric: Metric to use for model selection ('rmse', 'mae', 'r2')
    
    Returns:
        Best performing model instance
    """
    # Default set of models to try
    if model_types is None:
        model_types = [
            'linear_regression',  # Linear
            'ridge',              # Regularized linear
            'arima',              # Time series
            'random_forest'       # Ensemble
        ]
    
    # Create all models
    models = create_model_pipeline(model_types)
    
    # Fit and evaluate each model
    best_model = None
    best_score = float('inf')  # Lower is better for RMSE and MAE
    if metric == 'r2':
        best_score = -float('inf')  # Higher is better for R²
    
    for name, model in models.items():
        logger.info(f"Fitting model: {name}")
        
        try:
            # Fit model
            if name.startswith(tuple(TIME_SERIES_MODELS)):
                # For time series models
                df_train = X_train.copy()
                df_train[y_train.name] = y_train
                model.fit(df_train, target_col=y_train.name)
                
                # Create validation dataframe
                df_val = X_val.copy()
                df_val[y_val.name] = y_val
                
                # Evaluate
                metrics = model.evaluate(df_val, target_col=y_val.name)
            else:
                # For traditional ML models
                model.fit(X_train, y_train)
                metrics = model.evaluate(X_val, y_val)
            
            # Check if this model is better
            if metric == 'rmse':
                score = metrics['rmse']
                is_better = score < best_score
            elif metric == 'mae':
                score = metrics['mae']
                is_better = score < best_score
            elif metric == 'r2':
                score = metrics['r2']
                is_better = score > best_score
            else:
                logger.warning(f"Unknown metric: {metric}, using RMSE")
                score = metrics['rmse']
                is_better = score < best_score
            
            logger.info(f"Model {name} {metric}: {score}")
            
            if is_better:
                best_score = score
                best_model = model
                logger.info(f"New best model: {name} with {metric} = {score}")
        
        except Exception as e:
            logger.error(f"Error fitting/evaluating model {name}: {e}")
    
    if best_model is None:
        logger.error("No valid models were successfully fitted")
        raise ValueError("No valid models were successfully fitted")
    
    return best_model