"""
Model configuration settings for the predictive model.
"""
from typing import Dict, Any, List, Optional

# Model categories
LINEAR_MODELS = ['linear_regression', 'lasso', 'ridge', 'elastic_net']
TIME_SERIES_MODELS = ['arima', 'sarima', 'prophet', 'ets']
ENSEMBLE_MODELS = ['random_forest', 'gradient_boosting', 'xgboost']
LIGHTGBM_MODELS = ['lightgbm', 'lightgbm_classifier', 'lightgbm_regressor']
NEURAL_NETWORK_MODELS = ['mlp', 'lstm']

# Default model configurations
DEFAULT_MODEL_CONFIGS = {
    # Linear models
    'linear_regression': {
        'fit_intercept': True,
    },
    'lasso': {
        'alpha': 0.1,
        'fit_intercept': True,
        'max_iter': 1000,
        'tol': 1e-4
    },
    'ridge': {
        'alpha': 1.0,
        'fit_intercept': True,
        'max_iter': 1000,
        'tol': 1e-4
    },
    'elastic_net': {
        'alpha': 0.1,
        'l1_ratio': 0.5,
        'fit_intercept': True,
        'max_iter': 1000,
        'tol': 1e-4
    },
    
    # Time series models
    'arima': {
        'p': 1,  # AR order
        'd': 1,  # differencing
        'q': 1,  # MA order
        'trend': 'c',   # Optionally, include a trend parameter if needed (e.g., 'trend': 'c')
    },
    'sarima': {
        'p': 1,  # AR order
        'd': 1,  # differencing
        'q': 1,  # MA order
        'P': 1,  # seasonal AR order
        'D': 1,  # seasonal differencing
        'Q': 1,  # seasonal MA order
        's': 12  # seasonal period (months)
    },
    'prophet': {
        # Prophet parameters
        'seasonality_mode': 'multiplicative',
        'yearly_seasonality': True,
        'weekly_seasonality': False,
        'daily_seasonality': False,
        'changepoint_prior_scale': 0.05,
        # Stan backend settings - not directly passed to Prophet
        'stan_settings': {
            'method': 'optimize',
            'algorithm': 'lbfgs',  # 'newton' or 'lbfgs'
            'iter': 2000,
            'refresh': 10,  # Show progress every N iterations
            'tol_rel_grad': 1e-07
        }
    },
    'ets': {
        'error': 'add',
        'trend': 'add',
        'seasonal': 'add',
        'seasonal_periods': 12
    },
    
    # Ensemble models
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'bootstrap': True,
        'random_state': 42
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'subsample': 1.0,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'random_state': 42
    },
    
    # LightGBM models
    'lightgbm': {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 100,
        'early_stopping_rounds': 50,
        'random_state': 42
    },
    'lightgbm_classifier': {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 100,
        'early_stopping_rounds': 50,
        'random_state': 42
    },
    'lightgbm_regressor': {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 100,
        'early_stopping_rounds': 50,
        'random_state': 42
    },
    
    # Neural network models
    'mlp': {
        'hidden_layer_sizes': (100,),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'batch_size': 'auto',
        'learning_rate': 'constant',
        'learning_rate_init': 0.001,
        'max_iter': 200,
        'random_state': 42
    },
    'lstm': {
        'units': 50,
        'activation': 'tanh',
        'recurrent_activation': 'sigmoid',
        'dropout': 0.0,
        'recurrent_dropout': 0.0,
        'batch_size': 32,
        'epochs': 100,
        'optimizer': 'adam',
        'loss': 'mse'
    },
}

# Hyperparameter grids for tuning
PARAM_GRIDS = {
    # Linear models
    'linear_regression': {
        'fit_intercept': [True, False],
    },
    'lasso': {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'fit_intercept': [True, False],
    },
    'ridge': {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'fit_intercept': [True, False],
    },
    'elastic_net': {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'fit_intercept': [True, False],
    },
    
    # Time series models
    'arima': {
        'p': [0, 1, 2],
        'd': [0, 1],
        'q': [0, 1, 2],
    },
    'sarima': {
        'p': [0, 1, 2],
        'd': [0, 1],
        'q': [0, 1, 2],
        'P': [0, 1],
        'D': [0, 1],
        'Q': [0, 1],
        's': [12],  # Monthly data
    },
    'prophet': {
        'seasonality_mode': ['additive', 'multiplicative'],
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'stan_settings': {
            'algorithm': ['newton', 'lbfgs'],
            'iter': [1000, 2000, 5000]
        }
    },
    'ets': {
        'error': ['add', 'mul'],
        'trend': ['add', 'mul', None],
        'seasonal': ['add', 'mul', None],
        'damped_trend': [True, False],
    },
    
    # Ensemble models
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    'gradient_boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0],
    },
    'xgboost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
    },
    
    # LightGBM models
    'lightgbm': {
        'num_leaves': [15, 31, 63],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
        'max_depth': [-1, 5, 10, 15],
        'min_child_samples': [5, 10, 20],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0.0, 0.1, 0.5],
        'reg_lambda': [0.0, 0.1, 0.5]
    },
    'lightgbm_classifier': {
        'num_leaves': [15, 31, 63],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
        'max_depth': [-1, 5, 10, 15],
        'min_child_samples': [5, 10, 20],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0.0, 0.1, 0.5],
        'reg_lambda': [0.0, 0.1, 0.5]
    },
    'lightgbm_regressor': {
        'num_leaves': [15, 31, 63],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
        'max_depth': [-1, 5, 10, 15],
        'min_child_samples': [5, 10, 20],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0.0, 0.1, 0.5],
        'reg_lambda': [0.0, 0.1, 0.5]
    },
    
    # Neural network models
    'mlp': {
        'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
        'activation': ['tanh', 'relu'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
    },
    'lstm': {
        'units': [32, 50, 64, 100],
        'dropout': [0.0, 0.1, 0.2],
        'recurrent_dropout': [0.0, 0.1, 0.2],
        'batch_size': [16, 32, 64],
        'epochs': [50, 100, 150],
    },
}

def get_model_config(model_type: str) -> Dict[str, Any]:
    """
    Get the default configuration for a specific model.
    
    Args:
        model_type: Type of model (e.g., 'linear_regression', 'arima', etc.)
        
    Returns:
        Dictionary with model configuration parameters
    """
    if model_type not in DEFAULT_MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return DEFAULT_MODEL_CONFIGS[model_type].copy()

def get_param_grid(model_type: str) -> Dict[str, List[Any]]:
    """
    Get the hyperparameter grid for tuning a specific model.
    
    Args:
        model_type: Type of model (e.g., 'linear_regression', 'arima', etc.)
        
    Returns:
        Dictionary with parameter grid for hyperparameter tuning
    """
    if model_type not in PARAM_GRIDS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return PARAM_GRIDS[model_type].copy()