"""
Module for LightGBM-based forecasting model.

This module implements a forecasting model based on LightGBM, which performed
exceptionally well in the M5 forecasting competition. It supports multi-step
forecasting with dynamic anchor variables.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Import LightGBM
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

logger = logging.getLogger(__name__)

class LightGBMForecastModel:
    """
    LightGBM model for time series forecasting with dynamic anchor variables.
    
    This class implements a gradient boosting model for time series forecasting,
    incorporating best practices from the M5 competition for handling multi-step
    forecasts and time-dependent features.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the LightGBM forecasting model.
        
        Args:
            params: Dictionary of parameters for LightGBM
        """
        self.params = params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        self.model = None
        self.feature_importances = None
        
        # Check if LightGBM is available
        if lgb is None:
            logger.warning("LightGBM is not installed. Please install it with 'pip install lightgbm'.")
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        categorical_features: Optional[List[str]] = None,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = False
    ):
        """
        Fit the LightGBM model to training data.
        
        Args:
            X: Feature matrix
            y: Target vector
            categorical_features: List of categorical feature names
            eval_set: Tuple of (X_val, y_val) for early stopping
            early_stopping_rounds: Number of rounds for early stopping
            verbose: Whether to display training progress
            
        Returns:
            Self for method chaining
        """
        if lgb is None:
            raise ImportError("LightGBM is not installed. Please install it with 'pip install lightgbm'.")
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(
            X, 
            label=y,
            categorical_feature=categorical_features if categorical_features else 'auto'
        )
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if eval_set is not None:
            X_val, y_val = eval_set
            valid_data = lgb.Dataset(
                X_val,
                label=y_val,
                categorical_feature=categorical_features if categorical_features else 'auto',
                reference=train_data
            )
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        # Train the model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=1000,
            valid_sets=valid_sets,
            valid_names=valid_names,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=10 if verbose else False
        )
        
        # Store feature importances
        self.feature_importances = {
            name: importance for name, importance in zip(
                X.columns, 
                self.model.feature_importance(importance_type='gain')
            )
        }
        
        return self
    
    def predict(
        self, 
        X: pd.DataFrame, 
        num_iteration: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate predictions from the model.
        
        Args:
            X: Feature matrix
            num_iteration: Use this many iterations for prediction (None = best iteration)
            
        Returns:
            Array of predictions
        """
        if lgb is None:
            raise ImportError("LightGBM is not installed. Please install it with 'pip install lightgbm'.")
        
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X, num_iteration=num_iteration)
    
    def evaluate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X)
        
        # Calculate evaluation metrics
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
        return metrics
    
    def get_feature_importance(
        self, 
        importance_type: str = 'gain'
    ) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance to return ('gain', 'split', 'weight')
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        
        importances = self.model.feature_importance(importance_type=importance_type)
        feature_names = self.model.feature_name()
        
        return {name: importance for name, importance in zip(feature_names, importances)}
    
    def tune_hyperparameters(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using time series cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            param_grid: Grid of hyperparameters to search
            cv: Number of cross-validation folds
            verbose: Whether to display tuning progress
            
        Returns:
            Dictionary with best parameters and best score
        """
        if lgb is None:
            raise ImportError("LightGBM is not installed. Please install it with 'pip install lightgbm'.")
        
        import itertools
        from sklearn.model_selection import TimeSeriesSplit
        
        # Create parameter combinations
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        # Create time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv)
        
        best_score = float('inf')
        best_params = None
        
        # Iterate over parameter combinations
        for i, param_tuple in enumerate(param_combinations):
            # Create parameter dictionary
            params = self.params.copy()
            params.update({key: value for key, value in zip(param_keys, param_tuple)})
            
            if verbose:
                logger.info(f"Evaluating parameters {i+1}/{len(param_combinations)}: {params}")
            
            # Cross-validation
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Create datasets
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                # Train model
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=1000,
                    valid_sets=[val_data],
                    early_stopping_rounds=50,
                    verbose_eval=False
                )
                
                # Get validation score
                y_pred = model.predict(X_val)
                score = np.sqrt(mean_squared_error(y_val, y_pred))
                cv_scores.append(score)
            
            # Calculate mean score
            mean_score = np.mean(cv_scores)
            
            if verbose:
                logger.info(f"  Mean RMSE: {mean_score:.4f}")
            
            # Update best parameters if better
            if mean_score < best_score:
                best_score = mean_score
                best_params = params
        
        # Update model parameters
        self.params = best_params
        
        return {
            'best_params': best_params,
            'best_score': best_score
        }
    
    def forecast_multi_step(
        self, 
        X: pd.DataFrame,
        steps: int,
        dynamic_features: Optional[pd.DataFrame] = None,
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Generate multi-step forecasts.
        
        Args:
            X: Initial feature matrix
            steps: Number of steps to forecast
            dynamic_features: DataFrame with known future values for dynamic features
            date_col: Name of date column
            
        Returns:
            DataFrame with forecasts
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")
        
        # Make a copy of the input data
        X_forecast = X.copy()
        
        # Determine if X has a date index
        has_date_index = isinstance(X_forecast.index, pd.DatetimeIndex)
        
        # If date column is in X, set it as index
        if not has_date_index and date_col in X_forecast.columns:
            X_forecast = X_forecast.set_index(date_col)
            has_date_index = True
        
        # Get the last date or index value
        if has_date_index:
            last_date = X_forecast.index[-1]
            # Infer frequency
            freq = pd.infer_freq(X_forecast.index)
            if freq is None:
                # Estimate frequency
                if len(X_forecast.index) > 1:
                    freq = (X_forecast.index[1] - X_forecast.index[0])
                else:
                    freq = pd.DateOffset(days=1)  # Default to daily
        else:
            last_idx = X_forecast.index[-1]
        
        # Initialize results
        forecasts = []
        forecast_dates = []
        
        # Generate forecasts step by step
        last_features = X_forecast.iloc[-1:].copy()
        
        for i in range(steps):
            # Generate prediction for this step
            pred = self.predict(last_features)[0]
            forecasts.append(pred)
            
            # Create next date or index
            if has_date_index:
                if isinstance(freq, pd.DateOffset):
                    next_date = last_date + freq
                else:
                    next_date = last_date + pd.DateOffset(days=freq.days)
                forecast_dates.append(next_date)
                last_date = next_date
            else:
                next_idx = last_idx + 1
                forecast_dates.append(next_idx)
                last_idx = next_idx
            
            # Update features for next step
            next_features = last_features.copy()
            
            # Update dynamic features if provided
            if dynamic_features is not None:
                for col in dynamic_features.columns:
                    if col in next_features.columns:
                        if i < len(dynamic_features):
                            next_features[col] = dynamic_features.iloc[i][col]
            
            # Update any autoregressive features
            # (This would need to be extended for real implementation)
            
            # Use the new features for the next step
            last_features = next_features
        
        # Create forecast DataFrame
        if has_date_index:
            forecast_df = pd.DataFrame({
                'forecast': forecasts
            }, index=forecast_dates)
            # Reset index to get date as column
            forecast_df = forecast_df.reset_index()
            forecast_df.rename(columns={'index': date_col}, inplace=True)
        else:
            forecast_df = pd.DataFrame({
                'index': forecast_dates,
                'forecast': forecasts
            })
        
        return forecast_df
    
    def save_model(self, filepath: str):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        self.model.save_model(filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'LightGBMForecastModel':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the model file
            
        Returns:
            Loaded LightGBMForecastModel
        """
        if lgb is None:
            raise ImportError("LightGBM is not installed. Please install it with 'pip install lightgbm'.")
        
        model = cls()
        model.model = lgb.Booster(model_file=filepath)
        
        # Extract feature importances
        if model.model:
            importances = model.model.feature_importance(importance_type='gain')
            feature_names = model.model.feature_name()
            model.feature_importances = {
                name: importance for name, importance in zip(feature_names, importances)
            }
        
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def feature_engineering(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        date_col: str = 'date',
        lags: List[int] = [1, 2, 3, 7, 14, 28],
        rolling_windows: List[int] = [7, 14, 30, 90],
        add_date_features: bool = True
    ) -> pd.DataFrame:
        """
        Perform feature engineering for time series forecasting.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            date_col: Name of date column
            lags: List of lag values to create
            rolling_windows: List of rolling window sizes
            add_date_features: Whether to add date-based features
            
        Returns:
            DataFrame with engineered features
        """
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Ensure date column is datetime if it exists
        if date_col in data.columns:
            data[date_col] = pd.to_datetime(data[date_col])
            
            # Set date as index for easier time series operations
            data = data.set_index(date_col)
        
        # Create lag features
        for lag in lags:
            data[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
        
        # Create rolling window features
        for window in rolling_windows:
            data[f'{target_col}_rolling_mean_{window}'] = data[target_col].rolling(window=window).mean()
            data[f'{target_col}_rolling_std_{window}'] = data[target_col].rolling(window=window).std()
            data[f'{target_col}_rolling_min_{window}'] = data[target_col].rolling(window=window).min()
            data[f'{target_col}_rolling_max_{window}'] = data[target_col].rolling(window=window).max()
        
        # Add percent change features
        data[f'{target_col}_pct_change_1'] = data[target_col].pct_change(periods=1)
        data[f'{target_col}_pct_change_7'] = data[target_col].pct_change(periods=7)
        data[f'{target_col}_pct_change_30'] = data[target_col].pct_change(periods=30)
        
        # Add date features if requested
        if add_date_features and isinstance(data.index, pd.DatetimeIndex):
            # Month and day features
            data['month'] = data.index.month
            data['day_of_month'] = data.index.day
            data['day_of_week'] = data.index.dayofweek
            data['day_of_year'] = data.index.dayofyear
            data['quarter'] = data.index.quarter
            
            # Week of year
            data['week_of_year'] = data.index.isocalendar().week
            
            # Is weekend
            data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
            
            # Is month start/end
            data['is_month_start'] = data.index.is_month_start.astype(int)
            data['is_month_end'] = data.index.is_month_end.astype(int)
            
            # Is quarter start/end
            data['is_quarter_start'] = data.index.is_quarter_start.astype(int)
            data['is_quarter_end'] = data.index.is_quarter_end.astype(int)
            
            # Is year start/end
            data['is_year_start'] = data.index.is_year_start.astype(int)
            data['is_year_end'] = data.index.is_year_end.astype(int)
            
            # Year
            data['year'] = data.index.year
            
            # Add cyclical encoding for month, day of week, etc.
            data['month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
            data['month_cos'] = np.cos(2 * np.pi * data.index.month / 12)
            data['day_of_week_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
            data['day_of_week_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)
            data['day_of_year_sin'] = np.sin(2 * np.pi * data.index.dayofyear / 365.25)
            data['day_of_year_cos'] = np.cos(2 * np.pi * data.index.dayofyear / 365.25)
        
        # Reset index if we set it earlier
        if date_col in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            data = data.reset_index()
        
        # Drop rows with NaN values (from lag and rolling calculations)
        data = data.dropna()
        
        return data
    
    def create_training_examples(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        sequence_length: int = 30,
        forecast_horizon: int = 1,
        stride: int = 1,
        date_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create training examples for sequence prediction (for LSTM-like approaches).
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            sequence_length: Length of input sequences
            forecast_horizon: Number of steps to forecast
            stride: Stride for creating sequences
            date_col: Name of date column (to include in output)
            
        Returns:
            Tuple of (X, y) DataFrames with features and targets
        """
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Optional: Set date as index if date_col is provided
        if date_col is not None and date_col in data.columns:
            data[date_col] = pd.to_datetime(data[date_col])
            data = data.set_index(date_col)
        
        # Get column names excluding target
        feature_cols = [col for col in data.columns if col != target_col]
        
        # Lists to store features and targets
        X_list = []
        y_list = []
        dates_list = []
        
        # Create sequences
        for i in range(0, len(data) - sequence_length - forecast_horizon + 1, stride):
            X_sequence = data.iloc[i:i+sequence_length][feature_cols]
            y_value = data.iloc[i+sequence_length+forecast_horizon-1][target_col]
            
            # Flatten sequence into a single row
            X_flat = X_sequence.values.flatten()
            
            # Create column names for the flattened sequence
            flat_columns = [f"{col}_{j}" for j in range(sequence_length) for col in feature_cols]
            
            X_list.append(dict(zip(flat_columns, X_flat)))
            y_list.append(y_value)
            
            # If date index, store the target date
            if isinstance(data.index, pd.DatetimeIndex):
                target_date = data.index[i+sequence_length+forecast_horizon-1]
                dates_list.append(target_date)
        
        # Convert lists to DataFrames
        X = pd.DataFrame(X_list)
        y = pd.Series(y_list)
        
        # Add dates if available
        if dates_list and isinstance(data.index, pd.DatetimeIndex):
            dates = pd.Series(dates_list)
            # Return with dates
            return X, y, dates
        
        return X, y
    
    def create_multistep_training_examples(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        sequence_length: int = 30,
        forecast_horizons: List[int] = [1, 7, 14],
        stride: int = 1,
        date_col: Optional[str] = None
    ) -> Dict[int, Tuple[pd.DataFrame, pd.Series]]:
        """
        Create training examples for multi-step forecasting.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            sequence_length: Length of input sequences
            forecast_horizons: List of forecast horizons
            stride: Stride for creating sequences
            date_col: Name of date column (to include in output)
            
        Returns:
            Dictionary with forecast horizons as keys and (X, y) tuples as values
        """
        results = {}
        
        for horizon in forecast_horizons:
            result = self.create_training_examples(
                df, target_col, sequence_length, horizon, stride, date_col
            )
            results[horizon] = result
        
        return results
    
    def direct_multistep_forecast(
        self,
        df: pd.DataFrame,
        target_col: str,
        steps: List[int],
        features: Optional[List[str]] = None,
        date_col: str = 'date',
        categorical_features: Optional[List[str]] = None,
        test_size: float = 0.2,
        verbose: bool = False
    ) -> Dict[int, 'LightGBMForecastModel']:
        """
        Train separate models for direct multi-step forecasting.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            steps: List of forecast horizons
            features: List of feature columns to use
            date_col: Name of date column
            categorical_features: List of categorical feature names
            test_size: Proportion of data to use for testing
            verbose: Whether to display training progress
            
        Returns:
            Dictionary with forecast horizons as keys and trained models as values
        """
        models = {}
        
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Ensure date column is datetime if it exists
        if date_col in data.columns:
            data[date_col] = pd.to_datetime(data[date_col])
            
        # Set features if not provided
        if features is None:
            features = [col for col in data.columns if col != target_col and col != date_col]
        
        # Split data by time
        data = data.sort_values(by=date_col)
        train_size = int(len(data) * (1 - test_size))
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        for step in steps:
            if verbose:
                logger.info(f"Training model for {step}-step ahead forecast")
            
            # Create target with appropriate shift
            X_train = train_data[features].copy()
            y_train = train_data[target_col].shift(-step).dropna()
            
            # Adjust X_train to match y_train length
            X_train = X_train.iloc[:len(y_train)]
            
            # Create validation set
            X_val = test_data[features].copy()
            y_val = test_data[target_col].shift(-step).dropna()
            X_val = X_val.iloc[:len(y_val)]
            
            # Skip if not enough data
            if len(y_train) < 10 or len(y_val) < 5:
                logger.warning(f"Not enough data for {step}-step ahead forecast")
                continue
            
            # Create and train model
            model = LightGBMForecastModel(self.params.copy())
            model.fit(
                X_train, 
                y_train,
                categorical_features=categorical_features,
                eval_set=(X_val, y_val),
                verbose=verbose
            )
            
            # Evaluate model
            metrics = model.evaluate(X_val, y_val)
            
            if verbose:
                logger.info(f"  RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
            
            # Store model
            models[step] = model
        
        return models
    
    def recursive_multistep_forecast(
        self,
        df: pd.DataFrame,
        target_col: str,
        steps: int,
        features: Optional[List[str]] = None,
        date_col: str = 'date',
        categorical_features: Optional[List[str]] = None,
        dynamic_features: Optional[pd.DataFrame] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Generate multi-step forecasts recursively using a single model.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            steps: Number of steps to forecast
            features: List of feature columns to use
            date_col: Name of date column
            categorical_features: List of categorical feature names
            dynamic_features: DataFrame with known future values for dynamic features
            verbose: Whether to display progress
            
        Returns:
            DataFrame with forecasts
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")
        
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Ensure date column is datetime if it exists
        if date_col in data.columns:
            data[date_col] = pd.to_datetime(data[date_col])
        
        # Set features if not provided
        if features is None:
            features = [col for col in data.columns if col != target_col and col != date_col]
        
        # Get the last row for initial forecast
        last_data = data.iloc[[-1]].copy()
        
        # Create forecast DataFrame
        if date_col in data.columns:
            last_date = last_data[date_col].iloc[0]
            
            # Determine frequency
            if len(data) > 1:
                # Try to infer frequency
                dates = pd.DatetimeIndex(data[date_col])
                freq = pd.infer_freq(dates)
                
                if freq is None:
                    # Estimate frequency from average difference
                    date_diffs = np.diff(dates.astype(np.int64)) / 10**9  # Convert to seconds
                    avg_diff = np.mean(date_diffs)
                    
                    # Convert to days
                    days = avg_diff / (24 * 60 * 60)
                    
                    if days < 2:
                        freq = pd.DateOffset(days=1)  # Daily
                    elif days < 8:
                        freq = pd.DateOffset(days=7)  # Weekly
                    elif days < 35:
                        freq = pd.DateOffset(months=1)  # Monthly
                    else:
                        freq = pd.DateOffset(months=3)  # Quarterly
                else:
                    freq = pd.tseries.frequencies.to_offset(freq)
            else:
                # Default to daily if only one data point
                freq = pd.DateOffset(days=1)
                
            # Create date range for forecast
            forecast_dates = pd.date_range(
                start=last_date + freq,
                periods=steps,
                freq=freq
            )
        else:
            # Use integer indices if no date column
            last_idx = data.index[-1]
            forecast_dates = range(last_idx + 1, last_idx + steps + 1)
        
        # Initialize forecast DataFrame
        forecasts = pd.DataFrame(index=range(steps))
        forecasts[date_col] = forecast_dates
        forecasts[target_col] = np.nan
        
        # Add dynamic features if provided
        if dynamic_features is not None:
            for col in dynamic_features.columns:
                if col in features:
                    forecasts[col] = dynamic_features[col].values[:steps]
        
        # Generate forecasts recursively
        for i in range(steps):
            # Prepare input for prediction
            if i == 0:
                input_data = last_data[features].copy()
            else:
                input_data = forecasts.iloc[[i-1]][features].copy()
            
            # Generate prediction
            pred = self.predict(input_data)[0]
            
            # Store prediction
            forecasts.loc[i, target_col] = pred
            
            # Update dynamic features for next step if needed
            for col in features:
                if col not in forecasts.columns:
                    if col.startswith(f'{target_col}_lag_'):
                        lag = int(col.split('_')[-1])
                        if i >= lag:
                            forecasts.loc[i, col] = forecasts.loc[i-lag, target_col]
                        else:
                            # Get value from original data
                            idx = -1 - (lag - i - 1)
                            if abs(idx) <= len(data):
                                forecasts.loc[i, col] = data.iloc[idx][target_col]
                    elif col.startswith(f'{target_col}_pct_change_'):
                        periods = int(col.split('_')[-1])
                        if i >= periods:
                            prev_value = forecasts.loc[i-periods, target_col]
                            curr_value = forecasts.loc[i, target_col]
                            if prev_value != 0:
                                pct_change = (curr_value - prev_value) / prev_value
                                forecasts.loc[i, col] = pct_change
                    elif '_rolling_' in col:
                        # For rolling features, we'd need more complex logic
                        # This is just a placeholder
                        forecasts.loc[i, col] = data[col].iloc[-1]
        
        return forecasts