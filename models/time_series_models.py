"""
Time series models for economic forecasting.
"""
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

class TimeSeriesModel:
    """
    Wrapper for time series models.
    """
    
    def __init__(self, model_type: str = 'arima', **kwargs):
        """
        Initialize a time series model.
        
        Args:
            model_type: Type of time series model 
                       ('arima', 'sarima', 'prophet', 'ets')
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.model_params = kwargs
        self.model = None
        self.is_fitted = False
        self.date_col = None
        self.target_col = None
        self.exog_cols = None
        
    def fit(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        date_col: str = 'date', 
        exog_cols: Optional[List[str]] = None
    ) -> 'TimeSeriesModel':
        """
        Fit the model to training data.
        
        Args:
            df: DataFrame with time series data
            target_col: Name of target column
            date_col: Name of date column
            exog_cols: Names of exogenous feature columns (optional)
            
        Returns:
            Self for method chaining
        """
        self.date_col = date_col
        self.target_col = target_col
        self.exog_cols = exog_cols
        
        # Make copy of data to avoid modifying original
        data = df.copy()
        
        # Ensure date is index for time series models
        if date_col in data.columns:
            data = data.set_index(date_col)
        
        # Extract target and exogenous variables
        y = data[target_col]
        X = data[exog_cols] if exog_cols else None
        
        if self.model_type == 'arima':
            # Set up ARIMA model parameters
            p = self.model_params.get('p', 1)
            d = self.model_params.get('d', 1)
            q = self.model_params.get('q', 1)
            
            # Initialize and fit model
            if X is not None:
                self.model = ARIMA(y, exog=X, order=(p, d, q))
                self.fit_result = self.model.fit()
            else:
                self.model = ARIMA(y, order=(p, d, q))
                self.fit_result = self.model.fit()
        
        elif self.model_type == 'sarima':
            # Set up SARIMA model parameters
            p = self.model_params.get('p', 1)
            d = self.model_params.get('d', 1)
            q = self.model_params.get('q', 1)
            P = self.model_params.get('P', 1)
            D = self.model_params.get('D', 1)
            Q = self.model_params.get('Q', 1)
            s = self.model_params.get('s', 12)  # Default to monthly seasonality
            
            # Initialize and fit model
            if X is not None:
                self.model = SARIMAX(
                    y, 
                    exog=X, 
                    order=(p, d, q), 
                    seasonal_order=(P, D, Q, s)
                )
                self.fit_result = self.model.fit(disp=False)
            else:
                self.model = SARIMAX(
                    y, 
                    order=(p, d, q), 
                    seasonal_order=(P, D, Q, s)
                )
                self.fit_result = self.model.fit(disp=False)
        
        elif self.model_type == 'prophet':
            # Prepare data for Prophet (requires specific format)
            prophet_data = pd.DataFrame({
                'ds': data.index,
                'y': y
            })
            
            # Add exogenous variables if provided
            if X is not None:
                regressor_df = pd.DataFrame({col: X[col].values for col in X.columns}, index=prophet_data.index)
                prophet_data = pd.concat([prophet_data, regressor_df], axis=1)
            
            # Extract stan_settings if present and remove from model_params
            prophet_params = self.model_params.copy()
            stan_settings = prophet_params.pop('stan_settings', None)
            
            # Set cmdstan backend 
            prophet_params['stan_backend'] = 'CMDSTANPY'
            
            # Create Prophet model with parameters
            self.model = Prophet(**prophet_params)
            
            # Add progress logging
            logger.info("Starting Prophet model fitting")
            
            # Add exogenous variables as regressors
            if X is not None:
                logger.info(f"Adding {len(X.columns)} exogenous variables as regressors")
                for col in X.columns:
                    self.model.add_regressor(col)
                    logger.debug(f"Added regressor: {col}")
            
            # Set cmdstan debug level to higher value for more output
            import cmdstanpy
            cmdstanpy.utils.get_logger().setLevel(logging.INFO)
            
            # Configure Stan options for more verbose output
            import os
            os.environ['CMDSTAN_VERBOSE'] = 'TRUE'
            
            # Report number of data points being fit
            logger.info(f"Fitting Prophet model on {len(prophet_data)} data points")
            
            # Fit the model with progress reporting
            try:
                # Add periodic progress updates during fitting
                logger.info("Running Stan optimization...")
                self.model.fit(prophet_data)
                logger.info("Prophet model fitting completed successfully")
            except Exception as e:
                logger.error(f"Error fitting Prophet model: {e}")
                raise
        
        elif self.model_type == 'ets':
            # Set up ETS model parameters
            seasonal_periods = self.model_params.get('seasonal_periods', 12)
            trend = self.model_params.get('trend', 'add')
            seasonal = self.model_params.get('seasonal', 'add')
            damped_trend = self.model_params.get('damped_trend', False)
            
            # Initialize and fit model (ETS doesn't support exogenous variables)
            if X is not None:
                logger.warning("ETS model does not support exogenous variables. Ignoring exog_cols.")
            
            self.model = ExponentialSmoothing(
                y,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                damped_trend=damped_trend
            )
            self.fit_result = self.model.fit()
        
        else:
            raise ValueError(f"Unknown time series model type: {self.model_type}")
        
        self.is_fitted = True
        return self
    
    def predict(
        self, 
        steps: int = 1, 
        future_exog: Optional[pd.DataFrame] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        include_history: bool = False
    ) -> pd.DataFrame:
        """
        Generate forecasts for future periods.
        
        Args:
            steps: Number of steps to forecast
            future_exog: Future values of exogenous variables (required if exog_cols was used)
            start: Start date for historical predictions (for include_history=True)
            end: End date for forecasting (alternative to steps)
            include_history: Whether to include historical predictions
            
        Returns:
            DataFrame with forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.model_type in ['arima', 'sarima']:
            # For ARIMA/SARIMA, check if exogenous variables are required
            if self.exog_cols and future_exog is None:
                raise ValueError("future_exog must be provided when model was fit with exogenous variables")
            
            if include_history:
                # Get historical in-sample predictions
                in_sample = self.fit_result.predict(start=start, end=end, exog=future_exog)
                result = pd.DataFrame(in_sample)
                result.columns = ['forecast']
                return result
            else:
                # Get out-of-sample forecasts
                forecast = self.fit_result.forecast(steps=steps, exog=future_exog)
                result = pd.DataFrame(forecast)
                result.columns = ['forecast']
                return result
        
        elif self.model_type == 'prophet':
            # Before creating the future DataFrame, get the correct date range
            if future_exog is not None:
                # Use the exogenous data's index for the future DataFrame
                future = pd.DataFrame({'ds': future_exog.index})
            else:
                # Create new date range for specified steps
                last_date = self.model.history['ds'].max()
                future = self.model.make_future_dataframe(
                    periods=steps,
                    freq='D'
                )

            # Add exogenous variables
            if self.exog_cols and future_exog is not None:
                for col in self.exog_cols:
                    future[col] = future_exog[col].values
            
            # Generate forecast
            forecast = self.model.predict(future)
            
            # If not including history, filter to only future dates
            if not include_history:
                last_train_date = self.model.history['ds'].max()
                forecast = forecast[forecast['ds'] > last_train_date]
            
            # Rename columns for consistency
            result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            result.columns = ['date', 'forecast', 'lower_bound', 'upper_bound']
            
            return result
        
        elif self.model_type == 'ets':
            # For ETS, there are no exogenous variables
            if include_history:
                # Get historical in-sample predictions
                in_sample = self.fit_result.fittedvalues
                result = pd.DataFrame(in_sample)
                result.columns = ['forecast']
                return result
            else:
                # Get out-of-sample forecasts
                forecast = self.fit_result.forecast(steps=steps)
                result = pd.DataFrame(forecast)
                result.columns = ['forecast']
                return result
        
        else:
            raise ValueError(f"Unknown time series model type: {self.model_type}")
    
    def evaluate(
        self, 
        test_df: pd.DataFrame,
        target_col: Optional[str] = None,
        date_col: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_df: DataFrame with test data
            target_col: Name of target column (defaults to self.target_col)
            date_col: Name of date column (defaults to self.date_col)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Use stored column names if not provided
        target_col = target_col or self.target_col
        date_col = date_col or self.date_col
        
        # Make copy of data to avoid modifying original
        data = test_df.copy()
        
        # Ensure date is index for time series models
        if date_col in data.columns:
            data = data.set_index(date_col)
        
        # Get actual target values
        y_true = data[target_col]
        
        # Generate predictions for test period
        if self.model_type in ['arima', 'sarima']:
            # For ARIMA/SARIMA, get exogenous variables if needed
            X_test = data[self.exog_cols] if self.exog_cols else None
            
            # Generate predictions
            predictions = self.fit_result.predict(
                start=data.index[0],
                end=data.index[-1],
                exog=X_test
            )
            
        elif self.model_type == 'prophet':
            # Prepare test data for Prophet
            prophet_data = pd.DataFrame({
                'ds': data.index,
                'y': y_true
            })
            
            # Add exogenous variables if needed
            if self.exog_cols:
                for col in self.exog_cols:
                    prophet_data[col] = data[col].values
            
            # Generate predictions
            future = prophet_data[['ds']].copy()
            if self.exog_cols:
                for col in self.exog_cols:
                    future[col] = prophet_data[col].values
            
            forecast = self.model.predict(future)
            predictions = forecast['yhat'].values
            
        elif self.model_type == 'ets':
            # For ETS, generate predictions
            predictions = self.fit_result.predict(
                start=data.index[0],
                end=data.index[-1]
            )
        
        # Calculate evaluation metrics
        metrics = {
            'mse': mean_squared_error(y_true, predictions),
            'rmse': np.sqrt(mean_squared_error(y_true, predictions)),
            'mae': mean_absolute_error(y_true, predictions),
            'r2': r2_score(y_true, predictions)
        }
        
        return metrics
    
    def get_summary(self) -> str:
        """
        Get summary statistics for the model (if available).
        
        Returns:
            Model summary string
        """
        if not self.is_fitted:
            return "Model not fitted yet."
        
        if self.model_type in ['arima', 'sarima']:
            return self.fit_result.summary().as_text()
        elif self.model_type == 'prophet':
            return "Summary statistics not available for Prophet models."
        elif self.model_type == 'ets':
            return str(self.fit_result.summary())
        else:
            return "Summary not available for this model type."
    
    def tune_hyperparameters(
        self, 
        df: pd.DataFrame,
        target_col: str,
        date_col: str = 'date',
        exog_cols: Optional[List[str]] = None,
        param_grid: Dict[str, List[Any]] = None,
        validation_steps: int = 1,
        scoring: str = 'mse'
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using a grid search with time series validation.
        
        Args:
            df: DataFrame with time series data
            target_col: Name of target column
            date_col: Name of date column
            exog_cols: Names of exogenous feature columns (optional)
            param_grid: Grid of parameters to search (dict of lists)
            validation_steps: Number of steps to use for validation
            scoring: Metric to use for evaluation ('mse', 'rmse', 'mae', 'r2')
            
        Returns:
            Dictionary with best parameters and best score
        """
        if param_grid is None:
            raise ValueError("param_grid must be provided for hyperparameter tuning")
        
        # Generate all parameter combinations
        import itertools
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        best_score = float('inf')  # Lower is better for MSE, RMSE, MAE
        best_params = None
        
        # Split data into train and validation sets
        train_size = len(df) - validation_steps
        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:].copy()
        
        # Try each parameter combination
        for params in param_combinations:
            param_dict = dict(zip(param_keys, params))
            
            # Update model parameters
            self.model_params.update(param_dict)
            
            # Fit model on training data
            self.fit(train_df, target_col=target_col, date_col=date_col, exog_cols=exog_cols)
            
            # Evaluate on validation data
            metrics = self.evaluate(val_df, target_col=target_col, date_col=date_col)
            
            # Get score based on specified metric
            if scoring == 'mse':
                score = metrics['mse']
            elif scoring == 'rmse':
                score = metrics['rmse']
            elif scoring == 'mae':
                score = metrics['mae']
            elif scoring == 'r2':
                # Convert to negative since we're minimizing
                score = -metrics['r2']
            
            # Update best parameters if this score is better
            if score < best_score:
                best_score = score
                best_params = param_dict
        
        # Update model with best parameters and refit on full data
        if best_params:
            self.model_params.update(best_params)
            self.fit(df, target_col=target_col, date_col=date_col, exog_cols=exog_cols)
        
        return {
            'best_params': best_params,
            'best_score': best_score
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
    def load(cls, filepath: str) -> 'TimeSeriesModel':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model
        """
        # Implement model loading logic
        pass