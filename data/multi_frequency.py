"""
Module for handling multi-frequency data in economic forecasting.

This module implements methods for temporal aggregation and disaggregation,
allowing the system to work with data at different frequencies (monthly, quarterly, annual)
and align them for consistent forecasting.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)

class MultiFrequencyHandler:
    """
    Class for handling data with different frequencies.
    
    This implements methods for converting between different time frequencies
    (e.g., monthly to quarterly, quarterly to monthly) using various techniques
    such as interpolation, Denton method, and MIDAS regression.
    """
    
    def __init__(self):
        """Initialize the MultiFrequencyHandler."""
        self.conversion_history = {}
        
    def aggregate_to_lower_frequency(
        self, 
        df: pd.DataFrame, 
        date_col: str = 'date',
        target_frequency: str = 'QS',  # 'QS' for quarter start, 'AS' for annual start
        aggregation_method: Union[str, Dict[str, str]] = 'mean'
    ) -> pd.DataFrame:
        """
        Aggregate higher frequency data (e.g., monthly) to lower frequency (e.g., quarterly).
        
        Args:
            df: DataFrame with time series data
            date_col: Name of date column
            target_frequency: Target frequency ('QS', 'AS', etc.)
            aggregation_method: Method for aggregation ('mean', 'sum', 'last', etc.)
                or a dictionary mapping column names to aggregation methods
                
        Returns:
            DataFrame with aggregated data at the target frequency
        """
        # Make a copy to avoid modifying the original DataFrame
        data = df.copy()
        
        # Ensure date column is datetime and set as index
        if date_col in data.columns:
            data[date_col] = pd.to_datetime(data[date_col])
            data = data.set_index(date_col)
        
        # Handle different aggregation methods for different columns
        if isinstance(aggregation_method, dict):
            agg_dict = aggregation_method
        else:
            agg_dict = {col: aggregation_method for col in data.columns}
        
        # Perform resampling with the specified aggregation methods
        aggregated = data.resample(target_frequency).agg(agg_dict)
        
        # Reset index to get date as a column again
        aggregated.reset_index(inplace=True)
        aggregated.rename(columns={'index': date_col}, inplace=True)
        
        logger.info(f"Aggregated data from {len(data)} records to {len(aggregated)} records "
                   f"with frequency '{target_frequency}'")
        
        return aggregated
    
    def disaggregate_simple(
        self, 
        low_freq_df: pd.DataFrame,
        date_col: str = 'date',
        target_frequency: str = 'MS',  # 'MS' for month start
        method: str = 'linear'  # 'linear', 'cubic', 'constant', etc.
    ) -> pd.DataFrame:
        """
        Disaggregate lower frequency data (e.g., quarterly) to higher frequency (e.g., monthly)
        using simple interpolation methods.
        
        Args:
            low_freq_df: DataFrame with low-frequency data
            date_col: Name of date column
            target_frequency: Target frequency ('MS', 'D', etc.)
            method: Interpolation method ('linear', 'cubic', 'constant', etc.)
                
        Returns:
            DataFrame with disaggregated data at the target frequency
        """
        # Make a copy to avoid modifying the original DataFrame
        data = low_freq_df.copy()
        
        # Ensure date column is datetime and set as index
        if date_col in data.columns:
            data[date_col] = pd.to_datetime(data[date_col])
            data = data.set_index(date_col)
        
        # Create a date range with the target frequency
        start_date = data.index.min()
        end_date = data.index.max()
        new_index = pd.date_range(start=start_date, end=end_date, freq=target_frequency)
        
        # Reindex and interpolate
        disaggregated = data.reindex(new_index)
        for col in disaggregated.columns:
            disaggregated[col] = disaggregated[col].interpolate(method=method)
        
        # Reset index to get date as a column again
        disaggregated.reset_index(inplace=True)
        disaggregated.rename(columns={'index': date_col}, inplace=True)
        
        logger.info(f"Disaggregated data from {len(data)} records to {len(disaggregated)} records "
                   f"with frequency '{target_frequency}' using {method} interpolation")
        
        return disaggregated
    
    def denton_disaggregation(
        self,
        low_freq_series: pd.Series,
        high_freq_indicator: pd.Series,
        date_col: str = 'date',
        low_freq: str = 'QS',
        high_freq: str = 'MS'
    ) -> pd.Series:
        """
        Implement the Denton method for temporal disaggregation.
        
        This method disaggregates low-frequency data (e.g., quarterly) to high-frequency data
        (e.g., monthly) using a related high-frequency indicator series, ensuring that the
        disaggregated values sum up to the original low-frequency values.
        
        Args:
            low_freq_series: Series with low-frequency data
            high_freq_indicator: Series with high-frequency indicator data
            date_col: Name of date column
            low_freq: Frequency of low_freq_series ('QS', 'AS', etc.)
            high_freq: Target frequency ('MS', 'D', etc.)
                
        Returns:
            Series with disaggregated data at the high frequency
        """
        # Ensure both series have datetime indices
        if not isinstance(low_freq_series.index, pd.DatetimeIndex):
            if date_col in low_freq_series.index.names:
                low_freq_series = low_freq_series.reset_index()
                low_freq_series = low_freq_series.set_index(date_col)
            else:
                raise ValueError("low_freq_series must have a datetime index or a date column")
        
        if not isinstance(high_freq_indicator.index, pd.DatetimeIndex):
            if date_col in high_freq_indicator.index.names:
                high_freq_indicator = high_freq_indicator.reset_index()
                high_freq_indicator = high_freq_indicator.set_index(date_col)
            else:
                raise ValueError("high_freq_indicator must have a datetime index or a date column")
        
        # Create aggregation mapping matrix
        start_date = max(low_freq_series.index.min(), high_freq_indicator.index.min())
        end_date = min(low_freq_series.index.max(), high_freq_indicator.index.max())
        
        # Create high-frequency date range
        high_freq_dates = pd.date_range(start=start_date, end=end_date, freq=high_freq)
        
        # Create low-frequency date range
        low_freq_dates = pd.date_range(start=start_date, end=end_date, freq=low_freq)
        
        # Filter indicator to the common date range
        indicator = high_freq_indicator.reindex(high_freq_dates)
        
        # Handle missing values in indicator
        indicator = indicator.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        
        # Filter low-frequency series to the common date range
        low_freq_values = low_freq_series.reindex(low_freq_dates)
        
        # Create aggregation matrix
        C = np.zeros((len(low_freq_dates), len(high_freq_dates)))
        
        for i, low_date in enumerate(low_freq_dates):
            # Determine which high-frequency dates belong to this low-frequency period
            if low_freq == 'QS':
                # For quarterly, include the next 3 months
                period_start = low_date
                period_end = low_date + pd.DateOffset(months=3) - pd.DateOffset(days=1)
            elif low_freq == 'AS':
                # For annual, include the next 12 months
                period_start = low_date
                period_end = low_date + pd.DateOffset(years=1) - pd.DateOffset(days=1)
            else:
                # Handle other frequencies as needed
                raise ValueError(f"Unsupported low frequency: {low_freq}")
            
            # Find indices of high-frequency dates in this period
            mask = (high_freq_dates >= period_start) & (high_freq_dates <= period_end)
            C[i, mask] = 1
        
        # Handle potential missing values in low_freq_values
        low_freq_values = low_freq_values.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        
        # Apply Denton method
        indicator_values = indicator.values
        
        # Normalize indicator to have same sum as low-frequency values within each period
        normalized_indicator = np.zeros_like(indicator_values)
        
        for i in range(len(low_freq_dates)):
            period_mask = C[i, :] > 0
            if np.sum(period_mask) > 0 and np.sum(indicator_values[period_mask]) > 0:
                normalized_indicator[period_mask] = (
                    indicator_values[period_mask] * 
                    low_freq_values.iloc[i] / 
                    np.sum(indicator_values[period_mask])
                )
        
        # Create result series
        result = pd.Series(normalized_indicator, index=high_freq_dates)
        
        logger.info(f"Performed Denton disaggregation from {len(low_freq_values)} {low_freq} periods "
                   f"to {len(result)} {high_freq} periods")
        
        return result
    
    def mixed_frequency_model(
        self,
        target_series: pd.Series,
        predictor_series: List[pd.Series],
        forecast_periods: int = 12,
        target_frequency: str = 'MS'
    ) -> pd.Series:
        """
        Implement a simple mixed frequency regression model (MIDAS-like approach).
        
        This method estimates a regression model using predictors at different frequencies
        and generates forecasts at the target frequency.
        
        Args:
            target_series: Series with target variable data
            predictor_series: List of predictor series (can have different frequencies)
            forecast_periods: Number of periods to forecast
            target_frequency: Frequency of the target series ('MS', 'QS', etc.)
                
        Returns:
            Series with forecasts at the target frequency
        """
        # Ensure target has datetime index
        if not isinstance(target_series.index, pd.DatetimeIndex):
            raise ValueError("target_series must have a datetime index")
        
        # Align all predictor series to target frequency
        aligned_predictors = []
        
        for i, predictor in enumerate(predictor_series):
            # Determine the predictor's frequency
            if isinstance(predictor.index, pd.DatetimeIndex):
                pred_freq = pd.infer_freq(predictor.index)
                if not pred_freq:
                    # If frequency can't be inferred, assume it's the same as target
                    pred_freq = target_frequency
            else:
                raise ValueError(f"Predictor {i} must have a datetime index")
            
            # Align to target frequency
            if pred_freq == target_frequency:
                aligned_predictors.append(predictor)
            else:
                # If predictor has lower frequency than target, disaggregate
                if pred_freq in ['QS', 'AS'] and target_frequency == 'MS':
                    # Create a dummy indicator at target frequency
                    dummy_indicator = pd.Series(
                        1,
                        index=pd.date_range(
                            start=predictor.index.min(),
                            end=predictor.index.max(),
                            freq=target_frequency
                        )
                    )
                    # Use Denton method for disaggregation
                    disaggregated = self.denton_disaggregation(
                        predictor, dummy_indicator, low_freq=pred_freq, high_freq=target_frequency
                    )
                    aligned_predictors.append(disaggregated)
                else:
                    # For other cases, use simple interpolation
                    reindexed = predictor.reindex(
                        pd.date_range(
                            start=predictor.index.min(),
                            end=predictor.index.max(),
                            freq=target_frequency
                        )
                    )
                    interpolated = reindexed.interpolate(method='linear')
                    aligned_predictors.append(interpolated)
        
        # Create a dataframe with target and aligned predictors
        data = pd.DataFrame({'target': target_series})
        
        for i, pred in enumerate(aligned_predictors):
            data[f'predictor_{i}'] = pred
        
        # Handle missing values
        data = data.dropna()
        
        if len(data) <= 1:
            logger.warning("Not enough data points for regression after alignment")
            return None
        
        # Fit regression model
        X = data.drop('target', axis=1)
        y = data['target']
        
        model = sm.OLS(y, sm.add_constant(X)).fit()
        
        # Generate forecasts
        last_date = target_series.index.max()
        forecast_dates = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(target_frequency),
            periods=forecast_periods,
            freq=target_frequency
        )
        
        # Prepare forecast inputs
        forecast_X = pd.DataFrame(index=forecast_dates)
        
        # Add predictor values for forecast period
        for i, predictor in enumerate(predictor_series):
            pred_series = aligned_predictors[i]
            
            # Extend each predictor series for the forecast period
            # This is a simple approach - in practice, you might use ARIMA or other methods
            # to forecast each predictor individually
            pred_forecast = self._forecast_series(pred_series, forecast_periods)
            
            forecast_X[f'predictor_{i}'] = pred_forecast
        
        # Add constant term
        forecast_X = sm.add_constant(forecast_X)
        
        # Generate forecasts
        forecasts = model.predict(forecast_X)
        
        return pd.Series(forecasts, index=forecast_dates)
    
    def _forecast_series(self, series: pd.Series, periods: int) -> pd.Series:
        """
        Simple method to forecast a series for a specified number of periods.
        
        Args:
            series: Input time series
            periods: Number of periods to forecast
            
        Returns:
            Series with forecasted values
        """
        from statsmodels.tsa.arima.model import ARIMA
        
        # Remove NaN values
        clean_series = series.dropna()
        
        if len(clean_series) <= 2:
            # Not enough data for ARIMA, use simple mean
            forecast_value = clean_series.mean()
            forecast_index = pd.date_range(
                start=clean_series.index[-1] + pd.tseries.frequencies.to_offset(pd.infer_freq(clean_series.index)),
                periods=periods,
                freq=pd.infer_freq(clean_series.index)
            )
            return pd.Series([forecast_value] * periods, index=forecast_index)
        
        try:
            # Fit a simple ARIMA model
            model = ARIMA(clean_series, order=(1, 1, 0))
            fit_model = model.fit()
            
            # Generate forecasts
            forecasts = fit_model.forecast(steps=periods)
            return forecasts
        except Exception as e:
            logger.warning(f"Error in ARIMA forecast: {e}. Using simple extrapolation.")
            
            # Fallback to simple trend extrapolation
            from sklearn.linear_model import LinearRegression
            
            X = np.arange(len(clean_series)).reshape(-1, 1)
            y = clean_series.values
            
            model = LinearRegression()
            model.fit(X, y)
            
            X_forecast = np.arange(len(clean_series), len(clean_series) + periods).reshape(-1, 1)
            forecasts = model.predict(X_forecast)
            
            forecast_index = pd.date_range(
                start=clean_series.index[-1] + pd.tseries.frequencies.to_offset(pd.infer_freq(clean_series.index)),
                periods=periods,
                freq=pd.infer_freq(clean_series.index)
            )
            
            return pd.Series(forecasts, index=forecast_index)
    
    def reconcile_hierarchical_forecasts(
        self,
        forecasts_dict: Dict[str, pd.DataFrame],
        hierarchy: Dict[str, List[str]],
        method: str = 'bottom_up'
    ) -> Dict[str, pd.DataFrame]:
        """
        Reconcile forecasts across hierarchical levels for consistency.
        
        Args:
            forecasts_dict: Dictionary of DataFrames with forecasts at different levels
            hierarchy: Dictionary defining hierarchical relationships
            method: Reconciliation method ('bottom_up', 'top_down', or 'optimal')
                
        Returns:
            Dictionary with reconciled forecasts
        """
        if method == 'bottom_up':
            return self._reconcile_bottom_up(forecasts_dict, hierarchy)
        elif method == 'top_down':
            return self._reconcile_top_down(forecasts_dict, hierarchy)
        elif method == 'optimal':
            return self._reconcile_optimal(forecasts_dict, hierarchy)
        else:
            logger.warning(f"Unknown reconciliation method: {method}. Using bottom-up.")
            return self._reconcile_bottom_up(forecasts_dict, hierarchy)
    
    def _reconcile_bottom_up(
        self,
        forecasts_dict: Dict[str, pd.DataFrame],
        hierarchy: Dict[str, List[str]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Implement bottom-up reconciliation of hierarchical forecasts.
        
        Args:
            forecasts_dict: Dictionary of DataFrames with forecasts at different levels
            hierarchy: Dictionary defining hierarchical relationships
                
        Returns:
            Dictionary with reconciled forecasts
        """
        # Create a copy of the original forecasts
        reconciled = {k: v.copy() for k, v in forecasts_dict.items()}
        
        # Identify the bottom level (leaves) in the hierarchy
        all_nodes = set(hierarchy.keys())
        all_children = set()
        for children in hierarchy.values():
            all_children.update(children)
        
        leaves = all_children - all_nodes
        
        # Reconcile from bottom to top
        for parent, children in hierarchy.items():
            # Check if any children are in the leaves or already reconciled
            valid_children = [c for c in children if c in reconciled]
            
            if valid_children:
                # Aggregate the children's forecasts to create the parent's reconciled forecast
                parent_forecast = reconciled[valid_children[0]].copy()
                parent_forecast.iloc[:, :] = 0  # Reset values
                
                for child in valid_children:
                    parent_forecast += reconciled[child]
                
                # Replace the parent's forecast with the aggregated one
                reconciled[parent] = parent_forecast
        
        return reconciled
    
    def _reconcile_top_down(
        self,
        forecasts_dict: Dict[str, pd.DataFrame],
        hierarchy: Dict[str, List[str]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Implement top-down reconciliation of hierarchical forecasts.
        
        Args:
            forecasts_dict: Dictionary of DataFrames with forecasts at different levels
            hierarchy: Dictionary defining hierarchical relationships
                
        Returns:
            Dictionary with reconciled forecasts
        """
        # Create a copy of the original forecasts
        reconciled = {k: v.copy() for k, v in forecasts_dict.items()}
        
        # Identify the top level (root) in the hierarchy
        all_nodes = set(hierarchy.keys())
        all_children = set()
        for children in hierarchy.values():
            all_children.update(children)
        
        roots = all_nodes - all_children
        
        if not roots:
            logger.warning("No root node found in hierarchy. Using first node as root.")
            root = next(iter(hierarchy.keys()))
        else:
            root = next(iter(roots))
        
        # Calculate historical proportions for each node
        proportions = self._calculate_historical_proportions(forecasts_dict, hierarchy)
        
        # Reconcile from top to down
        self._reconcile_node(root, hierarchy, reconciled, proportions)
        
        return reconciled
    
    def _calculate_historical_proportions(
        self,
        forecasts_dict: Dict[str, pd.DataFrame],
        hierarchy: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Calculate historical proportions for top-down reconciliation.
        
        Args:
            forecasts_dict: Dictionary of DataFrames with forecasts at different levels
            hierarchy: Dictionary defining hierarchical relationships
                
        Returns:
            Dictionary with proportions for each node
        """
        proportions = {}
        
        for parent, children in hierarchy.items():
            if parent in forecasts_dict and all(c in forecasts_dict for c in children):
                parent_data = forecasts_dict[parent].iloc[:-1]  # Historical data (exclude forecasts)
                
                for child in children:
                    child_data = forecasts_dict[child].iloc[:-1]  # Historical data
                    
                    # Calculate average proportion
                    if not parent_data.empty and not child_data.empty:
                        parent_sum = parent_data.sum().sum()
                        child_sum = child_data.sum().sum()
                        
                        if parent_sum > 0:
                            proportions[(parent, child)] = child_sum / parent_sum
                        else:
                            # Equal distribution if parent sum is zero
                            proportions[(parent, child)] = 1.0 / len(children)
                    else:
                        # Equal distribution if data is missing
                        proportions[(parent, child)] = 1.0 / len(children)
        
        return proportions
    
    def _reconcile_node(
        self,
        node: str,
        hierarchy: Dict[str, List[str]],
        reconciled: Dict[str, pd.DataFrame],
        proportions: Dict[Tuple[str, str], float]
    ):
        """
        Recursively reconcile a node and its children in top-down fashion.
        
        Args:
            node: Current node to reconcile
            hierarchy: Dictionary defining hierarchical relationships
            reconciled: Dictionary with reconciled forecasts (modified in-place)
            proportions: Dictionary with proportions for each node
        """
        if node in hierarchy:
            children = hierarchy[node]
            
            for child in children:
                if (node, child) in proportions and child in reconciled:
                    # Distribute parent forecast to child based on proportion
                    proportion = proportions[(node, child)]
                    reconciled[child] = reconciled[node] * proportion
                    
                    # Recursively reconcile child's children
                    self._reconcile_node(child, hierarchy, reconciled, proportions)
    
    def _reconcile_optimal(
        self,
        forecasts_dict: Dict[str, pd.DataFrame],
        hierarchy: Dict[str, List[str]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Implement optimal reconciliation of hierarchical forecasts using MinT method.
        
        Args:
            forecasts_dict: Dictionary of DataFrames with forecasts at different levels
            hierarchy: Dictionary defining hierarchical relationships
                
        Returns:
            Dictionary with reconciled forecasts
        """
        # This is a simplified implementation of optimal reconciliation
        # In practice, this would use the MinT approach or a similar method
        
        # For now, we'll use a simple weighted average of bottom-up and top-down
        bottom_up = self._reconcile_bottom_up(forecasts_dict, hierarchy)
        top_down = self._reconcile_top_down(forecasts_dict, hierarchy)
        
        # Combine with equal weights
        reconciled = {}
        for key in forecasts_dict:
            if key in bottom_up and key in top_down:
                reconciled[key] = 0.5 * bottom_up[key] + 0.5 * top_down[key]
            elif key in bottom_up:
                reconciled[key] = bottom_up[key]
            elif key in top_down:
                reconciled[key] = top_down[key]
        
        return reconciled
    
    def align_multi_frequency_data(
        self,
        dataframes: Dict[str, pd.DataFrame],
        target_frequency: str = 'MS',
        date_col: str = 'date'
    ) -> Dict[str, pd.DataFrame]:
        """
        Align multiple DataFrames with different frequencies to a target frequency.
        
        Args:
            dataframes: Dictionary of DataFrames with potentially different frequencies
            target_frequency: Target frequency for alignment
            date_col: Name of date column
                
        Returns:
            Dictionary of aligned DataFrames
        """
        aligned = {}
        
        for name, df in dataframes.items():
            # Skip empty DataFrames
            if df.empty:
                logger.warning(f"Skipping empty DataFrame: {name}")
                continue
            
            # Ensure date column exists
            if date_col not in df.columns:
                logger.warning(f"DataFrame {name} missing date column '{date_col}'. Skipping.")
                continue
            
            # Create a copy and ensure date column is datetime
            data = df.copy()
            data[date_col] = pd.to_datetime(data[date_col])
            
            # Infer the frequency
            try:
                data = data.set_index(date_col)
                freq = pd.infer_freq(data.index)
                
                if freq is None:
                    # If frequency can't be inferred, try to use the spacing
                    if len(data.index) > 1:
                        timedeltas = pd.Series(data.index[1:]) - pd.Series(data.index[:-1])
                        most_common = timedeltas.mode()[0]
                        
                        if most_common.days % 30 <= 3:
                            freq = 'MS'  # Monthly
                        elif most_common.days % 90 <= 5:
                            freq = 'QS'  # Quarterly
                        elif most_common.days % 365 <= 10:
                            freq = 'AS'  # Annual
                        else:
                            freq = 'D'   # Daily
                    else:
                        logger.warning(f"Cannot infer frequency for {name}. Assuming monthly.")
                        freq = 'MS'
                
                # Align data to target frequency
                if freq == target_frequency:
                    aligned[name] = data.reset_index()
                elif freq in ['QS', 'Q', 'AS', 'A'] and target_frequency in ['MS', 'M']:
                    # Convert from lower to higher frequency
                    disaggregated = self.disaggregate_simple(
                        data, target_frequency=target_frequency, method='linear'
                    )
                    aligned[name] = disaggregated
                elif freq in ['MS', 'M'] and target_frequency in ['QS', 'Q', 'AS', 'A']:
                    # Convert from higher to lower frequency
                    aggregated = self.aggregate_to_lower_frequency(
                        data, target_frequency=target_frequency
                    )
                    aligned[name] = aggregated
                else:
                    # For other conversions, use simple resampling
                    resampled = data.resample(target_frequency).mean()
                    resampled = resampled.reset_index()
                    aligned[name] = resampled
                    
            except Exception as e:
                logger.error(f"Error aligning DataFrame {name}: {e}")
                continue
        
        return aligned