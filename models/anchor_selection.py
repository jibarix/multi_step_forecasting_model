"""
Module for dynamic selection of anchor variables in econometric forecasting.

This module implements methods for selecting the most suitable anchor variables
based on statistical tests, information criteria, and machine learning feature importance.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# Set up logging
logger = logging.getLogger(__name__)

class AnchorVariableSelector:
    """
    Class for dynamic selection of anchor variables in econometric models.
    
    This implements various methods to identify the most suitable macro-economic 
    indicators to serve as anchor variables for forecasting, including Granger causality,
    information criteria, and feature importance from machine learning models.
    """
    
    def __init__(self):
        """Initialize the AnchorVariableSelector."""
        self.granger_results = {}
        self.correlation_results = {}
        self.information_criteria = {}
        self.forecasting_metrics = {}
        self.feature_importance = {}
        
    def granger_causality_test(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        max_lag: int = 12, 
        significance: float = 0.05
    ) -> Dict[str, Dict[str, Any]]:
        """
        Test Granger causality between potential anchor variables and target.
        
        Args:
            df: DataFrame with time series data
            target_col: Name of target column
            max_lag: Maximum number of lags to test
            significance: Significance level for causality determination
            
        Returns:
            Dictionary with Granger causality results for each variable
        """
        results = {}
        target = df[target_col]
        
        for column in df.columns:
            if column != target_col:
                data = pd.concat([target, df[column]], axis=1)
                # Handle NaN values
                data = data.dropna()
                
                if len(data) <= max_lag + 1:
                    logger.warning(f"Not enough observations for Granger test on {column}. Skipping.")
                    continue
                    
                try:
                    # Perform Granger causality test
                    test_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                    
                    # Extract p-values for each lag
                    p_values = [test_result[i+1][0]['ssr_chi2test'][1] for i in range(max_lag)]
                    
                    # Check if any lag shows causality at the specified significance level
                    causes_target = any(p < significance for p in p_values)
                    
                    # Find the best lag (lowest p-value)
                    best_lag = np.argmin(p_values) + 1
                    
                    results[column] = {
                        'causes_target': causes_target,
                        'best_lag': best_lag,
                        'min_p_value': min(p_values),
                        'p_values': p_values
                    }
                    
                    logger.debug(f"Granger test for {column}: causes_target={causes_target}, "
                                f"best_lag={best_lag}, min_p_value={min(p_values):.4f}")
                    
                except Exception as e:
                    logger.warning(f"Error in Granger causality test for {column}: {e}")
                    continue
        
        self.granger_results = results
        return results
    
    def correlation_analysis(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        max_lag: int = 12
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze correlation between potential anchor variables and target.
        
        Args:
            df: DataFrame with time series data
            target_col: Name of target column
            max_lag: Maximum number of lags to test
            
        Returns:
            Dictionary with correlation results for each variable
        """
        results = {}
        target = df[target_col]
        
        for column in df.columns:
            if column != target_col:
                variable = df[column]
                corr_by_lag = {}
                
                # Calculate correlation at different lags
                for lag in range(1, max_lag + 1):
                    lagged_var = variable.shift(lag)
                    # Skip if not enough data after shift
                    if lagged_var.isna().sum() >= len(lagged_var) * 0.5:
                        continue
                        
                    # Calculate correlation and handle potential NaN values
                    corr = target.corr(lagged_var, method='pearson')
                    if not np.isnan(corr):
                        corr_by_lag[lag] = corr
                
                if corr_by_lag:
                    # Find lag with strongest correlation (absolute value)
                    best_lag = max(corr_by_lag.items(), key=lambda x: abs(x[1]))[0]
                    max_corr = corr_by_lag[best_lag]
                    
                    results[column] = {
                        'max_correlation': max_corr,
                        'best_lag': best_lag,
                        'correlation_by_lag': corr_by_lag
                    }
                    
                    logger.debug(f"Correlation analysis for {column}: max_corr={max_corr:.4f}, "
                                f"best_lag={best_lag}")
        
        self.correlation_results = results
        return results
    
    def evaluate_var_model(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        anchor_variables: List[str], 
        max_lag: int = 12
    ) -> Dict[str, float]:
        """
        Evaluate a VAR model using the target and potential anchor variables.
        
        Args:
            df: DataFrame with time series data
            target_col: Name of target column
            anchor_variables: List of potential anchor variable columns
            max_lag: Maximum lag to consider for VAR model
            
        Returns:
            Dictionary with information criteria and other metrics
        """
        # Create a DataFrame with target and anchor variables
        model_df = df[[target_col] + anchor_variables].copy()
        model_df = model_df.dropna()
        
        if len(model_df) <= max_lag + len(anchor_variables) + 1:
            logger.warning("Not enough observations for VAR model evaluation.")
            return {}
        
        try:
            # Fit VAR model
            var_model = VAR(model_df)
            
            # Find optimal lag using information criteria
            lag_order_results = var_model.select_order(maxlags=max_lag)
            optimal_lags = {
                'aic': lag_order_results.aic,
                'bic': lag_order_results.bic, 
                'hqic': lag_order_results.hqic,
                'fpe': lag_order_results.fpe
            }
            
            # Get the lag with lowest AIC
            optimal_lag = lag_order_results.aic
            
            # Fit the model with optimal lag
            fitted_model = var_model.fit(maxlags=optimal_lag)
            
            # Get information criteria
            ic_values = {
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'hqic': fitted_model.hqic,
                'fpe': fitted_model.fpe
            }
            
            return {
                'information_criteria': ic_values,
                'optimal_lags': optimal_lags,
                'optimal_lag_aic': optimal_lag
            }
            
        except Exception as e:
            logger.warning(f"Error in VAR model evaluation: {e}")
            return {}
    
    def evaluate_forecasting_performance(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        anchor_variables: List[str], 
        forecast_horizon: int = 12, 
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Evaluate forecasting performance using the anchor variables.
        
        Args:
            df: DataFrame with time series data
            target_col: Name of target column
            anchor_variables: List of potential anchor variable columns
            forecast_horizon: Number of steps to forecast
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with forecasting performance metrics
        """
        # Create a DataFrame with target and anchor variables
        model_df = df[[target_col] + anchor_variables].copy()
        model_df = model_df.dropna()
        
        if len(model_df) <= forecast_horizon * 2:
            logger.warning("Not enough observations for forecasting evaluation.")
            return {}
        
        try:
            # Split data into train and test sets
            train_size = int(len(model_df) * (1 - test_size))
            train_df = model_df.iloc[:train_size]
            test_df = model_df.iloc[train_size:]
            
            # Fit VAR model on training data
            var_model = VAR(train_df)
            
            # Find optimal lag using AIC
            lag_order_results = var_model.select_order(maxlags=min(12, train_size // 2))
            optimal_lag = lag_order_results.aic
            
            # Fit the model with optimal lag
            fitted_model = var_model.fit(maxlags=optimal_lag)
            
            # Generate out-of-sample forecasts
            forecasts = fitted_model.forecast(train_df.values[-optimal_lag:], steps=len(test_df))
            
            # Extract forecasts for target variable (first column in VAR)
            target_idx = model_df.columns.get_loc(target_col)
            target_forecasts = forecasts[:, target_idx]
            
            # Calculate error metrics
            actual = test_df[target_col].values
            rmse = np.sqrt(mean_squared_error(actual, target_forecasts[:len(actual)]))
            mae = mean_absolute_error(actual, target_forecasts[:len(actual)])
            
            return {
                'rmse': rmse,
                'mae': mae,
                'optimal_lag': optimal_lag
            }
            
        except Exception as e:
            logger.warning(f"Error in forecasting evaluation: {e}")
            return {}
    
    def select_best_anchors(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        n_anchors: int = 3, 
        methods: List[str] = ['granger', 'correlation', 'var', 'forecasting']
    ) -> Dict[str, List[str]]:
        """
        Select the best anchor variables using multiple methods.
        
        Args:
            df: DataFrame with time series data
            target_col: Name of target column
            n_anchors: Number of anchor variables to select
            methods: List of methods to use for selection
                ('granger', 'correlation', 'var', 'forecasting', 'ensemble')
            
        Returns:
            Dictionary with selected anchor variables for each method and ensemble
        """
        all_results = {}
        
        # Run Granger causality tests if requested
        if 'granger' in methods and not self.granger_results:
            self.granger_causality_test(df, target_col)
            
        # Run correlation analysis if requested
        if 'correlation' in methods and not self.correlation_results:
            self.correlation_analysis(df, target_col)
            
        # Collect candidates from Granger causality
        if 'granger' in methods and self.granger_results:
            granger_candidates = [col for col, result in self.granger_results.items() 
                                if result['causes_target']]
            # Sort by p-value (lowest first)
            granger_candidates.sort(key=lambda x: self.granger_results[x]['min_p_value'])
            all_results['granger'] = granger_candidates[:n_anchors]
        
        # Collect candidates from correlation analysis
        if 'correlation' in methods and self.correlation_results:
            corr_candidates = list(self.correlation_results.keys())
            # Sort by absolute correlation (highest first)
            corr_candidates.sort(key=lambda x: abs(self.correlation_results[x]['max_correlation']), 
                                reverse=True)
            all_results['correlation'] = corr_candidates[:n_anchors]
        
        # Create ensemble selection (variables that appear in multiple methods)
        if len(methods) > 1 and len(all_results) > 1:
            # Count occurrences of each variable across methods
            var_counts = {}
            for method, vars_list in all_results.items():
                for var in vars_list:
                    var_counts[var] = var_counts.get(var, 0) + 1
            
            # Sort by count and then by correlation strength
            ensemble_candidates = list(var_counts.keys())
            ensemble_candidates.sort(
                key=lambda x: (
                    var_counts[x],  # First by count
                    abs(self.correlation_results.get(x, {}).get('max_correlation', 0))  # Then by correlation
                ),
                reverse=True
            )
            all_results['ensemble'] = ensemble_candidates[:n_anchors]
        
        return all_results
    
    def evaluate_candidate_combinations(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        candidate_anchors: List[str], 
        max_combination_size: int = 3
    ) -> Dict[Tuple[str, ...], Dict[str, float]]:
        """
        Evaluate different combinations of anchor variables.
        
        Args:
            df: DataFrame with time series data
            target_col: Name of target column
            candidate_anchors: List of candidate anchor variables
            max_combination_size: Maximum number of variables in a combination
            
        Returns:
            Dictionary with evaluation results for each combination
        """
        import itertools
        
        results = {}
        
        # Generate combinations of different sizes
        for size in range(1, min(max_combination_size + 1, len(candidate_anchors) + 1)):
            for combo in itertools.combinations(candidate_anchors, size):
                combo_vars = list(combo)
                
                # Evaluate forecasting performance
                forecast_metrics = self.evaluate_forecasting_performance(
                    df, target_col, combo_vars, forecast_horizon=12
                )
                
                if forecast_metrics:
                    results[combo] = forecast_metrics
        
        # Sort combinations by RMSE (ascending)
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['rmse']))
        
        return sorted_results
    
    def get_optimal_anchor_combination(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        max_anchors: int = 3
    ) -> List[str]:
        """
        Get the optimal combination of anchor variables through a comprehensive evaluation.
        
        Args:
            df: DataFrame with time series data
            target_col: Name of target column
            max_anchors: Maximum number of anchor variables to select
            
        Returns:
            List of optimal anchor variables
        """
        # Step 1: Run preliminary selection to get candidate anchors
        selection_results = self.select_best_anchors(
            df, target_col, n_anchors=max_anchors*2,
            methods=['granger', 'correlation']
        )
        
        # Combine candidates from different methods
        candidates = set()
        for method, vars_list in selection_results.items():
            candidates.update(vars_list)
        
        # Step 2: Evaluate different combinations of candidates
        if len(candidates) <= max_anchors:
            # If we have few candidates, just return them all
            return list(candidates)
        
        # Evaluate combinations of different sizes
        combination_results = self.evaluate_candidate_combinations(
            df, target_col, list(candidates), max_combination_size=max_anchors
        )
        
        # Get the best combination
        if combination_results:
            best_combo = next(iter(combination_results))
            return list(best_combo)
        else:
            # Fallback to ensemble selection if combination evaluation fails
            if 'ensemble' in selection_results:
                return selection_results['ensemble']
            elif 'correlation' in selection_results:
                return selection_results['correlation']
            else:
                return list(candidates)[:max_anchors]