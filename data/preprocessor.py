"""
Data preprocessing module for preparing datasets for model training.
"""
import logging
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from config.db_config import DATASETS
from data.connectors import DataConnector
from data.data_utils import (
    join_datasets,
    calculate_percent_change,
    calculate_rolling_statistics,
    create_lagged_features,
    detect_outliers,
    fill_missing_values
)

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Class for preprocessing data for model training.
    """
    
    def __init__(self, data_connector: Optional[DataConnector] = None):
        """
        Initialize with an optional data connector.
        
        Args:
            data_connector: Optional DataConnector instance for fetching data
        """
        self.data_connector = data_connector or DataConnector()
        self.scalers = {}  # Store scalers for inverse transformation

    def prepare_dataset(
            self,
            target_dataset: str,
            feature_datasets: Optional[List[str]] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            frequency: str = 'monthly',
            create_lags: bool = True,
            lag_periods: List[int] = [1],
            calculate_changes: bool = True,
            calculate_rolling: bool = True,
            rolling_window: int = 3,
            fill_na_method: str = 'interpolate',
            handle_outliers: bool = True,
            scale_data: bool = True,
            scaler_type: str = 'standard',
            align_temporal_window: bool = True
        ) -> Tuple[pd.DataFrame, Dict[str, any]]:
            """
            Prepare a dataset for model training by combining target and feature datasets.
            
            Args:
                target_dataset: Name of the target variable dataset
                feature_datasets: List of feature datasets to include
                start_date: Start date for data (defaults to earliest available)
                end_date: End date for data (defaults to latest available)
                frequency: Data frequency ('monthly' or 'quarterly')
                create_lags: Whether to create lagged features
                lag_periods: List of lag periods to create
                calculate_changes: Whether to calculate percent changes
                calculate_rolling: Whether to calculate rolling statistics
                rolling_window: Window size for rolling calculations
                fill_na_method: Method for filling missing values
                handle_outliers: Whether to detect and handle outliers
                scale_data: Whether to scale the data
                scaler_type: Type of scaler to use ('standard', 'minmax', 'robust')
                align_temporal_window: Whether to align datasets to their common time window.
                                    When True, only the time period where all datasets have 
                                    data is used, ensuring temporal consistency.
                
            Returns:
                Tuple of (DataFrame with prepared data, Dict of metadata including scalers)
            """
            # Ensure feature_datasets is a list
            if feature_datasets is None:
                feature_datasets = []
            
            # Get dataset configurations
            target_config = DATASETS.get(target_dataset)
            if not target_config:
                raise ValueError(f"Unknown target dataset: {target_dataset}")
            
            # Determine target value column
            if 'value_cols' in target_config:
                target_value_col = target_config['value_cols'][0]  # Use first value column as target
            else:
                target_value_col = target_config['value_col']
            
            # Fetch datasets
            logger.info(f"Fetching target dataset: {target_dataset}")
            target_df = self.data_connector.fetch_dataset(target_dataset, start_date, end_date)
            
            feature_dfs = {}
            for feature in feature_datasets:
                logger.info(f"Fetching feature dataset: {feature}")
                feature_dfs[feature] = self.data_connector.fetch_dataset(feature, start_date, end_date)
            
            # Record original data sizes for alignment warnings
            original_sizes = {name: len(df) for name, df in {target_dataset: target_df, **feature_dfs}.items() if df is not None}
            
            # Join datasets
            logger.info("Joining datasets")
            data_format = 'MS' if frequency == 'monthly' else 'QS'
            
            # Include target dataset in the join
            all_dfs = {target_dataset: target_df, **feature_dfs}
            df = join_datasets(
                all_dfs,
                date_format=data_format,
                fill_method='ffill',
                align_temporal_window=align_temporal_window,
                target_dataset=target_dataset
            )
            
            if df.empty:
                logger.warning("No data available after joining datasets")
                return pd.DataFrame(), {}
            
            # Check for significant data truncation if alignment was used
            if align_temporal_window:
                for name, original_size in original_sizes.items():
                    # Calculate how many points we'd expect after reindexing to the frequency
                    expected_size = len(df.index)
                    truncation_percent = (original_size - expected_size) / original_size * 100
                    if truncation_percent > 20:  # More than 20% data truncated
                        logger.warning(f"Significant data truncation for {name}: {truncation_percent:.1f}% of records removed during temporal alignment")
                    elif truncation_percent > 5:  # 5-20% data truncated
                        logger.info(f"Moderate data truncation for {name}: {truncation_percent:.1f}% of records removed during temporal alignment")
            
            # Make sure we have the target column with correct name
            if target_dataset not in df.columns:
                # Try to find the target column using value_col information
                target_column = None
                if 'value_cols' in target_config:
                    # For multi-column targets, we need to choose one or create a composite
                    target_columns = [f"{target_dataset}_{col}" for col in target_config['value_cols']]
                    for col in target_columns:
                        if col in df.columns:
                            target_column = col
                            break
                else:
                    # For single column targets
                    target_column = f"{target_dataset}_{target_value_col}"
                    if target_column in df.columns:
                        # Rename to match target dataset name for simplicity
                        df.rename(columns={target_column: target_dataset}, inplace=True)
                    
                if not target_column and target_dataset not in df.columns:
                    logger.error(f"Target column not found for {target_dataset}")
                    return pd.DataFrame(), {}
            
            # Initial preprocessing
            logger.info("Performing initial preprocessing")
            
            # Fill missing values
            df = fill_missing_values(df, method=fill_na_method)
            
            # Handle outliers if requested
            outliers_info = {}
            if handle_outliers:
                logger.info("Detecting and handling outliers")
                df_with_outliers, outliers = detect_outliers(df, method='zscore', threshold=3.0)
                
                # Store outlier information
                outliers_info = outliers
                
                # Replace outliers with NaN and then interpolate
                for col, indices in outliers.items():
                    if indices:  # If there are outliers
                        logger.info(f"Found {len(indices)} outliers in {col}")
                        df.loc[indices, col] = np.nan
                
                # Fill outliers
                df = fill_missing_values(df, method='interpolate')
            
            # Feature engineering
            logger.info("Performing feature engineering")
            
            # Calculate percent changes
            if calculate_changes:
                logger.info("Calculating percent changes")
                df = calculate_percent_change(df, periods=1)
            
            # Calculate rolling statistics
            if calculate_rolling:
                logger.info(f"Calculating rolling statistics with window={rolling_window}")
                df = calculate_rolling_statistics(df, window=rolling_window)
            
            # Create lagged features
            if create_lags:
                logger.info(f"Creating lagged features with periods={lag_periods}")
                df = create_lagged_features(df, lags=lag_periods)
            
            # Log the shape before dropping NaN values
            logger.info(f"Shape before dropping NaN values: {df.shape}")
            
            # Log columns with the most missing values
            missing_counts = df.isnull().sum()
            top_missing = missing_counts[missing_counts > 0].sort_values(ascending=False).head(10)
            if not top_missing.empty:
                logger.info(f"Top columns with missing values:")
                for col, count in top_missing.items():
                    logger.info(f"  {col}: {count} missing values ({count/len(df)*100:.1f}%)")
            
            # Drop rows with NaN values after feature engineering
            original_len = len(df)
            df_before_drop = df.copy()
            df = df.dropna()
            dropped_count = original_len - len(df)
            logger.info(f"Dropped {dropped_count} rows with missing values ({dropped_count/original_len*100:.1f}% of data)")
            
            # If significant data was dropped, provide more information
            if dropped_count > 0:
                # Find which rows were dropped
                dropped_mask = ~df_before_drop.index.isin(df.index)
                dropped_rows = df_before_drop[dropped_mask]
                
                # Check which features caused the most drops
                if not dropped_rows.empty:
                    # Get the count of NaN values by column in the dropped rows
                    nan_counts = dropped_rows.isnull().sum().sort_values(ascending=False)
                    most_problematic = nan_counts[nan_counts > 0].head(5)
                    
                    logger.info(f"Most problematic columns causing row drops:")
                    for col, count in most_problematic.items():
                        logger.info(f"  {col}: {count} NaNs in dropped rows")
                    
                    # Analyze date ranges of dropped data
                    if hasattr(dropped_rows.index, 'min') and hasattr(dropped_rows.index, 'max'):
                        logger.info(f"Date range of dropped data: {dropped_rows.index.min()} to {dropped_rows.index.max()}")
                        
                        # Check if drops are concentrated in certain periods
                        if dropped_count > 5:
                            # Group by year-month if datetime index
                            if hasattr(dropped_rows.index, 'year'):
                                dropped_by_period = dropped_rows.groupby([dropped_rows.index.year, dropped_rows.index.month]).size()
                                logger.info(f"Drops by period (year-month): {dropped_by_period.to_dict()}")
            
            # Scale features if requested
            scalers = {}
            if scale_data:
                logger.info(f"Scaling data using {scaler_type} scaler")
                
                # Get numeric columns excluding the target
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                feature_cols = [col for col in numeric_cols if col != target_dataset]
                
                # Select scaler based on type
                if scaler_type == 'standard':
                    scaler = StandardScaler()
                elif scaler_type == 'minmax':
                    scaler = MinMaxScaler()
                elif scaler_type == 'robust':
                    scaler = RobustScaler()
                else:
                    logger.warning(f"Unknown scaler type: {scaler_type}, using StandardScaler")
                    scaler = StandardScaler()
                
                # Scale features
                df[feature_cols] = scaler.fit_transform(df[feature_cols])
                
                # Store scaler for future inverse transformations
                scalers['features'] = scaler
                
                # Scale target separately
                if target_dataset in df.columns:
                    target_scaler = StandardScaler()
                    df[[target_dataset]] = target_scaler.fit_transform(df[[target_dataset]])
                    scalers['target'] = target_scaler
            
            # Generate metadata
            metadata = {
                'target_dataset': target_dataset,
                'feature_datasets': feature_datasets,
                'frequency': frequency,
                'n_rows': len(df),
                'n_features': len(df.columns) - 1,  # Exclude target
                'feature_names': [col for col in df.columns if col != target_dataset],
                'date_range': (df.index.min(), df.index.max()) if not df.empty else (None, None),
                'outliers': outliers_info,
                'scalers': scalers,
                'temporal_alignment': {
                    'aligned': align_temporal_window,
                    'original_sizes': original_sizes,
                    'aligned_size': len(df),
                    'truncation_percentages': {
                        name: ((original_size - len(df)) / original_size * 100) 
                        for name, original_size in original_sizes.items() if original_size > 0
                    }
                }
            }
            
            # Store scalers for later use
            self.scalers.update(scalers)
            
            logger.info(f"Dataset preparation complete. Shape: {df.shape}")
            return df, metadata
    
    def split_train_test(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        test_size: float = 0.2,
        validation_size: float = 0.0,
        time_based: bool = True
    ) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        Split the dataset into training and testing sets.
        
        Args:
            df: Dataframe with prepared data
            target_col: Name of target column
            test_size: Proportion of data to use for testing
            validation_size: Proportion of data to use for validation
            time_based: Whether to split based on time (True) or randomly (False)
            
        Returns:
            Dictionary with X_train, y_train, X_test, y_test (and X_val, y_val if validation_size > 0)
        """
        if df.empty:
            logger.warning("Empty dataframe provided for train-test split")
            return {}
        
        features = [col for col in df.columns if col != target_col]
        X = df[features]
        y = df[target_col]
        
        result = {}
        
        if time_based:
            # Time-based split (respects temporal order)
            train_size = 1.0 - test_size - validation_size
            
            n = len(df)
            train_end = int(n * train_size)
            
            if validation_size > 0:
                val_end = int(n * (train_size + validation_size))
                
                result['X_train'] = X.iloc[:train_end]
                result['y_train'] = y.iloc[:train_end]
                result['X_val'] = X.iloc[train_end:val_end]
                result['y_val'] = y.iloc[train_end:val_end]
                result['X_test'] = X.iloc[val_end:]
                result['y_test'] = y.iloc[val_end:]
            else:
                result['X_train'] = X.iloc[:train_end]
                result['y_train'] = y.iloc[:train_end]
                result['X_test'] = X.iloc[train_end:]
                result['y_test'] = y.iloc[train_end:]
        else:
            # Random split
            from sklearn.model_selection import train_test_split
            
            if validation_size > 0:
                # First split into training and temp (test + validation)
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=test_size + validation_size, random_state=42
                )
                
                # Then split temp into validation and test
                val_ratio = validation_size / (test_size + validation_size)
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=1-val_ratio, random_state=42
                )
                
                result['X_train'] = X_train
                result['y_train'] = y_train
                result['X_val'] = X_val
                result['y_val'] = y_val
                result['X_test'] = X_test
                result['y_test'] = y_test
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                result['X_train'] = X_train
                result['y_train'] = y_train
                result['X_test'] = X_test
                result['y_test'] = y_test
        
        # Log split information
        logger.info(f"Data split: train={len(result['X_train'])}, " + 
                   (f"validation={len(result['X_val'])}, " if validation_size > 0 else "") + 
                   f"test={len(result['X_test'])}")
        
        return result
    
    def inverse_transform_predictions(
        self, 
        predictions: np.ndarray, 
        target_dataset: str
    ) -> np.ndarray:
        """
        Inverse transform scaled predictions back to original scale.
        
        Args:
            predictions: Array of predictions
            target_dataset: Name of target dataset
            
        Returns:
            Array of predictions in original scale
        """
        if 'target' not in self.scalers:
            logger.warning("No target scaler found, returning predictions as is")
            return predictions
        
        try:
            # Reshape for inverse transform if needed
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(-1, 1)
            
            return self.scalers['target'].inverse_transform(predictions)
        except Exception as e:
            logger.error(f"Error inverse transforming predictions: {e}")
            return predictions