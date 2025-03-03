"""
Utility functions for data operations.
"""
import logging
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config.db_config import DATASETS

logger = logging.getLogger(__name__)

def join_datasets(
    datasets: Dict[str, pd.DataFrame], 
    date_format: str = 'MS',
    fill_method: str = 'ffill',
    align_temporal_window: bool = True,
    target_dataset: Optional[str] = None
) -> pd.DataFrame:
    """
    Join multiple datasets on date, aligning to the target dataset's date range if specified.
    
    Args:
        datasets: Dictionary of DataFrames with dataset names as keys.
        date_format: Frequency of the resulting DataFrame 
                     ('MS' for month start, 'QS' for quarter start).
        fill_method: Method to use for filling missing values 
                     ('ffill' for forward fill, 'bfill' for backward fill,
                      'linear' for linear interpolation, None for no filling).
        align_temporal_window: If True, aligns datasets based on their common time window.
        target_dataset: Optional key in datasets to indicate which dataset should dictate
                        the final date range.
    
    Returns:
        DataFrame with all datasets joined on date and reindexed to the target dataset's
        date range if specified.
    """
    if not datasets:
        return pd.DataFrame()
    
    # Create empty DataFrame to accumulate joins
    result = pd.DataFrame()
    
    for name, df in datasets.items():
        if df.empty:
            logger.warning(f"Dataset {name} is empty, skipping")
            continue
        
        # Retrieve config for dataset (default to 'date' column)
        config = DATASETS.get(name, {})
        date_col = config.get('date_col', 'date')
        
        # Ensure that the date column is in datetime format
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        df_copy.set_index(date_col, inplace=True)
        
        # Rename columns to avoid conflicts
        if 'value_cols' in config:
            value_cols = config['value_cols']
            rename_dict = {col: f"{name}_{col}" for col in value_cols}
            df_copy.rename(columns=rename_dict, inplace=True)
        else:
            value_col = config.get('value_col', name)
            df_copy.rename(columns={value_col: name}, inplace=True)
            df_copy = df_copy[[name]]  # Retain only the target value column
        
        # Join datasets sequentially
        if result.empty:
            result = df_copy
        else:
            result = result.join(df_copy, how='outer')
    
    # Determine date range based on target dataset if specified
    if target_dataset and target_dataset in datasets:
        target_config = DATASETS.get(target_dataset, {})
        target_date_col = target_config.get('date_col', 'date')
        target_df = datasets[target_dataset].copy()
        target_df[target_date_col] = pd.to_datetime(target_df[target_date_col], errors='coerce')
        target_df.set_index(target_date_col, inplace=True)
        
        min_date = target_df.index.min()
        max_date = target_df.index.max()
    else:
        # Fallback: use the overall min and max dates across the joined result
        min_date = result.index.min()
        max_date = result.index.max()
    
    # Create new DatetimeIndex based on the chosen date range
    new_index = pd.date_range(start=min_date, end=max_date, freq=date_format)
    result = result.reindex(new_index)
    
    # Fill missing values according to the specified method
    if fill_method:
        if fill_method in ['ffill', 'bfill']:
            result = result.fillna(method=fill_method)
        elif fill_method == 'linear':
            result = result.interpolate(method='linear')
    
    return result


def resample_to_monthly(
    df: pd.DataFrame, 
    date_col: str = 'date',
    aggregation: str = 'mean'
) -> pd.DataFrame:
    """
    Resample data to monthly frequency.
    
    Args:
        df: DataFrame to resample
        date_col: Name of the date column
        aggregation: Aggregation method ('mean', 'sum', 'last', etc.)
    
    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df
    
    # Make copy to avoid modifying original
    df_copy = df.copy()
    
    # Set date as index if it's not already
    if date_col in df_copy.columns:
        df_copy.set_index(date_col, inplace=True)
    
    # Resample to month start frequency
    df_resampled = df_copy.resample('MS').agg(aggregation)
    
    # Reset index to get date as a column again
    df_resampled.reset_index(inplace=True)
    df_resampled.rename(columns={'index': date_col}, inplace=True)
    
    return df_resampled

def resample_to_quarterly(
    df: pd.DataFrame, 
    date_col: str = 'date',
    aggregation: str = 'mean'
) -> pd.DataFrame:
    """
    Resample data to quarterly frequency.
    
    Args:
        df: DataFrame to resample
        date_col: Name of the date column
        aggregation: Aggregation method ('mean', 'sum', 'last', etc.)
    
    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df
    
    # Make copy to avoid modifying original
    df_copy = df.copy()
    
    # Set date as index if it's not already
    if date_col in df_copy.columns:
        df_copy.set_index(date_col, inplace=True)
    
    # Resample to quarter start frequency
    df_resampled = df_copy.resample('QS').agg(aggregation)
    
    # Reset index to get date as a column again
    df_resampled.reset_index(inplace=True)
    df_resampled.rename(columns={'index': date_col}, inplace=True)
    
    return df_resampled

def calculate_percent_change(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    periods: int = 1
) -> pd.DataFrame:
    """
    Calculate percent change for specified columns.
    
    Args:
        df: DataFrame with data
        cols: List of columns to calculate percent change for (None means all numeric columns)
        periods: Number of periods to shift for calculation
    
    Returns:
        DataFrame with percent change columns added
    """
    if df.empty:
        return df
    
    result = df.copy()
    
    # If no columns specified, use all numeric columns
    if cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cols = [col for col in numeric_cols if col != 'date']
    
    # Calculate percent change
    for col in cols:
        if col in result.columns:
            result[f'{col}_pct_change'] = result[col].pct_change(periods=periods) * 100
    
    return result

def calculate_rolling_statistics(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    window: int = 3,
    statistics: List[str] = ['mean', 'std']
) -> pd.DataFrame:
    """
    Calculate rolling statistics for specified columns without causing DataFrame fragmentation.
    
    Args:
        df: DataFrame with data.
        cols: List of columns to calculate statistics for (if None, all numeric columns except 'date').
        window: Rolling window size.
        statistics: List of statistics to calculate ('mean', 'std', 'min', 'max', etc.).
    
    Returns:
        DataFrame with rolling statistic columns appended.
    """
    if df.empty:
        return df
    
    result = df.copy()
    new_columns = {}  # Dictionary to store new rolling statistic columns
    
    # If no columns specified, use all numeric columns (excluding 'date')
    if cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cols = [col for col in numeric_cols if col != 'date']
    
    # Calculate rolling statistics and store them in new_columns
    for col in cols:
        if col in result.columns:
            roll_obj = result[col].rolling(window=window)
            for stat in statistics:
                if stat == 'mean':
                    new_columns[f'{col}_rolling_mean'] = roll_obj.mean()
                elif stat == 'std':
                    new_columns[f'{col}_rolling_std'] = roll_obj.std()
                elif stat == 'min':
                    new_columns[f'{col}_rolling_min'] = roll_obj.min()
                elif stat == 'max':
                    new_columns[f'{col}_rolling_max'] = roll_obj.max()
    
    # Create a DataFrame from the new columns and concatenate once with the original data
    new_cols_df = pd.DataFrame(new_columns, index=result.index)
    final_result = pd.concat([result, new_cols_df], axis=1)
    
    # Optionally defragment by creating a copy of the DataFrame
    final_result = final_result.copy()
    
    return final_result

def create_lagged_features(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    lags: List[int] = [1]  # Adjust default as needed
) -> pd.DataFrame:
    """
    Create lagged features for time series analysis without causing DataFrame fragmentation.
    
    Args:
        df: DataFrame with data.
        cols: List of columns to create lags for (if None, all numeric columns except 'date').
        lags: List of lag periods to create.
        
    Returns:
        DataFrame with lagged columns added.
    """
    if df.empty:
        return df

    result = df.copy()
    new_columns = {}  # Dictionary to accumulate new lagged features

    # If no columns specified, use all numeric columns (excluding 'date' if present)
    if cols is None:
        numeric_cols = result.select_dtypes(include=['number']).columns.tolist()
        cols = [col for col in numeric_cols if col != 'date']
    
    # Compute lagged features and store in dictionary
    for col in cols:
        if col in result.columns:
            for lag in lags:
                new_col_name = f'{col}_lag_{lag}'
                new_columns[new_col_name] = result[col].shift(lag)
    
    # Create a DataFrame from all new columns
    lagged_df = pd.DataFrame(new_columns, index=result.index)
    
    # Concatenate the original DataFrame with the new lagged features all at once
    final_df = pd.concat([result, lagged_df], axis=1)
    
    # Defragment the DataFrame by copying it
    final_df = final_df.copy()
    
    return final_df

def detect_outliers(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    method: str = 'zscore',
    threshold: float = 3.0
) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    """
    Detect outliers in the data.
    
    Args:
        df: DataFrame with data
        cols: List of columns to check for outliers (None means all numeric columns)
        method: Detection method ('zscore' or 'iqr')
        threshold: Threshold for outlier detection
    
    Returns:
        Tuple of (DataFrame with outlier flags, Dictionary with outlier indices by column)
    """
    if df.empty:
        return df, {}
    
    result = df.copy()
    outliers = {}
    
    # If no columns specified, use all numeric columns
    if cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cols = [col for col in numeric_cols if col != 'date']
    
    for col in cols:
        if col in result.columns:
            if method == 'zscore':
                # Z-score method
                z_scores = np.abs((result[col] - result[col].mean()) / result[col].std())
                is_outlier = z_scores > threshold
                result[f'{col}_is_outlier'] = is_outlier
                outliers[col] = result.index[is_outlier].tolist()
            
            elif method == 'iqr':
                # IQR method
                q1 = result[col].quantile(0.25)
                q3 = result[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (threshold * iqr)
                upper_bound = q3 + (threshold * iqr)
                is_outlier = (result[col] < lower_bound) | (result[col] > upper_bound)
                result[f'{col}_is_outlier'] = is_outlier
                outliers[col] = result.index[is_outlier].tolist()
    
    return result, outliers

def fill_missing_values(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    method: str = 'interpolate',
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Fill missing values in the data.
    
    Args:
        df: DataFrame with data
        cols: List of columns to fill (None means all columns)
        method: Filling method ('interpolate', 'ffill', 'bfill', 'mean', 'median', 'mode')
        limit: Maximum number of consecutive NaN values to fill
    
    Returns:
        DataFrame with filled values
    """
    if df.empty:
        return df
    
    result = df.copy()
    
    # If no columns specified, use all columns
    if cols is None:
        cols = df.columns.tolist()
    
    for col in cols:
        if col in result.columns:
            if method == 'interpolate':
                result[col] = result[col].interpolate(limit=limit)
            elif method == 'ffill':
                result[col] = result[col].fillna(method='ffill', limit=limit)
            elif method == 'bfill':
                result[col] = result[col].fillna(method='bfill', limit=limit)
            elif method == 'mean':
                result[col] = result[col].fillna(result[col].mean())
            elif method == 'median':
                result[col] = result[col].fillna(result[col].median())
            elif method == 'mode':
                result[col] = result[col].fillna(result[col].mode()[0])
    
    return result