#!/usr/bin/env python
"""
Enhanced main script for auto sales forecasting model.

This script provides a complete pipeline for:
1. Data acquisition from databases
2. Data preprocessing and feature engineering
3. Model training and evaluation
4. Forecasting future values
5. Visualization and reporting
6. Model comparison and dashboard generation

Usage:
    python main.py [--target TARGET] [--features FEATURES [FEATURES ...]]
                  [--start_date START_DATE] [--end_date END_DATE]
                  [--forecast_horizon FORECAST_HORIZON]
                  [--model_types MODEL_TYPES [MODEL_TYPES ...]]
                  [--output_dir OUTPUT_DIR]
                  [--mode {train,forecast,evaluate,compare,dashboard,all}]

Example:
    python main.py --target auto_sales --features unemployment_rate gas_price
                  --start_date 2015-01-01 --forecast_horizon 12
                  --model_types linear_regression random_forest prophet
                  --mode compare
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple, Any

# Import project modules
from config import DATASETS, DEFAULT_START_DATE, DEFAULT_END_DATE
from config.model_config import (
    get_model_config, LINEAR_MODELS, TIME_SERIES_MODELS, 
    ENSEMBLE_MODELS, NEURAL_NETWORK_MODELS
)
from data import DataConnector, DataPreprocessor
from models import get_model, ModelEvaluator
from visualizations import (
    plot_time_series,
    plot_correlation_matrix,
    plot_forecast,
    plot_feature_importance,
    plot_model_comparison,
    plot_residuals,
    create_dashboard
)

# Set up logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Add after importing logging
import logging
import cmdstanpy
logging.getLogger('cmdstanpy').setLevel(logging.INFO)
logging.getLogger('prophet').setLevel(logging.INFO)

# Set environment variable 
import os
os.environ['CMDSTAN_VERBOSE'] = 'TRUE'

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Auto Sales Forecasting Tool')
    
    parser.add_argument('--target', type=str, default='auto_sales',
                        help='Target dataset to forecast (default: auto_sales)')
    
    parser.add_argument('--features', type=str, nargs='+',
                        help='Feature datasets to use for prediction (if empty, all datasets are considered)')
    
    parser.add_argument('--start_date', type=str, default=DEFAULT_START_DATE,
                        help=f'Start date for data (default: {DEFAULT_START_DATE})')
    
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date for data (default: latest available)')
    
    parser.add_argument('--forecast_horizon', type=int, default=12,
                        help='Number of periods to forecast (default: 12)')
    
    parser.add_argument('--model_types', type=str, nargs='+', 
                        default=['linear_regression'],
                        help='Types of models to use (default: linear_regression)')
    
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs (default: outputs)')
    
    parser.add_argument('--mode', type=str, 
                        choices=['train', 'forecast', 'evaluate', 'compare', 'dashboard', 'all'], 
                        default='all',
                        help='Mode of operation (default: all)')
    
    parser.add_argument('--feature_lag', type=int, default=1,
                        help='Number of months to lag features (default: 1)')
    
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing (default: 0.2)')
    
    parser.add_argument('--scale_data', action='store_true',
                        help='Whether to scale the data')
    
    parser.add_argument('--save_model', action='store_true',
                        help='Whether to save the trained model')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()


def prepare_output_directory(output_dir: str):
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Path to output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Create subdirectories
    plots_dir = os.path.join(output_dir, 'plots')
    models_dir = os.path.join(output_dir, 'models')
    data_dir = os.path.join(output_dir, 'data')
    dashboard_dir = os.path.join(output_dir, 'dashboard')
    
    for directory in [plots_dir, models_dir, data_dir, dashboard_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)


def load_datasets(
    connector: DataConnector,
    target_dataset: str,
    feature_datasets: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load target and feature datasets from the database.
    
    Args:
        connector: DataConnector instance
        target_dataset: Name of target dataset
        feature_datasets: List of feature dataset names
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        Dictionary with dataset names as keys and DataFrames as values
    """
    datasets = {}
    
    # Load target dataset
    logger.info(f"Loading target dataset: {target_dataset}")
    target_df = connector.fetch_dataset(target_dataset, start_date, end_date)
    datasets[target_dataset] = target_df
    
    # Load feature datasets if specified
    if feature_datasets:
        for dataset in feature_datasets:
            if dataset != target_dataset:  # Avoid duplicate loading
                logger.info(f"Loading feature dataset: {dataset}")
                feature_df = connector.fetch_dataset(dataset, start_date, end_date)
                datasets[dataset] = feature_df
    else:
        # If no features specified, load all available datasets except target
        logger.info("No feature datasets specified, attempting to load all datasets")
        for dataset in DATASETS.keys():
            if dataset != target_dataset:  # Skip target dataset
                logger.info(f"Loading feature dataset: {dataset}")
                try:
                    feature_df = connector.fetch_dataset(dataset, start_date, end_date)
                    datasets[dataset] = feature_df
                except Exception as e:
                    logger.warning(f"Failed to load dataset {dataset}: {e}")
    
    return datasets


def train_model(
    prepared_data: pd.DataFrame,
    target_col: str,
    model_type: str,
    test_size: float = 0.2,
    output_dir: Optional[str] = None,
    save_model: bool = False,
    verbose: bool = False
) -> Tuple[Union[object, Dict[str, object]], Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Train a forecasting model on prepared data.
    
    Args:
        prepared_data: DataFrame with prepared data.
        target_col: Name of target column.
        model_type: Type of model to train.
        test_size: Fraction of data to use for testing.
        output_dir: Directory to save outputs.
        save_model: Whether to save the trained model.
        verbose: Whether to enable verbose output.
        
    Returns:
        Tuple of (trained model or models, split data dict, training results).
    """
    logger.info(f"Training {model_type} model")
    
    # Initialize preprocessor to use split_train_test
    preprocessor = DataPreprocessor()
    
    # Split data into train and test sets
    splits = preprocessor.split_train_test(
        prepared_data,
        target_col=target_col,
        test_size=test_size,
        time_based=True  # Use time-based split for time series data
    )
    
    # Get model instance
    model = get_model(model_type)
    
    # Fit model based on type
    if model_type in TIME_SERIES_MODELS:
        # For time series models, prepare the training data differently
        train_df = splits['X_train'].copy()
        train_df[target_col] = splits['y_train']
        
        # Extract date index (if needed)
        date_index = train_df.index
        
        # Determine exogenous columns (all except target)
        exog_cols = train_df.columns.tolist()
        if target_col in exog_cols:
            exog_cols.remove(target_col)
        
        model.fit(
            train_df,
            target_col=target_col,
            date_col='date' if 'date' in train_df.columns else None,
            exog_cols=exog_cols if exog_cols else None
        )
    else:
        # For standard ML models, directly fit using the training split
        model.fit(splits['X_train'], splits['y_train'])
    
    # Evaluate model
    if model_type in TIME_SERIES_MODELS:
        test_df = splits['X_test'].copy()
        test_df[target_col] = splits['y_test']
        metrics = model.evaluate(test_df, target_col=target_col)
    else:
        metrics = model.evaluate(splits['X_test'], splits['y_test'])
    
    # Get predictions
    if model_type in TIME_SERIES_MODELS:
        predictions = model.predict(
            steps=len(splits['X_test']),
            future_exog=splits['X_test'] if hasattr(model, 'exog_cols') and model.exog_cols else None,
            include_history=False
        )
        y_pred = predictions['forecast'].values
    else:
        y_pred = model.predict(splits['X_test'])
    
    # Calculate residuals
    residuals = splits['y_test'].values - y_pred
    
    # Log evaluation metrics
    logger.info("Model evaluation metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    # Save model if requested
    if save_model and output_dir:
        models_dir = os.path.join(output_dir, 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_file = os.path.join(models_dir, f"{model_type}_model.pkl")
        try:
            import pickle  # Import here to keep dependencies local to this block
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model successfully saved to {model_file}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    # Get feature importance if available
    feature_importance = None
    if hasattr(model, 'get_feature_importance'):
        try:
            feature_importance = model.get_feature_importance()
            logger.info("Feature importance calculated")
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
    
    # Create evaluation visualizations if output directory is provided
    if output_dir:
        plots_dir = os.path.join(output_dir, 'plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Plot residuals
        fig_residuals = plot_residuals(
            splits['y_test'],
            y_pred,
            dates=splits['X_test'].index if hasattr(splits['X_test'], 'index') else None,
            title=f"{model_type.upper()} Model Residuals"
        )
        fig_residuals.savefig(os.path.join(plots_dir, f"{model_type}_residuals.png"))
        
        # Plot predictions vs. actual
        fig_predictions = ModelEvaluator.plot_predictions(
            splits['y_test'],
            y_pred,
            dates=splits['X_test'].index if hasattr(splits['X_test'], 'index') else None,
            title=f"{model_type.upper()} Predictions vs Actual"
        )
        fig_predictions.savefig(os.path.join(plots_dir, f"{model_type}_predictions.png"))
        
        # Plot feature importance if available
        if feature_importance:
            fig_importance = plot_feature_importance(
                feature_importance,
                title=f"{model_type.upper()} Feature Importance"
            )
            fig_importance.savefig(os.path.join(plots_dir, f"{model_type}_feature_importance.png"))
    
    # Prepare and return results
    results = {
        'metrics': metrics,
        'feature_importance': feature_importance,
        'residuals': residuals,
        'predictions': y_pred
    }
    
    return model, splits, results

def forecast_future(
    model,
    prepared_data: pd.DataFrame,
    target_col: str,
    model_type: str,
    horizon: int = 12,
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate forecasts for future periods.
    
    Args:
        model: Trained model
        prepared_data: DataFrame with prepared data
        target_col: Name of target column
        model_type: Type of model used
        horizon: Number of periods to forecast
        output_dir: Directory to save outputs
        
    Returns:
        DataFrame with forecasts
    """
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Generating {horizon}-period forecast using {model_type}")
    
    if model_type in TIME_SERIES_MODELS:
        # For time series models
        
        # Get the last date in the data
        last_date = prepared_data.index[-1]
        
        # Generate future dates (assuming monthly frequency)
        if isinstance(last_date, pd.Timestamp):
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon,
                freq='MS'  # Month start
            )
        else:
            future_dates = range(1, horizon + 1)
        
        # For models that need exogenous variables, we need to provide them
        future_exog = None
        if hasattr(model, 'exog_cols') and model.exog_cols:
            logger.warning("Time series model requires exogenous variables for forecasting")
            logger.warning("Using simplistic approach to generate future exogenous variables")
            future_exog = pd.DataFrame(
                {col: [prepared_data[col].iloc[-1]] * horizon for col in model.exog_cols},
                index=future_dates
            )
        
        # Generate forecast using the time series model
        forecast = model.predict(steps=horizon, future_exog=future_exog)
        
        # Create forecast DataFrame
        if isinstance(forecast, pd.DataFrame):
            forecast_df = forecast
            if 'date' not in forecast_df.columns and isinstance(future_dates, pd.DatetimeIndex):
                forecast_df['date'] = future_dates
        else:
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'forecast': forecast
            })
    
    else:
        # For traditional ML models, generate future feature values first
        
        if isinstance(prepared_data.index, pd.DatetimeIndex):
            last_date = prepared_data.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon,
                freq='MS'  # Month start
            )
        else:
            last_idx = len(prepared_data)
            future_dates = range(last_idx, last_idx + horizon)
        
        # Get feature columns (exclude the target)
        feature_cols = [col for col in prepared_data.columns if col != target_col]
        
        # Instead of inserting one column at a time, accumulate future features in a dictionary
        future_feature_dict = {}
        for col in feature_cols:
            feature_series = prepared_data[col]
            x = np.arange(len(feature_series))
            y = feature_series.values
            
            # Fit a simple linear regression for trend extrapolation
            trend_model = LinearRegression()
            trend_model.fit(x.reshape(-1, 1), y)
            
            future_x = np.arange(len(feature_series), len(feature_series) + horizon)
            future_y = trend_model.predict(future_x.reshape(-1, 1))
            
            future_feature_dict[col] = future_y
        
        # Create the future features DataFrame in one step
        future_features = pd.DataFrame(future_feature_dict, index=future_dates)
        # Defragment the DataFrame by creating a copy
        future_features = future_features.copy()
        
        # Generate predictions using the trained model
        future_predictions = model.predict(future_features)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecast': future_predictions
        })
    
    # Save forecast if output directory provided
    if output_dir:
        import os
        data_dir = os.path.join(output_dir, 'data')
        forecast_df.to_csv(os.path.join(data_dir, f"{model_type}_forecast.csv"), index=False)
        
        # Generate forecast visualization
        from visualizations import plot_forecast
        plots_dir = os.path.join(output_dir, 'plots')
        
        # Get historical target data for plotting
        historical = prepared_data[[target_col]].copy()
        historical.reset_index(inplace=True)
        if 'index' in historical.columns:
            historical.rename(columns={'index': 'date'}, inplace=True)
        
        fig_forecast = plot_forecast(
            historical,
            forecast_df,
            target_col=target_col,
            date_col='date',
            forecast_col='forecast',
            title=f"{model_type.upper()} {horizon}-Period Forecast"
        )
        fig_forecast.savefig(os.path.join(plots_dir, f"{model_type}_forecast.png"))
    
    return forecast_df

def compare_models(
    prepared_data: pd.DataFrame,
    target_col: str,
    model_types: List[str],
    test_size: float = 0.2,
    output_dir: Optional[str] = None,
    verbose: bool = False
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Train and compare multiple models.
    
    Args:
        prepared_data: DataFrame with prepared data
        target_col: Name of target column
        model_types: List of model types to compare
        test_size: Fraction of data to use for testing
        output_dir: Directory to save outputs
        verbose: Whether to enable verbose output
        
    Returns:
        Tuple of (Dictionary with models and results, DataFrame with comparison metrics)
    """
    logger.info(f"Comparing {len(model_types)} models: {', '.join(model_types)}")
    
    # Initialize preprocessor to use split_train_test
    preprocessor = DataPreprocessor()
    
    # Split data into train and test sets (same for all models)
    splits = preprocessor.split_train_test(
        prepared_data,
        target_col=target_col,
        test_size=test_size,
        time_based=True
    )
    
    # Dictionary to store models and results
    models_dict = {}
    results_dict = {}
    metrics_list = []
    
    # Train and evaluate each model
    for model_type in model_types:
        try:
            logger.info(f"Training and evaluating {model_type} model")
            
            # Train model
            model, _, results = train_model(
                prepared_data,
                target_col=target_col,
                model_type=model_type,
                test_size=test_size,
                output_dir=output_dir,
                verbose=verbose
            )
            
            # Store model and results
            models_dict[model_type] = model
            results_dict[model_type] = results
            
            # Add metrics to list for comparison
            metrics = results['metrics'].copy()
            metrics['model'] = model_type
            metrics_list.append(metrics)
            
        except Exception as e:
            logger.error(f"Error with model {model_type}: {e}")
    
    # Create metrics comparison DataFrame
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.set_index('model', inplace=True)
        
        # Save metrics to CSV
        if output_dir:
            data_dir = os.path.join(output_dir, 'data')
            metrics_df.to_csv(os.path.join(data_dir, 'model_comparison.csv'))
        
        # Create model comparison visualization
        if output_dir:
            plots_dir = os.path.join(output_dir, 'plots')
            
            fig_comparison = plot_model_comparison(
                metrics_df,
                metric_cols=['rmse', 'mae', 'r2'],
                title="Model Performance Comparison"
            )
            fig_comparison.savefig(os.path.join(plots_dir, 'model_comparison.png'))
        
        return {
            'models': models_dict,
            'results': results_dict,
            'splits': splits
        }, metrics_df
    else:
        logger.error("No models were successfully trained and evaluated")
        return {'models': {}, 'results': {}, 'splits': splits}, pd.DataFrame()


def create_model_dashboard(
    prepared_data: pd.DataFrame,
    target_col: str,
    model_comparison_results: Dict[str, Any],
    metrics_df: pd.DataFrame,
    forecasts_dict: Dict[str, pd.DataFrame],
    output_dir: str
) -> None:
    """
    Create a comprehensive dashboard with model comparisons and forecasts.
    
    Args:
        prepared_data: DataFrame with prepared data
        target_col: Name of target column
        model_comparison_results: Dictionary with models, results and splits
        metrics_df: DataFrame with model comparison metrics
        forecasts_dict: Dictionary with model forecasts
        output_dir: Directory to save dashboard
    """
    logger.info("Creating comprehensive dashboard")
    
    # Extract data for dashboard
    results_dict = model_comparison_results['results']
    splits = model_comparison_results['splits']
    
    # Prepare data for dashboard
    target_data = prepared_data[[target_col]].copy()
    target_data.reset_index(inplace=True)
    if 'index' in target_data.columns:
        target_data.rename(columns={'index': 'date'}, inplace=True)
    
    # Get feature importance from all models
    feature_importance_dict = {}
    for model_name, results in results_dict.items():
        if results.get('feature_importance'):
            feature_importance_dict[model_name] = results['feature_importance']
    
    # Use the first feature importance if available
    feature_importance = next(iter(feature_importance_dict.values())) if feature_importance_dict else None
    
    # Create dashboard directory
    dashboard_dir = os.path.join(output_dir, 'dashboard')
    if not os.path.exists(dashboard_dir):
        os.makedirs(dashboard_dir)
    
    # Create dashboard using plotly
    dashboard = create_dashboard(
        data_dict={'target': target_data},
        target_col=target_col,
        forecast_df=next(iter(forecasts_dict.values())) if forecasts_dict else None,
        feature_importance=feature_importance,
        model_metrics=metrics_df,
        output_file=os.path.join(dashboard_dir, 'forecast_dashboard.html')
    )
    
    logger.info(f"Dashboard created and saved to {dashboard_dir}")


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Prepare output directory
    prepare_output_directory(args.output_dir)
    
    # Initialize data connector
    connector = DataConnector()
    
    # Check database connection
    if not connector.check_database_connection():
        logger.error("Database connection failed. Please check your credentials.")
        sys.exit(1)
    
    # Load datasets
    datasets = load_datasets(
        connector,
        args.target,
        args.features,
        args.start_date,
        args.end_date
    )
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(connector)
    
    # Prepare dataset for modeling with enhanced logging
    logger.info("Preparing dataset for modeling with enhanced logging")
    prepared_data, metadata = preprocessor.prepare_dataset(
        target_dataset=args.target,
        feature_datasets=args.features if args.features else list(datasets.keys()),
        start_date=args.start_date,
        end_date=args.end_date,
        frequency='monthly',  # Auto sales are typically monthly data
        create_lags=True,
        lag_periods=[args.feature_lag],  # Use the value from the arguments
        calculate_changes=True,
        calculate_rolling=True,
        rolling_window=3,
        fill_na_method='interpolate',
        handle_outliers=True,
        scale_data=args.scale_data
    )
    
    # Add more detailed logging about the prepared dataset
    logger.info(f"Prepared dataset details:")
    logger.info(f"  Shape: {prepared_data.shape}")
    logger.info(f"  Date range: {prepared_data.index.min()} to {prepared_data.index.max()}")
    logger.info(f"  Target variable: {args.target}")
    logger.info(f"  Feature count: {prepared_data.shape[1] - 1}")  # Subtract 1 for target

    # Log information about feature types
    numeric_features = prepared_data.select_dtypes(include=['number']).columns.tolist()
    categorical_features = prepared_data.select_dtypes(include=['object', 'category']).columns.tolist()
    logger.info(f"  Numeric features: {len(numeric_features)}")
    logger.info(f"  Categorical features: {len(categorical_features)}")

    # Log temporal alignment information if available
    if 'temporal_alignment' in metadata:
        alignment = metadata['temporal_alignment']
        logger.info(f"  Temporal alignment applied: {alignment['aligned']}")
        if 'truncation_percentages' in alignment:
            for dataset, percent in alignment['truncation_percentages'].items():
                if percent > 5:  # Only log significant truncations
                    logger.info(f"  Dataset '{dataset}' had {percent:.1f}% of data truncated for alignment")

    # Log sample of column names to help understand what features were created
    sample_cols = []
    lag_cols = [col for col in prepared_data.columns if '_lag_' in col]
    pct_change_cols = [col for col in prepared_data.columns if '_pct_change' in col]
    rolling_cols = [col for col in prepared_data.columns if '_rolling_' in col]

    logger.info(f"  Lag features created: {len(lag_cols)}")
    logger.info(f"  Percent change features created: {len(pct_change_cols)}")
    logger.info(f"  Rolling statistic features created: {len(rolling_cols)}")

    # Log any potential issues
    if prepared_data.shape[0] < 30:
        logger.warning(f"  Warning: Small dataset size ({prepared_data.shape[0]} rows) may affect model quality")
    if prepared_data.shape[1] > 100:
        logger.warning(f"  Warning: Large feature count ({prepared_data.shape[1]} columns) may cause overfitting")
    
    # Save prepared data
    data_dir = os.path.join(args.output_dir, 'data')
    prepared_data.to_csv(os.path.join(data_dir, 'prepared_data.csv'))
    
    # Generate data visualizations
    plots_dir = os.path.join(args.output_dir, 'plots')
    
    # Time series plot of target variable
    target_col = args.target
    fig_time_series = plot_time_series(
        prepared_data.reset_index(),  # Reset index to convert DatetimeIndex to column
        columns=[target_col],
        date_col='index',  # Use 'index' which will contain the datetime values
        title=f"{target_col.capitalize()} Time Series"
    )
    fig_time_series.savefig(os.path.join(plots_dir, f"{target_col}_time_series.png"))
    
    # Correlation heatmap
    fig_corr = plot_correlation_matrix(
        prepared_data,
        title="Feature Correlation Matrix"
    )
    fig_corr.savefig(os.path.join(plots_dir, "correlation_matrix.png"))
    
    # Initialize dictionaries for models and forecasts
    trained_models = {}
    forecasts_dict = {}
    model_comparison_results = None
    metrics_df = None
    
    # Train model if requested
    if args.mode in ['train', 'all']:
        logger.info(f"Training single model: {args.model_types[0]}")
        model, splits, train_results = train_model(
            prepared_data,
            target_col=args.target,
            model_type=args.model_types[0],
            test_size=args.test_size,
            output_dir=args.output_dir,
            save_model=args.save_model,
            verbose=args.verbose
        )
        trained_models[args.model_types[0]] = model
    
    # Compare models if requested
    if args.mode in ['compare', 'dashboard', 'all'] and len(args.model_types) > 1:
        logger.info("Comparing multiple models")
        model_comparison_results, metrics_df = compare_models(
            prepared_data,
            target_col=args.target,
            model_types=args.model_types,
            test_size=args.test_size,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        
        # Update trained models
        if 'models' in model_comparison_results:
            trained_models.update(model_comparison_results['models'])
    
    # Generate forecast if requested
    if args.mode in ['forecast', 'dashboard', 'all']:
        logger.info("Generating forecasts")
        
        for model_type, model in trained_models.items():
            # Train model if not already trained
            if model is None:
                logger.info(f"Training {model_type} model for forecasting")
                model, _, _ = train_model(
                    prepared_data,
                    target_col=args.target,
                    model_type=model_type,
                    test_size=args.test_size,
                    output_dir=args.output_dir,
                    save_model=args.save_model,
                    verbose=args.verbose
                )
                trained_models[model_type] = model
            
            # Generate forecast
            logger.info(f"Generating forecast using {model_type} model")
            forecast_df = forecast_future(
                model,
                prepared_data,
                target_col=args.target,
                model_type=model_type,
                horizon=args.forecast_horizon,
                output_dir=args.output_dir
            )
            
            # Store forecast
            forecasts_dict[model_type] = forecast_df
    
    # Create dashboard if requested
    if args.mode in ['dashboard', 'all'] and metrics_df is not None:
        logger.info("Creating model comparison dashboard")
        
        # If we haven't compared models yet, do it now
        if model_comparison_results is None and len(trained_models) > 1:
            model_comparison_results, metrics_df = compare_models(
                prepared_data,
                target_col=args.target,
                model_types=list(trained_models.keys()),
                test_size=args.test_size,
                output_dir=args.output_dir,
                verbose=args.verbose
            )
        
        # Create dashboard
        if model_comparison_results is not None:
            create_model_dashboard(
                prepared_data,
                target_col=args.target,
                model_comparison_results=model_comparison_results,
                metrics_df=metrics_df,
                forecasts_dict=forecasts_dict,
                output_dir=args.output_dir
            )
    
    logger.info("Process completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred during execution: {e}")
        sys.exit(1)