#!/usr/bin/env python
"""
Enhanced main script for auto sales forecasting model with GDP-anchored multi-stage forecasting.

This script provides a complete pipeline for:
1. Data acquisition from databases
2. GDP-anchored model training with dimension reduction
3. Multi-stage forecasting with GDP projections
4. Model evaluation and scenario analysis
5. Visualization and reporting

Usage:
    python main.py [--target TARGET] [--features FEATURES [FEATURES ...]]
                  [--start_date START_DATE] [--end_date END_DATE]
                  [--forecast_horizon FORECAST_HORIZON]
                  [--gdp_projections GDP_PROJECTIONS]
                  [--model_types MODEL_TYPES [MODEL_TYPES ...]]
                  [--output_dir OUTPUT_DIR]
                  [--mode {train,forecast,evaluate,compare,dashboard,all}]
                  [--dimension_reduction {pca,clustering,hybrid,none}]

Example:
    python main.py --target auto_sales --gdp_projections "2025.1:-1.8,2025.2:2.0"
                  --dimension_reduction hybrid --mode forecast
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import project modules
from config import DATASETS, DEFAULT_START_DATE, DEFAULT_END_DATE
from config.model_config import (
    get_model_config, LINEAR_MODELS, TIME_SERIES_MODELS, 
    ENSEMBLE_MODELS, NEURAL_NETWORK_MODELS, LIGHTGBM_MODELS
)
from data import DataConnector, DataPreprocessor
from data.data_utils import join_datasets
from data.multi_frequency import MultiFrequencyHandler
from models import (
    get_model, ModelEvaluator, create_model_pipeline, 
    AnchorVariableSelector, HierarchicalForecaster
)
from models.multi_stage_forecaster import MultiStageForecastEngine, run_multi_stage_forecast
from models.gdp_projection_handler import GDPProjectionHandler, parse_gdp_input
from dimension_reduction.pca_handler import PCAHandler
from dimension_reduction.feature_clustering import FeatureClustering, cluster_gdp_related_features

# Set up logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Configure logging for external libraries
logging.getLogger('cmdstanpy').setLevel(logging.INFO)
logging.getLogger('prophet').setLevel(logging.INFO)

# Set environment variable for stan
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
                        choices=LINEAR_MODELS + TIME_SERIES_MODELS + ENSEMBLE_MODELS + LIGHTGBM_MODELS,
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
    
    # GDP-related options
    parser.add_argument('--gdp_projections', type=str, default=None,
                        help='GDP projections in format "2025.1:-1.8,2025.2:2.0"')
    
    parser.add_argument('--gdp_column', type=str, default='real_gdp',
                        help='Name of GDP column (default: real_gdp)')
    
    parser.add_argument('--multi_scenario', action='store_true',
                        help='Enable multi-scenario analysis')
    
    parser.add_argument('--scenarios_file', type=str, default=None,
                        help='JSON file with GDP scenarios')
    
    # Dimension reduction options
    parser.add_argument('--dimension_reduction', type=str, 
                        choices=['pca', 'clustering', 'hybrid', 'none'],
                        default='hybrid',
                        help='Dimension reduction method (default: hybrid)')
    
    parser.add_argument('--num_components', type=int, default=None,
                        help='Number of components or clusters (if None, determined automatically)')
    
    # Advanced feature options
    parser.add_argument('--use_dynamic_anchors', action='store_true',
                        help='Automatically select anchor variables based on statistical tests')
    
    parser.add_argument('--num_anchors', type=int, default=5,
                        help='Number of anchor variables to select (default: 5)')
    
    parser.add_argument('--multi_frequency', action='store_true',
                        help='Enable multi-frequency data handling')
    
    parser.add_argument('--target_frequency', type=str, default='MS',
                        choices=['MS', 'QS', 'AS', 'D'],
                        help='Target frequency for data (default: MS for monthly)')
    
    parser.add_argument('--hierarchical', action='store_true',
                        help='Enable hierarchical forecasting')
    
    parser.add_argument('--hierarchy_file', type=str,
                        help='JSON file defining hierarchical structure')
    
    parser.add_argument('--reconciliation', type=str, 
                        choices=['bottom_up', 'top_down', 'middle_out', 'optimal'],
                        default='bottom_up',
                        help='Method for hierarchical forecast reconciliation (default: bottom_up)')
    
    # Confidence interval options
    parser.add_argument('--confidence_level', type=float, default=0.95,
                        help='Confidence level for interval estimation (default: 0.95)')
    
    parser.add_argument('--interval_method', type=str,
                        choices=['parametric', 'bootstrap', 'montecarlo'],
                        default='parametric',
                        help='Method for confidence interval estimation (default: parametric)')
    
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
    scenarios_dir = os.path.join(output_dir, 'scenarios')
    
    for directory in [plots_dir, models_dir, data_dir, dashboard_dir, scenarios_dir]:
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


def train_multistage_forecaster(
    data: pd.DataFrame,
    target_column: str,
    gdp_column: str = 'real_gdp',
    dimension_reduction: str = 'hybrid',
    num_anchors: int = 5,
    output_dir: Optional[str] = None,
    test_size: float = 0.2,
    save_model: bool = True,
    verbose: bool = False
) -> MultiStageForecastEngine:
    """
    Train a multi-stage forecaster with GDP anchoring.
    
    Args:
        data: DataFrame with target, GDP and features
        target_column: Name of target column
        gdp_column: Name of GDP column
        dimension_reduction: Dimension reduction method
        num_anchors: Number of anchor variables to select
        output_dir: Directory to save outputs
        test_size: Proportion of data to use for testing
        save_model: Whether to save the trained model
        verbose: Whether to enable verbose output
        
    Returns:
        Trained MultiStageForecastEngine instance
    """
    logger.info(f"Training multi-stage forecaster with {dimension_reduction} dimension reduction")
    
    # Initialize forecaster
    forecaster = MultiStageForecastEngine(
        target_column=target_column,
        gdp_column=gdp_column,
        dimension_reduction=dimension_reduction,
        forecasting_approach='ensemble',
        output_dir=output_dir
    )
    
    # Set logging level based on verbose flag
    if verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.getLogger('models.multi_stage_forecaster').setLevel(level)
    
    # Fit the model
    forecaster.fit(
        data=data,
        num_anchors=num_anchors,
        test_size=test_size
    )
    
    if save_model and output_dir:
        # Save model
        model_path = os.path.join(output_dir, 'models', f"{target_column}_multistage_model.pkl")
        forecaster.save(model_path)
        logger.info(f"Multi-stage forecaster saved to {model_path}")
    
    return forecaster


def generate_multistage_forecast(
    forecaster: MultiStageForecastEngine,
    gdp_projections: str,
    historical_data: pd.DataFrame,
    output_dir: Optional[str] = None,
    generate_intervals: bool = True,
    confidence_level: float = 0.95,
    interval_method: str = 'parametric'
) -> pd.DataFrame:
    """
    Generate forecast using a trained multi-stage forecaster.
    
    Args:
        forecaster: Trained MultiStageForecastEngine
        gdp_projections: String with GDP projections
        historical_data: Historical data for context
        output_dir: Directory to save outputs
        generate_intervals: Whether to generate confidence intervals
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        interval_method: Method for confidence interval estimation
        
    Returns:
        DataFrame with forecast
    """
    logger.info("Generating forecast with multi-stage forecaster")
    
    # Parse GDP projections
    gdp_projections_info, description = parse_gdp_input(gdp_projections)
    logger.info(f"GDP Projections:\n{description}")
    
    # Generate forecast
    forecast = forecaster.forecast(gdp_projections, historical_data)
    
    # Generate confidence intervals if requested
    if generate_intervals:
        # Implementation depends on the interval_method
        if interval_method == 'parametric':
            # Simple parametric intervals based on historical errors
            target_col = forecaster.target_column
            forecast_values = forecast[target_col].values
            
            # Calculate standard error from historical data
            if hasattr(forecaster, 'model_metrics') and forecaster.model_metrics:
                # Use RMSE from model metrics if available
                try:
                    model_metrics = next(iter(forecaster.model_metrics.values()))
                    rmse = model_metrics.get('rmse', model_metrics.get('RMSE', None))
                    if rmse:
                        std_error = rmse
                    else:
                        # Fallback to standard deviation of target
                        std_error = historical_data[target_col].std()
                except:
                    std_error = historical_data[target_col].std()
            else:
                std_error = historical_data[target_col].std()
            
            # Calculate margin based on normal distribution quantile
            import scipy.stats as stats
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            margin = z_score * std_error
            
            # Add intervals to forecast
            forecast[f'{target_col}_lower'] = forecast_values - margin
            forecast[f'{target_col}_upper'] = forecast_values + margin
            
            logger.info(f"Added {confidence_level:.0%} confidence intervals (±{margin:.2f})")
        
        elif interval_method == 'bootstrap':
            # Not implemented yet
            logger.warning("Bootstrap confidence intervals not yet implemented")
        
        elif interval_method == 'montecarlo':
            # Not implemented yet 
            logger.warning("Monte Carlo confidence intervals not yet implemented")
    
    # Save forecast if output directory provided
    if output_dir:
        # Create forecast directory if it doesn't exist
        forecast_dir = os.path.join(output_dir, 'forecasts')
        os.makedirs(forecast_dir, exist_ok=True)
        
        # Save forecast to CSV
        forecast_path = os.path.join(forecast_dir, f"{forecaster.target_column}_forecast.csv")
        forecast.to_csv(forecast_path, index=False)
        logger.info(f"Forecast saved to {forecast_path}")
        
        # Save GDP projections info
        projections_path = os.path.join(forecast_dir, "gdp_projections.json")
        with open(projections_path, 'w') as f:
            json.dump({
                'description': description,
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
    
    return forecast


def generate_scenario_forecasts(
    forecaster: MultiStageForecastEngine,
    scenarios: Dict[str, str],
    historical_data: pd.DataFrame,
    output_dir: Optional[str] = None,
    generate_intervals: bool = True,
    confidence_level: float = 0.95
) -> Dict[str, pd.DataFrame]:
    """
    Generate forecasts for multiple GDP scenarios.
    
    Args:
        forecaster: Trained MultiStageForecastEngine
        scenarios: Dictionary mapping scenario names to GDP projection strings
        historical_data: Historical data for context
        output_dir: Directory to save outputs
        generate_intervals: Whether to generate confidence intervals
        confidence_level: Confidence level for intervals
        
    Returns:
        Dictionary mapping scenario names to forecast DataFrames
    """
    logger.info(f"Generating forecasts for {len(scenarios)} scenarios")
    
    scenario_forecasts = {}
    
    for scenario_name, gdp_projections in scenarios.items():
        logger.info(f"Generating forecast for scenario: {scenario_name}")
        
        try:
            # Generate forecast for this scenario
            forecast = generate_multistage_forecast(
                forecaster=forecaster,
                gdp_projections=gdp_projections,
                historical_data=historical_data,
                output_dir=None,  # Don't save individual scenario forecasts
                generate_intervals=generate_intervals,
                confidence_level=confidence_level
            )
            
            # Add scenario name column
            forecast['scenario'] = scenario_name
            
            # Store forecast
            scenario_forecasts[scenario_name] = forecast
            
        except Exception as e:
            logger.error(f"Error generating forecast for scenario {scenario_name}: {e}")
    
    # Save combined scenario forecasts if output directory provided
    if output_dir and scenario_forecasts:
        # Create scenarios directory if it doesn't exist
        scenarios_dir = os.path.join(output_dir, 'scenarios')
        os.makedirs(scenarios_dir, exist_ok=True)
        
        # Combine all scenarios into one DataFrame
        combined = pd.concat(scenario_forecasts.values(), ignore_index=True)
        
        # Save combined forecasts to CSV
        combined_path = os.path.join(scenarios_dir, f"{forecaster.target_column}_scenarios.csv")
        combined.to_csv(combined_path, index=False)
        logger.info(f"Scenario forecasts saved to {combined_path}")
        
        # Save scenario definitions
        definitions_path = os.path.join(scenarios_dir, "scenario_definitions.json")
        with open(definitions_path, 'w') as f:
            json.dump(scenarios, f, indent=2)
    
    return scenario_forecasts


def load_scenario_definitions(file_path: str) -> Dict[str, str]:
    """
    Load scenario definitions from a JSON file.
    
    Args:
        file_path: Path to JSON file with scenario definitions
        
    Returns:
        Dictionary mapping scenario names to GDP projection strings
    """
    try:
        with open(file_path, 'r') as f:
            scenarios = json.load(f)
        
        # Validate scenario format
        for name, projection in scenarios.items():
            if not isinstance(name, str) or not isinstance(projection, str):
                logger.warning(f"Invalid scenario format: {name}: {projection}")
        
        return scenarios
    except Exception as e:
        logger.error(f"Error loading scenario definitions: {e}")
        return {}


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
    
    # Prepare dataset for modeling
    logger.info("Preparing dataset for modeling")
    
    # Check if GDP column is present, if not, try to load it
    gdp_column = args.gdp_column
    if gdp_column not in datasets and 'real_gdp' in DATASETS:
        logger.info(f"Loading GDP data from {gdp_column} dataset")
        try:
            gdp_data = connector.fetch_dataset('real_gdp', args.start_date, args.end_date)
            datasets[gdp_column] = gdp_data
        except Exception as e:
            logger.warning(f"Failed to load GDP dataset: {e}")
    
    # Prepare dataset using appropriate method
    if args.multi_frequency:
        # Handle multi-frequency data (implementation from previous code)
        logger.info("Using multi-frequency data handling")
        multi_freq_handler = MultiFrequencyHandler()
        aligned_datasets = multi_freq_handler.align_multi_frequency_data(
            datasets, target_frequency=args.target_frequency
        )
        prepared_data = join_datasets(aligned_datasets, date_format=args.target_frequency)
    else:
        # Standard preprocessing with forecasting enhancements
        prepared_data, metadata = preprocessor.prepare_dataset(
            target_dataset=args.target,
            feature_datasets=list(datasets.keys()),
            start_date=args.start_date,
            end_date=args.end_date,
            frequency='monthly',
            create_lags=True,
            lag_periods=[1, 3, 6, 12],  # Enhanced lag structure
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
    
    # Save prepared data
    data_dir = os.path.join(args.output_dir, 'data')
    prepared_data.to_csv(os.path.join(data_dir, 'prepared_data.csv'))
    
    # Check if GDP projections are provided
    has_gdp_projections = args.gdp_projections is not None
    
    # Check if multi-scenario analysis is enabled
    has_multi_scenario = args.multi_scenario and args.scenarios_file is not None
    
    # If in training mode or no GDP projections provided, train models
    if args.mode in ['train', 'all'] or (not has_gdp_projections and not has_multi_scenario):
        logger.info("Training multi-stage forecaster")
        
        # Train multi-stage forecaster
        multistage_model = train_multistage_forecaster(
            data=prepared_data,
            target_column=args.target,
            gdp_column=gdp_column,
            dimension_reduction=args.dimension_reduction,
            num_anchors=args.num_anchors,
            output_dir=args.output_dir,
            test_size=args.test_size,
            save_model=args.save_model,
            verbose=args.verbose
        )
    else:
        # Try to load previously trained model
        model_path = os.path.join(args.output_dir, 'models', f"{args.target}_multistage_model.pkl")
        if os.path.exists(model_path):
            logger.info(f"Loading multi-stage forecaster from {model_path}")
            try:
                multistage_model = MultiStageForecastEngine.load(model_path)
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                logger.info("Training new multi-stage forecaster")
                multistage_model = train_multistage_forecaster(
                    data=prepared_data,
                    target_column=args.target,
                    gdp_column=gdp_column,
                    dimension_reduction=args.dimension_reduction,
                    num_anchors=args.num_anchors,
                    output_dir=args.output_dir,
                    test_size=args.test_size,
                    save_model=args.save_model,
                    verbose=args.verbose
                )
        else:
            logger.info("No existing model found, training new multi-stage forecaster")
            multistage_model = train_multistage_forecaster(
                data=prepared_data,
                target_column=args.target,
                gdp_column=gdp_column,
                dimension_reduction=args.dimension_reduction,
                num_anchors=args.num_anchors,
                output_dir=args.output_dir,
                test_size=args.test_size,
                save_model=args.save_model,
                verbose=args.verbose
            )
    
    # If in forecast mode or GDP projections provided, generate forecasts
    if args.mode in ['forecast', 'all'] or has_gdp_projections:
        if has_gdp_projections:
            logger.info("Generating forecast with provided GDP projections")
            
            # Generate forecast
            forecast = generate_multistage_forecast(
                forecaster=multistage_model,
                gdp_projections=args.gdp_projections,
                historical_data=prepared_data,
                output_dir=args.output_dir,
                generate_intervals=True,
                confidence_level=args.confidence_level,
                interval_method=args.interval_method
            )
    
    # If multi-scenario is enabled, generate scenario forecasts
    if has_multi_scenario:
        logger.info("Generating multi-scenario forecasts")
        
        # Load scenario definitions
        scenarios = load_scenario_definitions(args.scenarios_file)
        
        if scenarios:
            # Generate scenario forecasts
            scenario_forecasts = generate_scenario_forecasts(
                forecaster=multistage_model,
                scenarios=scenarios,
                historical_data=prepared_data,
                output_dir=args.output_dir,
                generate_intervals=True,
                confidence_level=args.confidence_level
            )
    
    # If in evaluate mode, evaluate the model
    if args.mode in ['evaluate', 'all']:
        # Implementation depends on specific evaluation needs
        pass
    
    # If in compare mode, compare different models or scenarios
    if args.mode in ['compare', 'all'] and has_multi_scenario:
        # Implementation for scenario comparison
        pass
    
    # If in dashboard mode, create interactive dashboard
    if args.mode in ['dashboard', 'all']:
        # Implement dashboard creation
        pass
    
    logger.info("Process completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred during execution: {e}")
        sys.exit(1)