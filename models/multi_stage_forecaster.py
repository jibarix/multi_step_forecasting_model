"""
Multi-stage forecasting engine for GDP-anchored forecasting.

This module implements a two-tier forecasting process where GDP projections
drive intermediate economic indicator forecasts, which in turn drive
target variable forecasts. It handles model selection, feature optimization,
and ensures consistency across the forecasting stages.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import json

from models.gdp_projection_handler import GDPProjectionHandler
from models.model_factory import get_model, create_model_pipeline, auto_select_model
from models.anchor_selection import AnchorVariableSelector
from models.evaluator import ModelEvaluator
from dimension_reduction.pca_handler import PCAHandler
from dimension_reduction.feature_clustering import FeatureClustering, cluster_gdp_related_features
from data.multi_frequency import MultiFrequencyHandler
from data.preprocessor import DataPreprocessor

# Set up logging
logger = logging.getLogger(__name__)

class MultiStageForecastEngine:
    """
    Engine for GDP-anchored multi-stage forecasting.
    
    This class implements a two-tier forecasting process:
    1. GDP projections → Economic indicator forecasts (Tier 1)
    2. Economic indicators → Target variable forecast (Tier 2)
    
    It supports dimension reduction through PCA or feature clustering
    and provides flexibility in model selection and feature optimization.
    """
    
    def __init__(
        self,
        target_column: str,
        date_column: str = 'date',
        gdp_column: str = 'real_gdp',
        dimension_reduction: str = 'hybrid',
        forecasting_approach: str = 'ensemble',
        output_dir: Optional[str] = None
    ):
        """
        Initialize the multi-stage forecast engine.
        
        Args:
            target_column: Name of target column to forecast
            date_column: Name of date column
            gdp_column: Name of GDP column
            dimension_reduction: Dimension reduction method
                ('pca', 'clustering', 'hybrid', 'none')
            forecasting_approach: Forecasting approach
                ('ensemble', 'best_model', 'lightgbm', 'prophet')
            output_dir: Directory to save outputs
        """
        self.target_column = target_column
        self.date_column = date_column
        self.gdp_column = gdp_column
        self.dimension_reduction = dimension_reduction
        self.forecasting_approach = forecasting_approach
        self.output_dir = output_dir
        
        # Initialize handlers and models
        self.gdp_handler = GDPProjectionHandler(gdp_column=gdp_column, date_column=date_column)
        self.anchor_selector = AnchorVariableSelector()
        self.multi_freq_handler = MultiFrequencyHandler()
        self.preprocessor = DataPreprocessor()
        
        # Component models and transformation objects
        self.pca_model = None
        self.clustering_model = None
        self.gdp_to_indicator_models = {}
        self.target_model = None
        
        # Store metadata and state
        self.anchor_variables = []
        self.selected_features = []
        self.feature_importance = {}
        self.model_metrics = {}
        self.is_trained = False
        
        # Cache for intermediate results
        self.cache = {}
    
    def fit(
        self,
        data: pd.DataFrame,
        gdp_data: pd.DataFrame = None,
        num_anchors: int = 5,
        test_size: float = 0.2
    ):
        """
        Fit the complete two-tier forecasting model.
        
        Args:
            data: DataFrame with target and features
            gdp_data: DataFrame with GDP data (if different from main data)
            num_anchors: Number of anchor variables to select
            test_size: Proportion of data to use for testing
            
        Returns:
            Self for method chaining
        """
        # Ensure data has datetime index
        main_data = data.copy()
        if self.date_column in main_data.columns:
            main_data[self.date_column] = pd.to_datetime(main_data[self.date_column])
            main_data = main_data.set_index(self.date_column)
        
        # Process GDP data
        if gdp_data is not None:
            gdp_df = gdp_data.copy()
            if self.date_column in gdp_df.columns:
                gdp_df[self.date_column] = pd.to_datetime(gdp_df[self.date_column])
                gdp_df = gdp_df.set_index(self.date_column)
            
            # Merge GDP data with main data
            main_data = main_data.join(gdp_df, how='left')
            
            # Fill missing GDP values (e.g., interpolate for monthly data)
            if self.gdp_column in main_data.columns:
                main_data[self.gdp_column] = main_data[self.gdp_column].interpolate(method='cubic')
        
        # Step 1: Select anchor variables based on GDP relationships
        logger.info("Selecting anchor variables based on GDP relationships")
        
        if self.dimension_reduction == 'pca' or self.dimension_reduction == 'hybrid':
            # Use PCA for dimension reduction
            logger.info("Applying PCA for dimension reduction")
            
            # Create feature matrix excluding target and date
            feature_cols = [col for col in main_data.columns 
                           if col != self.target_column]
            
            # Initialize and fit PCA
            self.pca_model = PCAHandler(variance_threshold=0.95)
            self.pca_model.fit(main_data[feature_cols], target_col=self.target_column)
            
            # Find GDP-related components
            gdp_comps = self.pca_model.find_gdp_related_components(
                main_data, self.gdp_column, threshold=0.3
            )
            
            # Transform data to component space
            pc_data = self.pca_model.transform(main_data)
            
            # Use top GDP-related components as anchors
            self.anchor_variables = [comp for comp, _ in gdp_comps[:num_anchors]]
            
            # Cache component data for later use
            self.cache['pc_data'] = pc_data
        
        if self.dimension_reduction == 'clustering' or self.dimension_reduction == 'hybrid':
            # Use feature clustering
            logger.info("Applying feature clustering")
            
            # Create feature matrix excluding target
            feature_cols = [col for col in main_data.columns 
                           if col != self.target_column]
            feature_data = main_data[feature_cols]
            
            # Apply clustering
            self.clustering_model, gdp_representatives, _ = cluster_gdp_related_features(
                feature_data,
                self.gdp_column,
                correlation_threshold=0.3,
                max_clusters=num_anchors
            )
            
            # Get cluster representatives
            cluster_reps = list(self.clustering_model.get_cluster_representatives().values())
            
            # Add to anchor variables if using hybrid approach
            if self.dimension_reduction == 'hybrid':
                for rep in cluster_reps:
                    if rep not in self.anchor_variables and len(self.anchor_variables) < num_anchors:
                        self.anchor_variables.append(rep)
            else:
                self.anchor_variables = cluster_reps
        
        if self.dimension_reduction == 'none':
            # Use direct variable selection without dimension reduction
            logger.info("Using direct variable selection (no dimension reduction)")
            
            # Get optimal anchor combination
            selected_anchors = self.anchor_selector.get_optimal_anchor_combination(
                main_data, self.target_column, max_anchors=num_anchors
            )
            
            self.anchor_variables = selected_anchors
        
        logger.info(f"Selected anchor variables: {self.anchor_variables}")
        
        # Step 2: Build GDP-to-Anchor models
        logger.info("Building GDP-to-anchor variable models")
        
        for anchor in self.anchor_variables:
            logger.info(f"Training model for anchor: {anchor}")
            
            # Create data for this anchor
            anchor_data = main_data[[self.gdp_column, anchor]].dropna()
            
            # Check if we have enough data
            if len(anchor_data) < 10:
                logger.warning(f"Not enough data for anchor {anchor}, skipping")
                continue
            
            # Split data
            train_size = int(len(anchor_data) * (1 - test_size))
            train_data = anchor_data.iloc[:train_size]
            test_data = anchor_data.iloc[train_size:]
            
            # Prepare X and y
            X_train = train_data[[self.gdp_column]]
            y_train = train_data[anchor]
            X_test = test_data[[self.gdp_column]]
            y_test = test_data[anchor]
            
            # Select model type based on data characteristics
            if len(train_data) < 30:
                # For small datasets, use simpler models
                model_type = 'linear_regression'
            else:
                # For larger datasets, prefer more complex models
                model_type = 'lightgbm'
            
            # Create and fit model
            try:
                model = get_model(model_type)
                model.fit(X_train, y_train)
                
                # Evaluate model
                metrics = model.evaluate(X_test, y_test)
                logger.info(f"GDP-to-{anchor} model metrics: {metrics}")
                
                # Store model and metrics
                self.gdp_to_indicator_models[anchor] = {
                    'model': model,
                    'metrics': metrics,
                    'type': model_type
                }
            except Exception as e:
                logger.error(f"Error training GDP-to-{anchor} model: {e}")
        
        # Step 3: Build Indicators-to-Target model
        logger.info("Building indicators-to-target model")
        
        # Create data for target model
        target_features = self.anchor_variables.copy()
        
        # Add target column
        model_data = main_data[target_features + [self.target_column]].dropna()
        
        # Check if we have enough data
        if len(model_data) < 10:
            raise ValueError(f"Not enough data for target model: {len(model_data)} rows")
        
        # Split data
        X = model_data[target_features]
        y = model_data[self.target_column]
        
        splits = self.preprocessor.split_train_test(
            model_data,
            target_col=self.target_column,
            test_size=test_size
        )
        
        # Determine best model based on forecasting approach
        if self.forecasting_approach == 'ensemble':
            # Create and evaluate multiple models
            models = create_model_pipeline(
                ['linear_regression', 'random_forest', 'lightgbm', 'prophet'],
                ensemble_method='average'
            )
            
            # Train each model
            trained_models = {}
            for name, model in models.items():
                try:
                    if name.startswith('prophet'):
                        # Special handling for Prophet
                        df_train = splits['X_train'].copy()
                        df_train[self.target_column] = splits['y_train']
                        model.fit(df_train, target_col=self.target_column)
                    else:
                        # Standard ML models
                        model.fit(splits['X_train'], splits['y_train'])
                    
                    trained_models[name] = model
                except Exception as e:
                    logger.error(f"Error training {name} model: {e}")
            
            # Compare models
            comparison = ModelEvaluator.compare_models(
                trained_models,
                splits['X_test'],
                splits['y_test']
            )
            
            # Select best model
            best_model_name = comparison['rmse'].idxmin()
            self.target_model = trained_models[best_model_name]
            logger.info(f"Selected best model: {best_model_name}")
            
            # Store comparison
            self.model_metrics = comparison.to_dict()
            
        elif self.forecasting_approach == 'best_model':
            # Automatically select best model
            self.target_model = auto_select_model(
                splits['X_train'],
                splits['y_train'],
                splits['X_val'] if 'X_val' in splits else splits['X_test'],
                splits['y_val'] if 'y_val' in splits else splits['y_test']
            )
            
            # Evaluate selected model
            metrics = self.target_model.evaluate(splits['X_test'], splits['y_test'])
            self.model_metrics = {'auto_selected': metrics}
            
        else:
            # Use specified model type
            model_type = self.forecasting_approach
            model = get_model(model_type)
            
            # Fit model
            if model_type in ['prophet']:
                # Special handling for Prophet
                df_train = pd.concat([splits['X_train'], splits['y_train']], axis=1)
                model.fit(df_train, target_col=self.target_column)
            else:
                # Standard ML models
                model.fit(splits['X_train'], splits['y_train'])
            
            # Evaluate model
            metrics = model.evaluate(splits['X_test'], splits['y_test'])
            self.model_metrics = {model_type: metrics}
            
            self.target_model = model
        
        # Get feature importance if available
        if hasattr(self.target_model, 'get_feature_importance'):
            try:
                self.feature_importance = self.target_model.get_feature_importance()
                logger.info(f"Feature importance: {self.feature_importance}")
            except:
                pass
        
        # Mark as trained
        self.selected_features = target_features
        self.is_trained = True
        
        # Save metadata if output directory is specified
        if self.output_dir:
            self._save_metadata()
        
        return self
    
    def forecast(
        self,
        gdp_projections: str,
        historical_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate forecast using GDP projections.
        
        Args:
            gdp_projections: String with GDP projections (e.g., '2025.1: -1.8, 2025.2: 2.0')
            historical_data: Optional historical data for context
            
        Returns:
            DataFrame with forecasts
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting")
        
        # Step 1: Parse GDP projections
        logger.info("Parsing GDP projections")
        self.gdp_handler.parse_projections(gdp_projections)
        
        # Get monthly GDP projections
        monthly_gdp = self.gdp_handler.get_monthly_projections()
        
        # Step 2: Generate anchor variable forecasts from GDP
        logger.info("Generating anchor variable forecasts from GDP")
        
        # Create DataFrame to store anchor forecasts
        anchor_forecasts = pd.DataFrame({self.date_column: monthly_gdp[self.date_column]})
        anchor_forecasts[self.gdp_column] = monthly_gdp['gdp_growth']
        
        # Generate forecast for each anchor variable
        for anchor, model_info in self.gdp_to_indicator_models.items():
            model = model_info['model']
            
            # Prepare input for prediction
            X_pred = anchor_forecasts[[self.gdp_column]]
            
            # Generate prediction
            try:
                predictions = model.predict(X_pred)
                anchor_forecasts[anchor] = predictions
            except Exception as e:
                logger.error(f"Error predicting anchor {anchor}: {e}")
                # Fill with last value from historical data if available
                if historical_data is not None and anchor in historical_data.columns:
                    last_value = historical_data[anchor].iloc[-1]
                    anchor_forecasts[anchor] = last_value
                else:
                    # Fill with zeros
                    anchor_forecasts[anchor] = 0
        
        # Step 3: Generate target forecast from anchor variables
        logger.info("Generating target forecast from anchor variables")
        
        # Prepare input for target model
        X_target = anchor_forecasts[self.selected_features]
        
        # Generate target forecast
        if hasattr(self.target_model, 'predict'):
            # Standard ML model
            target_forecast = self.target_model.predict(X_target)
        else:
            # Alternative interface
            forecast_df = self.target_model.forecast(steps=len(anchor_forecasts))
            target_forecast = forecast_df['forecast'].values
        
        # Add target forecast to results
        anchor_forecasts[self.target_column] = target_forecast
        
        return anchor_forecasts
    
    def evaluate_forecast(
        self,
        historical_data: pd.DataFrame,
        forecast_data: pd.DataFrame,
        evaluation_period: Optional[Tuple[str, str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate forecast against historical data.
        
        Args:
            historical_data: Historical data with actual values
            forecast_data: Forecast data with predicted values
            evaluation_period: Optional tuple of (start_date, end_date) strings
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Copy data to avoid modifying originals
        historical = historical_data.copy()
        forecast = forecast_data.copy()
        
        # Ensure date columns are datetime
        if self.date_column in historical.columns:
            historical[self.date_column] = pd.to_datetime(historical[self.date_column])
        
        if self.date_column in forecast.columns:
            forecast[self.date_column] = pd.to_datetime(forecast[self.date_column])
        
        # Filter to evaluation period if specified
        if evaluation_period:
            start_date, end_date = evaluation_period
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            historical = historical[
                (historical[self.date_column] >= start_date) &
                (historical[self.date_column] <= end_date)
            ]
            
            forecast = forecast[
                (forecast[self.date_column] >= start_date) &
                (forecast[self.date_column] <= end_date)
            ]
        
        # Merge on date to align data
        merged = pd.merge(
            historical[[self.date_column, self.target_column]],
            forecast[[self.date_column, self.target_column]],
            on=self.date_column,
            suffixes=('_actual', '_forecast')
        )
        
        # Check if there's enough data for evaluation
        if len(merged) < 1:
            return {"error": "No overlapping data for evaluation"}
        
        # Calculate metrics
        actual = merged[f"{self.target_column}_actual"]
        predicted = merged[f"{self.target_column}_forecast"]
        
        return ModelEvaluator.calculate_metrics(actual, predicted)
    
    def generate_scenario_forecasts(
        self,
        gdp_scenarios: Dict[str, str],
        historical_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for multiple GDP scenarios.
        
        Args:
            gdp_scenarios: Dictionary mapping scenario names to GDP projection strings
            historical_data: Optional historical data for context
            
        Returns:
            Dictionary mapping scenario names to forecast DataFrames
        """
        scenario_forecasts = {}
        
        for scenario_name, gdp_projection in gdp_scenarios.items():
            logger.info(f"Generating forecast for scenario: {scenario_name}")
            
            try:
                forecast = self.forecast(gdp_projection, historical_data)
                scenario_forecasts[scenario_name] = forecast
            except Exception as e:
                logger.error(f"Error generating forecast for scenario {scenario_name}: {e}")
        
        return scenario_forecasts
    
    def _save_metadata(self):
        """
        Save model metadata to output directory.
        """
        if not self.output_dir:
            return
        
        # Create metadata dictionary
        metadata = {
            'target_column': self.target_column,
            'date_column': self.date_column,
            'gdp_column': self.gdp_column,
            'dimension_reduction': self.dimension_reduction,
            'forecasting_approach': self.forecasting_approach,
            'anchor_variables': self.anchor_variables,
            'selected_features': self.selected_features,
            'model_metrics': self.model_metrics,
            'feature_importance': {k: float(v) for k, v in self.feature_importance.items()},
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Create directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save metadata to file
        metadata_path = os.path.join(self.output_dir, 'forecast_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    def save(self, filepath: str):
        """
        Save the entire multi-stage forecaster to a file.
        
        Args:
            filepath: Path to save the model
        """
        import pickle
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Multi-stage forecaster saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'MultiStageForecastEngine':
        """
        Load a multi-stage forecaster from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded MultiStageForecastEngine instance
        """
        import pickle
        
        # Load from file
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Multi-stage forecaster loaded from {filepath}")
        return model


def run_multi_stage_forecast(
    data: pd.DataFrame,
    target_column: str,
    gdp_projections: str,
    dimension_reduction: str = 'hybrid',
    output_dir: Optional[str] = None,
    test_size: float = 0.2,
    num_anchors: int = 5,
    output_format: str = 'csv'
) -> Dict[str, Any]:
    """
    Run a complete multi-stage forecast workflow.
    
    Args:
        data: DataFrame with historical data
        target_column: Name of target column to forecast
        gdp_projections: String with GDP projections
        dimension_reduction: Dimension reduction method
        output_dir: Directory to save outputs
        test_size: Proportion of data to use for testing
        num_anchors: Number of anchor variables to use
        output_format: Format for output files ('csv', 'json')
        
    Returns:
        Dictionary with results and metadata
    """
    # Initialize forecaster
    forecaster = MultiStageForecastEngine(
        target_column=target_column,
        dimension_reduction=dimension_reduction,
        output_dir=output_dir
    )
    
    # Fit model
    forecaster.fit(
        data=data,
        num_anchors=num_anchors,
        test_size=test_size
    )
    
    # Generate forecast
    forecast = forecaster.forecast(gdp_projections)
    
    # Evaluate forecast
    metrics = forecaster.evaluate_forecast(data, forecast)
    
    # Save results if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save forecast
        forecast_path = os.path.join(output_dir, f"{target_column}_forecast.{output_format}")
        if output_format == 'csv':
            forecast.to_csv(forecast_path, index=False)
        elif output_format == 'json':
            forecast.to_json(forecast_path, orient='records', date_format='iso')
        
        # Save metrics
        metrics_path = os.path.join(output_dir, f"{target_column}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save model
        model_path = os.path.join(output_dir, f"{target_column}_model.pkl")
        forecaster.save(model_path)
    
    # Return results
    return {
        'forecast': forecast,
        'metrics': metrics,
        'forecaster': forecaster
    }