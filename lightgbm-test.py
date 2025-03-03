#!/usr/bin/env python
"""
Test script for LightGBM model integration.
This script verifies that LightGBM models can be created and used through the model factory.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from config.model_config import get_model_config, LIGHTGBM_MODELS
from models.model_factory import get_model, create_model_pipeline
from models.lightgbm_model import LightGBMForecastModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_sample_data(rows=100, frequency='D'):
    """Generate sample time series data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=rows, freq=frequency)
    
    # Create a trend component
    trend = np.linspace(0, 10, rows)
    
    # Create a seasonal component
    seasonal_period = 7 if frequency == 'D' else 12  # Weekly for days, yearly for months
    seasonal = 5 * np.sin(2 * np.pi * np.arange(rows) / seasonal_period)
    
    # Add some noise
    noise = np.random.normal(0, 1, rows)
    
    # Combine components
    y = trend + seasonal + noise
    
    # Create features
    X = pd.DataFrame({
        'date': dates,
        'trend': trend,
        'seasonal': seasonal,
        'lag1': np.concatenate([np.array([0]), y[:-1]]),
        'lag2': np.concatenate([np.array([0, 0]), y[:-2]]),
        'lag3': np.concatenate([np.array([0, 0, 0]), y[:-3]]),
    })
    
    # Create target
    y_series = pd.Series(y, name='target')
    
    return X, y_series

def test_lightgbm_factory():
    """Test that LightGBM models can be created through the factory."""
    logger.info("Testing LightGBM model creation through factory...")
    
    # Try to create each LightGBM model type
    for model_type in LIGHTGBM_MODELS:
        try:
            model = get_model(model_type)
            logger.info(f"✅ Successfully created {model_type} model")
            
            # Check model is correct type
            assert isinstance(model, LightGBMForecastModel), f"Expected LightGBMForecastModel, got {type(model)}"
            logger.info(f"✅ Model is correct type: {type(model).__name__}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create {model_type} model: {e}")
            raise

def test_lightgbm_training():
    """Test that LightGBM models can be trained and make predictions."""
    logger.info("Testing LightGBM model training and prediction...")
    
    # Generate sample data
    X, y = generate_sample_data(rows=100)
    
    # Split into train and test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Create and train model
    try:
        model = get_model('lightgbm')
        model.fit(X_train.drop('date', axis=1), y_train)
        logger.info("✅ Successfully trained LightGBM model")
        
        # Make predictions
        y_pred = model.predict(X_test.drop('date', axis=1))
        logger.info(f"✅ Successfully made predictions: {len(y_pred)} values")
        
        # Check predictions shape
        assert len(y_pred) == len(y_test), f"Expected {len(y_test)} predictions, got {len(y_pred)}"
        
        # Get feature importance
        importance = model.get_feature_importance()
        logger.info(f"✅ Feature importances: {importance}")
        
    except Exception as e:
        logger.error(f"❌ Error during training or prediction: {e}")
        raise

def test_model_pipeline():
    """Test creating a pipeline with LightGBM models."""
    logger.info("Testing model pipeline with LightGBM...")
    
    # Create pipeline with multiple models
    try:
        pipeline = create_model_pipeline(['linear_regression', 'lightgbm', 'random_forest'])
        logger.info(f"✅ Successfully created pipeline with {len(pipeline)} models")
        
        # Check model types
        models_by_type = {}
        for name, model in pipeline.items():
            model_type = type(model).__name__
            models_by_type[model_type] = models_by_type.get(model_type, 0) + 1
            
        logger.info(f"Models in pipeline: {models_by_type}")
        
        # Verify LightGBM model is in pipeline
        assert any(isinstance(model, LightGBMForecastModel) for model in pipeline.values()), \
            "No LightGBMForecastModel found in pipeline"
        logger.info("✅ LightGBM model found in pipeline")
        
    except Exception as e:
        logger.error(f"❌ Error creating model pipeline: {e}")
        raise

def run_all_tests():
    """Run all tests."""
    logger.info("Running all LightGBM integration tests...")
    
    try:
        test_lightgbm_factory()
        test_lightgbm_training()
        test_model_pipeline()
        
        logger.info("✅ All tests passed!")