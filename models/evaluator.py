"""
Model evaluation utilities.
"""
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Utilities for evaluating model performance.
    """
    
    @staticmethod
    def calculate_metrics(
        y_true: Union[pd.Series, np.ndarray], 
        y_pred: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Convert to numpy arrays if needed
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage
        }
        
        return metrics
    
    @staticmethod
    def compare_models(
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        metrics: List[str] = ['rmse', 'mae', 'r2', 'mape']
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same test data.
        
        Args:
            models: Dictionary mapping model names to model objects
            X_test: Test feature matrix
            y_test: Test target vector
            metrics: List of metrics to calculate
            
        Returns:
            DataFrame with model comparison
        """
        results = []
        
        for name, model in models.items():
            # Generate predictions
            try:
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                model_metrics = ModelEvaluator.calculate_metrics(y_test, y_pred)
                
                # Keep only requested metrics
                model_results = {'model': name}
                for metric in metrics:
                    if metric in model_metrics:
                        model_results[metric] = model_metrics[metric]
                
                results.append(model_results)
            except Exception as e:
                logger.error(f"Error evaluating model {name}: {e}")
        
        # Create DataFrame from results
        comparison_df = pd.DataFrame(results)
        
        # Set model as index
        if 'model' in comparison_df.columns:
            comparison_df.set_index('model', inplace=True)
        
        return comparison_df
    
    @staticmethod
    def plot_predictions(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        dates: Optional[Union[pd.DatetimeIndex, List[str]]] = None,
        title: str = 'Model Predictions vs Actual',
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot predicted vs actual values.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            dates: Date index for time series plots
            title: Plot title
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if dates is not None:
            # Plot as time series
            ax.plot(dates, y_true, label='Actual', marker='o', markersize=4)
            ax.plot(dates, y_pred, label='Predicted', marker='x', markersize=4)
            ax.set_xlabel('Date')
        else:
            # Plot as sequence
            ax.plot(y_true, label='Actual', marker='o', markersize=4)
            ax.plot(y_pred, label='Predicted', marker='x', markersize=4)
            ax.set_xlabel('Index')
        
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        # Calculate metrics to show on plot
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
        metrics_text = f"RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.2f}"
        
        # Add metrics as text box
        ax.text(
            0.05, 0.05, metrics_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_residuals(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        dates: Optional[Union[pd.DatetimeIndex, List[str]]] = None,
        title: str = 'Residual Analysis',
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        Plot residual analysis.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            dates: Date index for time series plots
            title: Plot title
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure object
        """
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Residuals over time/index
        if dates is not None:
            axes[0, 0].plot(dates, residuals, 'o', markersize=4)
            axes[0, 0].axhline(y=0, color='r', linestyle='-')
            axes[0, 0].set_xlabel('Date')
        else:
            axes[0, 0].plot(residuals, 'o', markersize=4)
            axes[0, 0].axhline(y=0, color='r', linestyle='-')
            axes[0, 0].set_xlabel('Index')
        
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].grid(True)
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=20, edgecolor='black')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Histogram of Residuals')
        axes[0, 1].grid(True)
        
        # Predicted vs Residuals
        axes[1, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='-')
        axes[1, 0].set_xlabel('Predicted Value')
        axes[1, 0].set_ylabel('Residual')
        axes[1, 0].set_title('Predicted vs Residuals')
        axes[1, 0].grid(True)
        
        # QQ plot of residuals
        from scipy import stats
        stats.probplot(residuals, plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        axes[1, 1].grid(True)
        
        # Add overall title
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        return fig
    
    @staticmethod
    def plot_feature_importance(
        model: Any,
        feature_names: Optional[List[str]] = None,
        title: str = 'Feature Importance',
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            model: Trained model with feature_importances_ or coef_ attribute
            feature_names: Names of features
            title: Plot title
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure object
        """
        # Try to get feature importance from model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        elif hasattr(model, 'get_feature_importance'):
            # For our wrapper classes
            importances_dict = model.get_feature_importance()
            feature_names = list(importances_dict.keys())
            importances = np.array(list(importances_dict.values()))
        else:
            raise ValueError("Model does not have feature_importances_ or coef_ attribute")
        
        # If feature names not provided, use generic names
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort features by importance
        indices = np.argsort(importances)
        sorted_names = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]
        
        # Plot horizontal bar chart
        ax.barh(sorted_names, sorted_importances)
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.grid(True, axis='x')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def forecast_evaluation(
        future_df: pd.DataFrame,
        target_col: str,
        forecast_col: str = 'forecast',
        date_col: str = 'date',
        title: str = 'Forecast Evaluation',
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Evaluate and plot forecasts.
        
        Args:
            future_df: DataFrame with forecasts and actual values
            target_col: Name of target column with actual values
            forecast_col: Name of forecast column
            date_col: Name of date column
            title: Plot title
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot actual values
        ax.plot(
            future_df[date_col], 
            future_df[target_col], 
            label='Actual', 
            marker='o', 
            markersize=4
        )
        
        # Plot forecast
        ax.plot(
            future_df[date_col], 
            future_df[forecast_col], 
            label='Forecast', 
            marker='x', 
            markersize=4
        )
        
        # Plot confidence intervals if available
        if 'lower_bound' in future_df.columns and 'upper_bound' in future_df.columns:
            ax.fill_between(
                future_df[date_col],
                future_df['lower_bound'],
                future_df['upper_bound'],
                alpha=0.2,
                label='95% Confidence Interval'
            )
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        # Format x-axis date labels
        fig.autofmt_xdate()
        
        # Calculate metrics
        metrics = ModelEvaluator.calculate_metrics(
            future_df[target_col], future_df[forecast_col]
        )
        
        # Add metrics text
        metrics_text = (
            f"RMSE: {metrics['rmse']:.2f}\n"
            f"MAE: {metrics['mae']:.2f}\n"
            f"MAPE: {metrics['mape']:.2f}%\n"
            f"R²: {metrics['r2']:.2f}"
        )
        
        ax.text(
            0.02, 0.95, metrics_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        return fig