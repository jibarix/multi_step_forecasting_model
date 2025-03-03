"""
Plotting utilities for economic forecasting.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

def plot_time_series(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    date_col: str = 'date',
    title: str = 'Time Series Plot',
    figsize: Tuple[int, int] = (12, 6),
    backend: str = 'matplotlib'
) -> Union[Figure, go.Figure]:
    """
    Plot one or more time series from a DataFrame.
    
    Args:
        df: DataFrame containing time series data
        columns: List of columns to plot (None means all numeric columns)
        date_col: Name of date column
        title: Plot title
        figsize: Figure size (width, height) for matplotlib
        backend: Plotting backend ('matplotlib' or 'plotly')
    
    Returns:
        Figure object
    """
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # Make sure date column is datetime
    if date_col in data.columns and not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col])
    
    # If no columns specified, use all numeric columns except date
    if columns is None:
        numeric_cols = data.select_dtypes(include=['number']).columns
        columns = [col for col in numeric_cols if col != date_col]
    
    if backend == 'matplotlib':
        fig, ax = plt.subplots(figsize=figsize)
        
        for column in columns:
            if column in data.columns:
                ax.plot(data[date_col], data[column], marker='.', linestyle='-', label=column)
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        
        # Format date axis
        fig.autofmt_xdate()
        
        plt.tight_layout()
        return fig
        
    elif backend == 'plotly':
        fig = go.Figure()
        
        for column in columns:
            if column in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data[date_col],
                        y=data[column],
                        mode='lines+markers',
                        name=column
                    )
                )
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            legend_title='Series',
            template='plotly_white'
        )
        
        return fig
    
    else:
        raise ValueError(f"Unknown backend: {backend}")

def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = 'Correlation Matrix',
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'coolwarm',
    backend: str = 'matplotlib'
) -> Union[Figure, go.Figure]:
    """
    Plot correlation matrix for selected columns.
    
    Args:
        df: DataFrame containing data
        columns: List of columns to include in correlation (None means all numeric columns)
        title: Plot title
        figsize: Figure size (width, height) for matplotlib
        cmap: Colormap for matplotlib
        backend: Plotting backend ('matplotlib' or 'plotly')
    
    Returns:
        Figure object
    """
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # If no columns specified, use all numeric columns
    if columns is None:
        columns = data.select_dtypes(include=['number']).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = data[columns].corr()
    
    if backend == 'matplotlib':
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap=cmap,
            fmt='.2f',
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_title(title)
        plt.tight_layout()
        return fig
        
    elif backend == 'plotly':
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale='RdBu_r',
            labels=dict(color='Correlation'),
            title=title
        )
        
        # Add correlation values as text
        for i, row in enumerate(corr_matrix.index):
            for j, col in enumerate(corr_matrix.columns):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"{corr_matrix.iloc[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color='black' if abs(corr_matrix.iloc[i, j]) < 0.7 else 'white')
                )
        
        fig.update_layout(
            template='plotly_white',
            xaxis_title='',
            yaxis_title=''
        )
        
        return fig
    
    else:
        raise ValueError(f"Unknown backend: {backend}")

def plot_forecast(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    target_col: str,
    date_col: str = 'date',
    forecast_col: str = 'forecast',
    lower_bound_col: Optional[str] = 'lower_bound',
    upper_bound_col: Optional[str] = 'upper_bound',
    title: str = 'Forecast Plot',
    figsize: Tuple[int, int] = (12, 6),
    backend: str = 'matplotlib'
) -> Union[Figure, go.Figure]:
    """
    Plot historical data and forecast.
    
    Args:
        historical_df: DataFrame with historical data
        forecast_df: DataFrame with forecast data
        target_col: Name of target column in historical_df
        date_col: Name of date column
        forecast_col: Name of forecast column in forecast_df
        lower_bound_col: Name of lower bound column (optional)
        upper_bound_col: Name of upper bound column (optional)
        title: Plot title
        figsize: Figure size (width, height) for matplotlib
        backend: Plotting backend ('matplotlib' or 'plotly')
    
    Returns:
        Figure object
    """
    # Make copies to avoid modifying originals
    hist_data = historical_df.copy()
    fore_data = forecast_df.copy()
    
    # Make sure date columns are datetime
    if date_col in hist_data.columns and not pd.api.types.is_datetime64_any_dtype(hist_data[date_col]):
        hist_data[date_col] = pd.to_datetime(hist_data[date_col])
    
    if date_col in fore_data.columns and not pd.api.types.is_datetime64_any_dtype(fore_data[date_col]):
        fore_data[date_col] = pd.to_datetime(fore_data[date_col])
    
    if backend == 'matplotlib':
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot historical data
        ax.plot(
            hist_data[date_col],
            hist_data[target_col],
            marker='.',
            linestyle='-',
            color='blue',
            label='Historical'
        )
        
        # Plot forecast
        ax.plot(
            fore_data[date_col],
            fore_data[forecast_col],
            marker='.',
            linestyle='-',
            color='red',
            label='Forecast'
        )
        
        # Plot confidence intervals if available
        if lower_bound_col in fore_data.columns and upper_bound_col in fore_data.columns:
            ax.fill_between(
                fore_data[date_col],
                fore_data[lower_bound_col],
                fore_data[upper_bound_col],
                color='red',
                alpha=0.2,
                label='95% Confidence Interval'
            )
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        
        # Format date axis
        fig.autofmt_xdate()
        
        plt.tight_layout()
        return fig
        
    elif backend == 'plotly':
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(
            go.Scatter(
                x=hist_data[date_col],
                y=hist_data[target_col],
                mode='lines+markers',
                name='Historical',
                line=dict(color='blue')
            )
        )
        
        # Add forecast
        fig.add_trace(
            go.Scatter(
                x=fore_data[date_col],
                y=fore_data[forecast_col],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red')
            )
        )
        
        # Add confidence intervals if available
        if lower_bound_col in fore_data.columns and upper_bound_col in fore_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=fore_data[date_col],
                    y=fore_data[upper_bound_col],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=fore_data[date_col],
                    y=fore_data[lower_bound_col],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    name='95% Confidence Interval'
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            legend_title='Series',
            template='plotly_white'
        )
        
        return fig
    
    else:
        raise ValueError(f"Unknown backend: {backend}")

def plot_feature_importance(
    feature_importance: Dict[str, float],
    title: str = 'Feature Importance',
    figsize: Tuple[int, int] = (10, 8),
    color: str = 'skyblue',
    backend: str = 'matplotlib'
) -> Union[Figure, go.Figure]:
    """
    Plot feature importance.
    
    Args:
        feature_importance: Dictionary mapping feature names to importance scores
        title: Plot title
        figsize: Figure size (width, height) for matplotlib
        color: Bar color for matplotlib
        backend: Plotting backend ('matplotlib' or 'plotly')
    
    Returns:
        Figure object
    """
    # Sort features by importance
    sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1]))
    features = list(sorted_features.keys())
    importances = list(sorted_features.values())
    
    if backend == 'matplotlib':
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, color=color)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Display highest importance at top
        
        ax.set_title(title)
        ax.set_xlabel('Importance')
        
        plt.tight_layout()
        return fig
        
    elif backend == 'plotly':
        fig = px.bar(
            x=importances,
            y=features,
            orientation='h',
            title=title
        )
        
        fig.update_layout(
            xaxis_title='Importance',
            yaxis_title='',
            template='plotly_white'
        )
        
        return fig
    
    else:
        raise ValueError(f"Unknown backend: {backend}")

def plot_model_comparison(
    metrics_df: pd.DataFrame,
    metric_cols: Optional[List[str]] = None,
    title: str = 'Model Comparison',
    figsize: Tuple[int, int] = (12, 8),
    backend: str = 'matplotlib'
) -> Union[Figure, go.Figure]:
    """
    Plot model comparison based on evaluation metrics.
    
    Args:
        metrics_df: DataFrame with model names as index and metrics as columns
        metric_cols: List of metric columns to include (None means all columns)
        title: Plot title
        figsize: Figure size (width, height) for matplotlib
        backend: Plotting backend ('matplotlib' or 'plotly')
    
    Returns:
        Figure object
    """
    # Make a copy to avoid modifying original
    data = metrics_df.copy()
    
    # If no metric columns specified, use all columns
    if metric_cols is None:
        metric_cols = data.columns.tolist()
    
    if backend == 'matplotlib':
        # Create subplot grid based on number of metrics
        n_metrics = len(metric_cols)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Convert single axis to array for consistent indexing
        if n_metrics == 1:
            axes = np.array([axes])
        
        # Flatten axes array for easy iteration
        axes = axes.flatten()
        
        for i, metric in enumerate(metric_cols):
            if i < len(axes) and metric in data.columns:
                # Sort by metric value
                sorted_data = data.sort_values(by=metric)
                
                # Create bar chart
                sorted_data[metric].plot(
                    kind='barh',
                    ax=axes[i],
                    title=f"{metric.upper()}",
                    color='skyblue'
                )
                
                axes[i].grid(True)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        return fig
        
    elif backend == 'plotly':
        fig = make_subplots(
            rows=len(metric_cols),
            cols=1,
            subplot_titles=[m.upper() for m in metric_cols],
            vertical_spacing=0.1
        )
        
        for i, metric in enumerate(metric_cols):
            if metric in data.columns:
                # Sort by metric value
                sorted_data = data.sort_values(by=metric)
                
                # Add bar chart
                fig.add_trace(
                    go.Bar(
                        x=sorted_data[metric],
                        y=sorted_data.index,
                        orientation='h',
                        marker_color='skyblue',
                        showlegend=False
                    ),
                    row=i+1,
                    col=1
                )
        
        fig.update_layout(
            title=title,
            height=300 * len(metric_cols),
            template='plotly_white'
        )
        
        return fig
    
    else:
        raise ValueError(f"Unknown backend: {backend}")

def plot_residuals(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    dates: Optional[pd.Series] = None,
    title: str = 'Residual Analysis',
    figsize: Tuple[int, int] = (12, 10),
    backend: str = 'matplotlib'
) -> Union[Figure, List[Figure], go.Figure]:
    """
    Plot residual analysis.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        dates: Optional series of dates for time plot
        title: Plot title
        figsize: Figure size (width, height) for matplotlib
        backend: Plotting backend ('matplotlib' or 'plotly')
    
    Returns:
        Figure object or list of Figure objects
    """
    # Calculate residuals
    residuals = np.array(y_true) - np.array(y_pred)
    
    if backend == 'matplotlib':
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Residuals over time/index
        if dates is not None:
            axes[0, 0].plot(dates, residuals, 'o', color='blue', alpha=0.6)
            axes[0, 0].axhline(y=0, color='red', linestyle='--')
            axes[0, 0].set_xlabel('Date')
        else:
            axes[0, 0].plot(residuals, 'o', color='blue', alpha=0.6)
            axes[0, 0].axhline(y=0, color='red', linestyle='--')
            axes[0, 0].set_xlabel('Index')
        
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].grid(True)
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=20, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Histogram of Residuals')
        axes[0, 1].grid(True)
        
        # Predicted vs Residuals
        axes[1, 0].scatter(y_pred, residuals, color='blue', alpha=0.6)
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
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
        
    elif backend == 'plotly':
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                'Residuals Over Time',
                'Histogram of Residuals',
                'Predicted vs Residuals',
                'Q-Q Plot of Residuals'
            ]
        )
        
        # Residuals over time/index
        if dates is not None:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=residuals,
                    mode='markers',
                    marker=dict(color='blue', opacity=0.6),
                    showlegend=False
                ),
                row=1,
                col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    y=residuals,
                    mode='markers',
                    marker=dict(color='blue', opacity=0.6),
                    showlegend=False
                ),
                row=1,
                col=1
            )
        
        fig.add_trace(
            go.Scatter(
                x=[fig.layout.xaxis.range[0], fig.layout.xaxis.range[1]],
                y=[0, 0],
                mode='lines',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1,
            col=1
        )
        
        # Histogram of residuals
        fig.add_trace(
            go.Histogram(
                x=residuals,
                marker=dict(color='skyblue', line=dict(color='black', width=1)),
                showlegend=False
            ),
            row=1,
            col=2
        )
        
        # Add zero line for histogram
        fig.add_vline(
            x=0,
            line=dict(color='red', dash='dash'),
            row=1,
            col=2
        )
        
        # Predicted vs Residuals
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                marker=dict(color='blue', opacity=0.6),
                showlegend=False
            ),
            row=2,
            col=1
        )
        
        fig.add_hline(
            y=0,
            line=dict(color='red', dash='dash'),
            row=2,
            col=1
        )
        
        # Q-Q plot of residuals
        from scipy import stats
        qq = stats.probplot(residuals)
        theoretical_quantiles = qq[0][0]
        sample_quantiles = qq[0][1]
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                marker=dict(color='blue'),
                name='Sample Data',
                showlegend=False
            ),
            row=2,
            col=2
        )
        
        # Add the diagonal line
        min_val = min(theoretical_quantiles)
        max_val = max(theoretical_quantiles)
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=2,
            col=2
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=800,
            template='plotly_white'
        )
        
        return fig
    
    else:
        raise ValueError(f"Unknown backend: {backend}")

def create_dashboard(
    data_dict: Dict[str, pd.DataFrame],
    target_col: str,
    forecast_df: Optional[pd.DataFrame] = None,
    feature_importance: Optional[Dict[str, float]] = None,
    model_metrics: Optional[pd.DataFrame] = None,
    output_file: Optional[str] = None
) -> go.Figure:
    """
    Create a Plotly dashboard with multiple visualizations.
    
    Args:
        data_dict: Dictionary with dataset names as keys and DataFrames as values
        target_col: Target column name
        forecast_df: DataFrame with forecast data (optional)
        feature_importance: Dictionary with feature importance (optional)
        model_metrics: DataFrame with model comparison metrics (optional)
        output_file: Path to save HTML dashboard (optional)
    
    Returns:
        Plotly Figure object
    """
    # Determine how many rows we need based on available data
    num_rows = 1  # Time series plot is always included
    
    if forecast_df is not None:
        num_rows += 1
    
    if feature_importance is not None:
        num_rows += 1
    
    if model_metrics is not None:
        num_rows += 1
    
    # Create subplot grid
    fig = make_subplots(
        rows=num_rows,
        cols=1,
        subplot_titles=[
            'Historical Time Series',
            'Forecast' if forecast_df is not None else None,
            'Feature Importance' if feature_importance is not None else None,
            'Model Comparison' if model_metrics is not None else None
        ],
        vertical_spacing=0.1
    )
    
    # Row counter
    current_row = 1
    
    # Plot historical time series
    for name, df in data_dict.items():
        if target_col in df.columns and 'date' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df[target_col],
                    mode='lines+markers',
                    name=name
                ),
                row=current_row,
                col=1
            )
    
    # Increment row counter
    current_row += 1
    
    # Add forecast if available
    if forecast_df is not None:
        # Add historical data (using first dataset in data_dict)
        first_df = list(data_dict.values())[0]
        if target_col in first_df.columns and 'date' in first_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=first_df['date'],
                    y=first_df[target_col],
                    mode='lines+markers',
                    name='Historical'
                ),
                row=current_row,
                col=1
            )
        
        # Add forecast
        if 'date' in forecast_df.columns and 'forecast' in forecast_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['date'],
                    y=forecast_df['forecast'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='red')
                ),
                row=current_row,
                col=1
            )
            
            # Add confidence intervals if available
            if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df['date'],
                        y=forecast_df['upper_bound'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ),
                    row=current_row,
                    col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df['date'],
                        y=forecast_df['lower_bound'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        name='95% Confidence Interval'
                    ),
                    row=current_row,
                    col=1
                )
        
        # Increment row counter
        current_row += 1
    
    # Add feature importance if available
    if feature_importance is not None:
        # Sort features by importance
        sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        fig.add_trace(
            go.Bar(
                x=list(sorted_features.values()),
                y=list(sorted_features.keys()),
                orientation='h',
                marker_color='skyblue'
            ),
            row=current_row,
            col=1
        )
        
        # Increment row counter
        current_row += 1
    
    # Add model comparison if available
    if model_metrics is not None:
        for col in model_metrics.columns:
            # Sort by metric value
            sorted_metrics = model_metrics.sort_values(by=col)
            
            fig.add_trace(
                go.Bar(
                    x=sorted_metrics[col],
                    y=sorted_metrics.index,
                    orientation='h',
                    name=col,
                    marker_color=f'rgba({hash(col) % 256}, {(hash(col) // 256) % 256}, {(hash(col) // 65536) % 256}, 0.7)'
                ),
                row=current_row,
                col=1
            )
    
    # Update layout
    fig.update_layout(
        title='Economic Forecasting Dashboard',
        height=400 * num_rows,
        template='plotly_white'
    )
    
    # Save to HTML if output_file is provided
    if output_file is not None:
        try:
            fig.write_html(output_file)
            logger.info(f"Dashboard saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving dashboard to {output_file}: {e}")
    
    return fig