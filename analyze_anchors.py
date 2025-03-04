import os
import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Any
from datetime import datetime

from data import DataConnector
from models.gdp_projection_handler import GDPProjectionHandler, parse_gdp_input
from models.anchor_selection import AnchorVariableSelector
from dimension_reduction.pca_handler import PCAHandler
from dimension_reduction.feature_clustering import cluster_gdp_related_features
from data.multi_frequency import MultiFrequencyHandler
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Try to import XGBoost, handle if not available
try:
    import xgboost as xgb
    has_xgboost = True
except ImportError:
    has_xgboost = False
    print("XGBoost not available, will skip XGBoost models")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('anchor_selection')

# Import configuration for dataset definitions
from config.db_config import DATASETS

def get_pd_freq(freq_str: str) -> str:
    """
    Map a human-readable frequency string to a Pandas offset alias.
    """
    mapping = {
        'monthly': 'MS',      # Month start
        'quarterly': 'QS',    # Quarter start
        'daily': 'D'
    }
    return mapping.get(freq_str.lower(), None)

def load_datasets(connector: DataConnector, datasets_config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[Tuple[str, pd.DataFrame]]]:
    """
    Load the GDP dataset and additional feature datasets (skipping GDPNOW).
    """
    logger.info("=== LOADING DATASETS ===")
    gdp_data = connector.fetch_dataset('real_gdp')
    logger.info(f"Loaded GDP data: {len(gdp_data)} records")

    feature_datasets = []
    for dataset_name in datasets_config.keys():
        # Skip GDPNOW data (and real_gdp is handled separately)
        if dataset_name in ['real_gdp', 'gdp_now_forecast', 'auto_sales']:
            continue
        try:
            df = connector.fetch_dataset(dataset_name)
            logger.info(f"Loaded {dataset_name}: {len(df)} records")
            feature_datasets.append((dataset_name, df))
        except Exception as e:
            logger.error(f"Error loading {dataset_name}: {e}")
    logger.info(f"Successfully loaded {len(feature_datasets)} feature datasets")
    return gdp_data, feature_datasets

def preprocess_gdp(gdp_data: pd.DataFrame, multi_freq_handler: MultiFrequencyHandler) -> pd.Series:
    """
    Disaggregate quarterly GDP data to monthly frequency and compute the compounded (annualized)
    growth rate from the 'growth' column.
    """
    logger.info("=== PREPROCESSING GDP DATA ===")
    value_col = 'growth'  # Using actual column name from data
    
    # Disaggregate quarterly GDP data to monthly using cubic interpolation
    gdp_monthly = multi_freq_handler.disaggregate_simple(
        gdp_data, 
        date_col='date', 
        target_frequency='MS',
        method='cubic'
    )
    # Convert 'date' to datetime, sort, and set as index with explicit monthly frequency
    gdp_monthly['date'] = pd.to_datetime(gdp_monthly['date'], errors='coerce')
    gdp_monthly = gdp_monthly.sort_values('date').set_index('date').asfreq('MS')
    
    # Convert the correct column to numeric
    gdp_monthly[value_col] = pd.to_numeric(gdp_monthly[value_col], errors='coerce')
    logger.info(f"Sample disaggregated GDP levels:\n{gdp_monthly[value_col].head()}")
    
    # Calculate compounded (annualized) growth rate using the quarterly compounding formula
    gdp_values = gdp_monthly[value_col]
    growth_rates = ((gdp_values / gdp_values.shift(1)) ** 4 - 1) * 100
    growth_rates = growth_rates.dropna()
    logger.info(f"Calculated {len(growth_rates)} growth rates")
    
    return growth_rates

def join_datasets(gdp_growth: pd.Series, feature_datasets: List[Tuple[str, pd.DataFrame]], 
                  datasets_config: Dict[str, Any], output_dir: str) -> pd.DataFrame:
    """
    Join the GDP growth data with additional feature datasets without filling missing values.
    """
    logger.info("=== JOINING DATASETS (RAW) ===")
    all_data = pd.DataFrame(index=gdp_growth.index)
    all_data['real_gdp'] = gdp_growth

    for name, df in feature_datasets:
        if 'date' in df.columns:
            try:
                if 'value_col' not in datasets_config[name]:
                    raise KeyError("Missing 'value_col' in configuration")
                value_col = datasets_config[name]['value_col']
                freq_str = datasets_config[name].get('frequency', 'monthly')
                pd_freq = get_pd_freq(freq_str)
                
                df_aligned = df.copy()
                df_aligned['date'] = pd.to_datetime(df_aligned['date'], errors='coerce')
                df_aligned = df_aligned.set_index('date')[[value_col]].rename(columns={value_col: name})
                if pd_freq:
                    df_aligned = df_aligned.asfreq(pd_freq)
                # Outer join preserves all dates from both series.
                all_data = all_data.join(df_aligned, how='outer')
                logger.info(f"Joined {name}: {len(df_aligned)} records")
            except Exception as e:
                logger.error(f"Error joining {name}: {e}")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(output_dir, f'preprocessed_raw_data_{timestamp}.csv')
    all_data.to_csv(filename)
    logger.info(f"Saved raw preprocessed data to {filename} with {len(all_data)} rows and {len(all_data.columns)} columns")
    return all_data

def fill_missing_values_with_moving_average(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Fill missing values in the DataFrame using a simple rolling average over the specified window.
    """
    logger.info("=== FILLING MISSING VALUES USING A 3-MONTH ROLLING AVERAGE ===")
    filled_df = df.copy()
    for col in filled_df.columns:
        filled_df[col] = filled_df[col].fillna(filled_df[col].rolling(window=window, min_periods=1).mean())
    return filled_df

def select_anchor_variables(all_data: pd.DataFrame, anchor_selector: AnchorVariableSelector) -> List[str]:
    """
    Select anchor variables using both a custom selection method and correlation analysis.
    """
    logger.info("=== SELECTING ANCHOR VARIABLES ===")
    optimal_anchors = anchor_selector.get_optimal_anchor_combination(all_data, 'real_gdp', max_anchors=5)
    logger.info(f"Selected optimal anchors: {optimal_anchors}")

    logger.info("Performing correlation analysis with GDP")
    correlations = {col: all_data['real_gdp'].corr(all_data[col]) for col in all_data.columns if col != 'real_gdp'}
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]) if pd.notnull(x[1]) else 0, reverse=True)
    corr_anchors = [item[0] for item in sorted_corr[:5]]
    logger.info(f"Top correlated features: {corr_anchors}")
    for name, corr in sorted_corr[:5]:
        logger.info(f"  {name}: {corr:.4f}" if pd.notnull(corr) else f"  {name}: NaN")
    
    combined_anchors = list(set(optimal_anchors + corr_anchors))
    logger.info(f"Combined anchor set: {combined_anchors}")
    return combined_anchors

def build_model_types() -> Dict[str, Any]:
    """
    Build and return a dictionary of model types to try.
    """
    model_types = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    if has_xgboost:
        try:
            model_types['XGBoost'] = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        except AttributeError:
            try:
                from xgboost import XGBRegressor
                model_types['XGBoost'] = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
            except (ImportError, AttributeError):
                logger.warning("XGBoost available but XGBRegressor not found, skipping")
    return model_types

def train_anchor_models(all_data: pd.DataFrame, combined_anchors: List[str], model_types: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train multiple models to predict each anchor from GDP and select the best model based on RMSE.
    """
    logger.info("=== TRAINING GDP-TO-ANCHOR MODELS ===")
    anchor_models = {}
    model_performance = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for anchor in combined_anchors:
        logger.info(f"\nTraining models for {anchor}:")
        X = all_data[['real_gdp']].dropna()
        y = all_data[anchor].loc[X.index]
        
        if y.isnull().all():
            logger.error(f"Skipping training for {anchor}: target variable is entirely NaN.")
            continue
        if y.isnull().any():
            logger.warning(f"Missing values found for {anchor}, dropping missing entries.")
            valid_idx = y.dropna().index
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
            if len(y) == 0:
                logger.error(f"Skipping training for {anchor}: no available data after dropping NaNs.")
                continue
        
        anchor_perf = {}
        for model_name, model in model_types.items():
            try:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('estimator', model)
                ])
                pipeline.fit(X, y)
                cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_squared_error')
                cv_rmse = np.sqrt(-cv_scores.mean())
                
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                anchor_perf[model_name] = {
                    'rmse': rmse,
                    'cv_rmse': cv_rmse,
                    'r2': r2,
                    'model': pipeline
                }
                logger.info(f"  {model_name}: RMSE = {rmse:.4f}, CV_RMSE = {cv_rmse:.4f}, R² = {r2:.4f}")
            except Exception as e:
                logger.error(f"  Error training {model_name} for {anchor}: {e}")
        
        if not anchor_perf:
            logger.error(f"Skipping {anchor} as no models could be trained due to missing data.")
            continue

        best_model_name = min(anchor_perf.items(), key=lambda x: x[1]['rmse'])[0]
        best_model = anchor_perf[best_model_name]['model']
        anchor_models[anchor] = best_model
        model_performance[anchor] = {
            'best_model': best_model_name,
            'metrics': anchor_perf[best_model_name],
            'all_models': anchor_perf
        }
        logger.info(f"Selected best model for {anchor}: {best_model_name}")
        if hasattr(best_model.named_steps['estimator'], 'coef_'):
            coef = best_model.named_steps['estimator'].coef_
            logger.info(f"  Coefficient: {coef[0]:.4f}")
        if hasattr(best_model.named_steps['estimator'], 'intercept_'):
            intercept = best_model.named_steps['estimator'].intercept_
            logger.info(f"  Intercept: {intercept:.4f}")
    
    if not anchor_models:
        logger.error("No anchor models were successfully trained.")
    
    return anchor_models, model_performance

def generate_anchor_forecasts(gdp_handler: GDPProjectionHandler, gdp_projections: str, 
                              anchor_models: Dict[str, Any], output_dir: str) -> pd.DataFrame:
    """Generate forecasts for anchor variables using GDP projections with equal quarterly distribution."""
    logger.info("=== GENERATING ANCHOR FORECASTS ===")
    
    # Parse the quarterly GDP projections
    quarterly_projections = gdp_handler.parse_projections(gdp_projections)
    
    # Create mapping of quarters to months
    quarters_to_months = {
        1: [1, 2, 3],
        2: [4, 5, 6],
        3: [7, 8, 9],
        4: [10, 11, 12]
    }
    
    # Create a dictionary for monthly GDP values
    monthly_gdp_dict = {}
    
    # For each quarterly projection, divide the value equally among its months
    for _, row in quarterly_projections.iterrows():
        year = row['year']
        quarter = row['quarter']
        gdp_value = row['value']
        
        # Divide quarterly value by 3 for each month
        monthly_value = gdp_value / 3
        
        for month in quarters_to_months[quarter]:
            date_key = pd.Timestamp(year=year, month=month, day=1)
            monthly_gdp_dict[date_key] = monthly_value
    
    # Create forecast dates and dataframe
    forecast_dates = sorted(monthly_gdp_dict.keys())
    forecast_data = pd.DataFrame({
        'date': forecast_dates,
        'real_gdp': [monthly_gdp_dict[date] for date in forecast_dates]
    })
    
    # Generate predictions for each anchor variable
    for anchor, model in anchor_models.items():
        try:
            predictions = model.predict(forecast_data[['real_gdp']])
            forecast_data[anchor] = predictions
            logger.info(f"Generated {anchor} forecast: values range from {predictions.min():.2f} to {predictions.max():.2f}")
        except Exception as e:
            logger.error(f"Error generating forecast for {anchor}: {e}")
            forecast_data[anchor] = np.nan
    
    # Format dates for output
    forecast_data['date'] = forecast_data['date'].dt.strftime('%Y-%m-%d')
    
    # Save forecast
    forecast_filename = os.path.join(output_dir, 'tier1_anchor_forecasts.csv')
    forecast_data.to_csv(forecast_filename, index=False)
    
    # Add debugging info
    logger.info("=== FORECAST DATA SAMPLE ===")
    logger.info(f"\n{forecast_data.head(2).to_string()}")
    logger.info(f"Forecast columns: {forecast_data.columns.tolist()}")
    
    return forecast_data

def train_anchor_models(all_data: pd.DataFrame, combined_anchors: List[str], model_types: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train multiple models to predict each anchor from GDP and select the best model based on RMSE.
    """
    logger.info("=== TRAINING GDP-TO-ANCHOR MODELS ===")
    anchor_models = {}
    model_performance = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for anchor in combined_anchors:
        logger.info(f"\nTraining models for {anchor}:")
        X = all_data[['real_gdp']].dropna()
        y = all_data[anchor].loc[X.index]
        
        if y.isnull().all():
            logger.error(f"Skipping training for {anchor}: target variable is entirely NaN.")
            continue
        if y.isnull().any():
            logger.warning(f"Missing values found for {anchor}, dropping missing entries.")
            valid_idx = y.dropna().index
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
            if len(y) == 0:
                logger.error(f"Skipping training for {anchor}: no available data after dropping NaNs.")
                continue
        
        anchor_perf = {}
        for model_name, model in model_types.items():
            try:
                # Create a fresh model clone for each anchor to avoid model contamination
                if hasattr(model, 'get_params'):
                    fresh_model = type(model)(**model.get_params())
                else:
                    # If the model doesn't support get_params, create a new instance
                    fresh_model = type(model)()
                
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('estimator', fresh_model)
                ])
                pipeline.fit(X, y)
                cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_squared_error')
                cv_rmse = np.sqrt(-cv_scores.mean())
                
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                anchor_perf[model_name] = {
                    'rmse': rmse,
                    'cv_rmse': cv_rmse,
                    'r2': r2,
                    'model': pipeline
                }
                logger.info(f"  {model_name}: RMSE = {rmse:.4f}, CV_RMSE = {cv_rmse:.4f}, R² = {r2:.4f}")
            except Exception as e:
                logger.error(f"  Error training {model_name} for {anchor}: {e}")
        
        if not anchor_perf:
            logger.error(f"Skipping {anchor} as no models could be trained due to missing data.")
            continue

        best_model_name = min(anchor_perf.items(), key=lambda x: x[1]['rmse'])[0]
        best_model = anchor_perf[best_model_name]['model']
        anchor_models[anchor] = best_model
        model_performance[anchor] = {
            'best_model': best_model_name,
            'metrics': anchor_perf[best_model_name],
            'all_models': anchor_perf
        }
        logger.info(f"Selected best model for {anchor}: {best_model_name}")
        
        # Debug predictions for each anchor
        sample_gdp = all_data['real_gdp'].median()
        sample_pred = best_model.predict([[sample_gdp]])[0]
        logger.info(f"  Test prediction for GDP={sample_gdp:.2f}: {anchor}={sample_pred:.2f}")
        
        if hasattr(best_model.named_steps['estimator'], 'coef_'):
            coef = best_model.named_steps['estimator'].coef_
            logger.info(f"  Coefficient: {coef[0]:.4f}")
        if hasattr(best_model.named_steps['estimator'], 'intercept_'):
            intercept = best_model.named_steps['estimator'].intercept_
            logger.info(f"  Intercept: {intercept:.4f}")
    
    if not anchor_models:
        logger.error("No anchor models were successfully trained.")
    
    return anchor_models, model_performance

def validate_anchor_models(anchor_models: Dict[str, Any], validation_data: pd.DataFrame) -> None:
    """
    Validate each anchor model with a common test input to ensure differentiation.
    """
    logger.info("=== VALIDATING ANCHOR MODELS ===")
    
    # Create a test input with a range of GDP values
    test_gdp_values = np.linspace(-2, 4, 5)  # Range from -2% to 4% GDP growth
    results = {}
    
    for gdp_value in test_gdp_values:
        test_input = pd.DataFrame({'real_gdp': [gdp_value]})
        
        # Get predictions for each anchor
        predictions = {}
        for anchor, model in anchor_models.items():
            try:
                pred = model.predict(test_input)[0]
                predictions[anchor] = pred
            except Exception as e:
                logger.error(f"Error validating model for {anchor}: {e}")
                predictions[anchor] = None
        
        results[gdp_value] = predictions
    
    # Display results in a table
    for gdp_value, preds in results.items():
        logger.info(f"For GDP growth {gdp_value:.2f}%:")
        for anchor, pred in preds.items():
            logger.info(f"  {anchor}: {pred}")
    
    # Check if predictions are identical
    unique_predictions = set()
    for anchor in anchor_models.keys():
        values = [results[gdp][anchor] for gdp in results if anchor in results[gdp]]
        prediction_pattern = tuple(values)
        unique_predictions.add(prediction_pattern)
    
    if len(unique_predictions) < len(anchor_models):
        logger.warning(f"WARNING: Only {len(unique_predictions)} unique prediction patterns for {len(anchor_models)} models!")
    else:
        logger.info(f"Validation successful: {len(unique_predictions)} unique prediction patterns for {len(anchor_models)} models.")

def plot_gdp_anchor_relationship(all_data: pd.DataFrame, anchor: str, anchor_models: Dict[str, Any], 
                                 model_performance: Dict[str, Any], output_dir: str) -> None:
    """
    Plot the relationship between real GDP and a specified anchor variable along with the regression line.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(all_data['real_gdp'], all_data[anchor], alpha=0.7)
    
    model = anchor_models[anchor]
    x_range = np.linspace(all_data['real_gdp'].min(), all_data['real_gdp'].max(), 100)
    x_range_df = pd.DataFrame({'real_gdp': x_range})
    y_pred = model.predict(x_range_df)
    plt.plot(x_range, y_pred, 'r-', linewidth=2)
    
    plt.xlabel('Real GDP Growth (%)')
    plt.ylabel(anchor)
    plt.title(f'Relationship between GDP Growth and {anchor}')
    best_model_name = model_performance[anchor]['best_model']
    r2 = model_performance[anchor]['metrics']['r2']
    plt.text(0.05, 0.95, f"Model: {best_model_name}\nR² = {r2:.4f}", transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.grid(True)
    plot_filename = os.path.join(output_dir, f'gdp_{anchor}_relationship.png')
    plt.savefig(plot_filename)
    logger.info(f"Saved GDP-{anchor} relationship plot to {plot_filename}")
    plt.close()

def plot_forecasts(forecast_data: pd.DataFrame, combined_anchors: List[str], output_dir: str) -> None:
    """
    Plot the forecasted anchor variables over time.
    """
    plt.figure(figsize=(12, 8))
    for anchor in combined_anchors:
        if anchor in forecast_data.columns:
            plt.plot(forecast_data['date'], forecast_data[anchor], marker='o', label=anchor)
    
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Tier 1 Anchor Forecasts based on GDP Projections')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, 'tier1_forecasts.png')
    plt.savefig(plot_filename)
    logger.info(f"Saved Tier 1 forecasts plot to {plot_filename}")
    plt.close()

def check_and_print_gdp_data(gdp_data):
    """Debug function to examine the GDP data structure"""
    logger.info("=== GDP DATA STRUCTURE CHECK ===")
    logger.info(f"Column names: {gdp_data.columns.tolist()}")
    logger.info(f"Sample data:\n{gdp_data.head().to_string()}")
    logger.info(f"Data types: {gdp_data.dtypes}")
    null_counts = gdp_data.isnull().sum()
    logger.info(f"Null value counts:\n{null_counts}")

def plot_model_differences(anchor_models: Dict[str, Any], output_dir: str) -> None:
    """
    Plot predictions from each model over a range of GDP values
    to visualize differences between models.
    """
    plt.figure(figsize=(12, 8))
    
    # Create a range of GDP values
    gdp_values = np.linspace(-4, 6, 100)  # Range from -4% to 6% GDP growth
    X_test = pd.DataFrame({'real_gdp': gdp_values})
    
    # Generate predictions for each anchor
    for anchor, model in anchor_models.items():
        try:
            predictions = model.predict(X_test)
            plt.plot(gdp_values, predictions, label=anchor)
        except Exception as e:
            logger.error(f"Error generating predictions for {anchor}: {e}")
    
    plt.xlabel('GDP Growth (%)')
    plt.ylabel('Predicted Value')
    plt.title('Anchor Model Predictions vs GDP Growth')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'model_prediction_curves.png'))
    plt.close()
    logger.info(f"Saved model prediction curves to {os.path.join(output_dir, 'model_prediction_curves.png')}")

def main():
    # Initialize components
    connector = DataConnector()
    gdp_handler = GDPProjectionHandler(gdp_column='real_gdp')
    multi_freq_handler = MultiFrequencyHandler()
    anchor_selector = AnchorVariableSelector()
    
    # Create output directory based on current datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'test_output/analyze_anchor_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    gdp_data, feature_datasets = load_datasets(connector, DATASETS)
    
    # Check GDP data structure
    check_and_print_gdp_data(gdp_data)
    
    # Preprocess GDP data
    gdp_growth = preprocess_gdp(gdp_data, multi_freq_handler)
    
    # Join GDP growth with raw feature datasets and save raw data
    raw_data = join_datasets(gdp_growth, feature_datasets, DATASETS, output_dir)
    
    # Fill missing values and save filled data
    filled_data = fill_missing_values_with_moving_average(raw_data, window=3)
    filled_csv_path = os.path.join(output_dir, 'preprocessed_filled_data.csv')
    filled_data.to_csv(filled_csv_path)
    logger.info(f"Saved filled data to {filled_csv_path}")
    
    # Select anchor variables
    combined_anchors = select_anchor_variables(filled_data, anchor_selector)
    
    # Parse GDP projections
    gdp_projections = "2025.1:-1.8, 2025.2:2.0, 2025.3:2.2, 2025.4:2.5"
    projections_df, proj_desc = parse_gdp_input(gdp_projections)
    logger.info(f"=== GDP PROJECTIONS ===\n{proj_desc}")
    
    # Build model types
    model_types = build_model_types()
    
    # Train models for each anchor
    anchor_models, model_performance = train_anchor_models(filled_data, combined_anchors, model_types)
    
    # Validate the trained models
    validate_anchor_models(anchor_models, filled_data)
    
    # Plot model differences to visualize variation
    plot_model_differences(anchor_models, output_dir)
    
    # Generate forecasts
    forecast_data = generate_anchor_forecasts(gdp_handler, gdp_projections, anchor_models, output_dir)
    
    # Create individual plots for each anchor
    for anchor in combined_anchors:
        if anchor in anchor_models:
            try:
                plot_gdp_anchor_relationship(filled_data, anchor, anchor_models, model_performance, output_dir)
            except Exception as e:
                logger.error(f"Error creating plot for {anchor}: {e}")
    
    # Create combined forecast plot
    plot_forecasts(forecast_data, combined_anchors, output_dir)
    
    logger.info("\nAnchor selection and forecasting process completed successfully!")

if __name__ == "__main__":
    main()