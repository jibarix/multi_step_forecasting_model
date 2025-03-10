# Economic Forecasting Project

## Overview
This project provides a comprehensive economic forecasting system focused on predicting auto sales in Puerto Rico using various macroeconomic indicators. The system integrates data acquisition, preprocessing, model training, evaluation, and visualization in a complete pipeline.

## Features
- **Data Acquisition**: Connect to Supabase database to fetch economic datasets
- **Data Preprocessing**: Handle missing values, create lag features, calculate rolling statistics
- **Model Training**: Support for multiple model types:
  - Linear models (Linear Regression, Ridge, Lasso, Elastic Net)
  - Time series models (ARIMA, SARIMA, Prophet, ETS)
  - Ensemble models (Random Forest, Gradient Boosting, XGBoost)
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Forecasting**: Generate predictions for future periods
- **Visualization**: Create time series plots, correlation matrices, residual analysis
- **Dashboards**: Interactive dashboards for comparing models and visualizing forecasts

## Prerequisites
- Python 3.8+
- Supabase account with database connection details
- Required Python packages (see requirements.txt)

## Setup

### Installation
1. Clone the repository
```bash
git clone <repository-url>
cd forecasting_project
```

python.exe -m pip install --upgrade pip

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your Supabase credentials:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

## Usage

### Command Line Interface
The main entry point is `main.py`, which provides a comprehensive CLI:

```bash
python main.py [--target TARGET] [--features FEATURES [FEATURES ...]]
               [--start_date START_DATE] [--end_date END_DATE]
               [--forecast_horizon FORECAST_HORIZON]
               [--model_types MODEL_TYPES [MODEL_TYPES ...]]
               [--output_dir OUTPUT_DIR]
               [--mode {train,forecast,evaluate,compare,dashboard,all}]
               [--feature_lag FEATURE_LAG]
               [--test_size TEST_SIZE]
               [--scale_data]
               [--save_model]
               [--verbose]
```

### Example Commands

#### Linear model testing
```bash
python main.py --target auto_sales --mode all --save_model --model_types linear_regression lasso ridge elastic_net
```

#### Time series model testing
```bash
python main.py --target auto_sales --mode all --save_model --model_types arima sarima prophet ets
```

#### Ensemble models testing
```bash
python main.py --target auto_sales --mode all --save_model --model_types random_forest gradient_boosting xgboost
```

#### Neural network models testing
```bash
python main.py --target auto_sales --mode all --save_model --model_types mlp lstm
```

#### Train a single model
```bash
python main.py --target auto_sales --features unemployment_rate gas_price --model_types linear_regression --mode train
```

#### Compare multiple models
```bash
python main.py --target auto_sales --features unemployment_rate gas_price labor_participation federal_funds_rate --model_types linear_regression ridge random_forest prophet --mode compare
```

#### Generate forecast
```bash
python main.py --target auto_sales --features unemployment_rate gas_price --model_types prophet --forecast_horizon 12 --mode forecast
```

#### Create dashboard
```bash
python main.py --target auto_sales --features unemployment_rate gas_price labor_participation --model_types linear_regression prophet random_forest --mode dashboard
```

#### Run complete pipeline
```bash
python main.py --target auto_sales --features unemployment_rate gas_price labor_participation federal_funds_rate retail_sales --model_types linear_regression prophet random_forest --mode all
```

## Available Data Sources
The system can access various economic datasets from Puerto Rico's Economic Development Bank (EDB) and Federal Reserve Economic Data (FRED) other sources:

### Puerto Rico Datasets
- auto_sales
- bankruptcies
- cement_production
- electricity_consumption
- gas_price
- gas_consumption
- labor_participation
- unemployment_rate
- employment_rate
- unemployment_claims
- trade_employment
- consumer_price_index
- transportation_price_index
- retail_sales
- imports
- commercial_banking

### US Macroeconomic Datasets
- federal_funds_rate
- auto_loan_rate
- used_car_retail_sales
- domestic_auto_inventories
- domestic_auto_production
- liquidty_credit_facilities
- semiconductor_manufacturing_units
- aluminum_new_orders
- equity_risk_premium

## Project Structure
```
forecasting_project/
├── .env                        # Environment variables (SUPABASE credentials)
├── config/                     # Configuration modules
│   ├── __init__.py
│   ├── db_config.py            # Database and dataset configurations
│   └── model_config.py         # Model configurations and hyperparameters
├── data/                       # Data handling modules
│   ├── __init__.py
│   ├── connectors.py           # Database connection utilities
│   ├── data_utils.py           # Data manipulation utilities
│   └── preprocessor.py         # Data preprocessing functionality
├── models/                     # Model implementations
│   ├── __init__.py
│   ├── linear_models.py        # Linear regression models
│   ├── time_series_models.py   # ARIMA, Prophet models
│   ├── ensemble_models.py      # Random forest, gradient boosting models
│   ├── model_factory.py        # Automatically generate predictive models
│   └── evaluator.py            # Model evaluation utilities
├── visualizations/             # Visualization utilities
│   ├── __init__.py
│   └── plots.py                # Plotting functions
├── outputs/                    # Directory for model outputs and visualizations
│   ├── data/                   # Processed datasets and forecasts
│   ├── models/                 # Saved model files
│   ├── plots/                  # Generated visualizations
│   └── dashboard/              # Interactive dashboards
├── main.py                     # Main script to run the pipeline
└── requirements.txt            # Project dependencies
```

## Understanding the Outputs

### Data Outputs
- **Prepared data**: Processed dataframe with engineered features, saved in `outputs/data/prepared_data.csv`
- **Forecasts**: Future predictions from each model, saved in `outputs/data/{model_type}_forecast.csv`
- **Model comparison metrics**: Performance metrics for all models, saved in `outputs/data/model_comparison.csv`

### Visualization Outputs
- **Time series plots**: Historical data visualization, saved in `outputs/plots/{target_col}_time_series.png`
- **Correlation matrix**: Feature correlation heatmap, saved in `outputs/plots/correlation_matrix.png`
- **Residual analysis**: Diagnostic plots for model residuals, saved in `outputs/plots/{model_type}_residuals.png`
- **Predictions vs Actual**: Comparison of model predictions with actual values, saved in `outputs/plots/{model_type}_predictions.png`
- **Feature importance**: Visualization of feature importance, saved in `outputs/plots/{model_type}_feature_importance.png`
- **Forecast plots**: Visualization of future forecasts, saved in `outputs/plots/{model_type}_forecast.png`
- **Model comparison**: Comparative visualization of model performance, saved in `outputs/plots/model_comparison.png`

### Dashboard
An interactive dashboard combining all visualizations is saved in `outputs/dashboard/forecast_dashboard.html`

## Customizing Models
You can modify model hyperparameters in `config/model_config.py`:
- `DEFAULT_MODEL_CONFIGS`: Default parameters for each model type
- `PARAM_GRIDS`: Hyperparameter grids for model tuning

## Advanced Usage

### Creating Custom Feature Engineering
Extend `data/preprocessor.py` to implement custom feature engineering:

```python
def prepare_dataset(self, target_dataset, feature_datasets, ...):
    # Add custom feature engineering here
    # Example: Create interaction features
    df['price_x_interest'] = df['gas_price'] * df['federal_funds_rate']
```

### Implementing New Models
To add a new model type:

1. Add new model type in `config/model_config.py`
2. Implement the model class in an appropriate file in the `models/` directory
3. Update `models/model_factory.py` to include the new model type

## Troubleshooting

### Database Connection Issues
- Verify `.env` file contains correct Supabase credentials
- Check database connection with `connector.check_database_connection()`
- Ensure your IP address is whitelisted in Supabase

### Memory Issues with Large Datasets
- Limit the number of features with `--features` argument
- Reduce date range with `--start_date` and `--end_date` arguments
- Use lightweight models (linear models instead of ensemble models)

### Model Performance Issues
- Check for data quality issues (missing values, outliers)
- Adjust preprocessing parameters (lag periods, rolling windows)
- Try different model types or hyperparameters
- Use `--scale_data` flag to normalize features