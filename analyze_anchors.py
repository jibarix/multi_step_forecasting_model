import pandas as pd
import logging
from data import DataConnector
from models.gdp_projection_handler import GDPProjectionHandler, parse_gdp_input
from models.anchor_selection import AnchorVariableSelector
from dimension_reduction.pca_handler import PCAHandler
from dimension_reduction.feature_clustering import cluster_gdp_related_features
from data.multi_frequency import MultiFrequencyHandler

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize components
connector = DataConnector()
gdp_handler = GDPProjectionHandler(gdp_column='real_gdp')
multi_freq_handler = MultiFrequencyHandler()
anchor_selector = AnchorVariableSelector()

# 1. Load datasets
gdp_data = connector.fetch_dataset('real_gdp')
feature_datasets = []
for dataset_name in ['unemployment_rate', 'retail_sales', 'consumer_price_index', 
                   'gas_price', 'federal_funds_rate', 'imports']:
    df = connector.fetch_dataset(dataset_name)
    feature_datasets.append((dataset_name, df))

# 2. Convert quarterly GDP to monthly
gdp_monthly = multi_freq_handler.disaggregate_simple(
    gdp_data, 
    date_col='date', 
    target_frequency='MS',
    method='cubic'
)

# Join with other datasets
all_data = gdp_monthly.set_index('date')
for name, df in feature_datasets:
    if 'date' in df.columns:
        # Get the value column based on dataset configuration
        from config.db_config import DATASETS
        value_col = DATASETS[name]['value_col']
        # Keep only the value column and rename it
        df_aligned = df.set_index('date')[[value_col]]
        df_aligned = df_aligned.rename(columns={value_col: name})
        # Join without overlapping columns
        all_data = all_data.join(df_aligned, how='outer')

# 3. Perform anchor variable selection
print("\n=== SELECTED ANCHOR VARIABLES ===")
optimal_anchors = anchor_selector.get_optimal_anchor_combination(
    all_data, 'real_gdp', max_anchors=5
)
print(f"Selected anchors: {optimal_anchors}")

# 4. Process GDP projections
gdp_projections = "2025.1:-1.8, 2025.2:2.0, 2025.3:2.2, 2025.4:2.5"
projections_df, proj_desc = parse_gdp_input(gdp_projections)
print(f"\n=== GDP PROJECTIONS ===\n{proj_desc}")

# 5. Create GDP-to-anchor models
from sklearn.linear_model import LinearRegression
anchor_models = {}
anchor_forecasts = {}

for anchor in optimal_anchors:
    # Prepare data
    X = all_data[['real_gdp']]
    y = all_data[anchor]
    
    # Create and fit model
    model = LinearRegression()
    model.fit(X, y)
    anchor_models[anchor] = model
    
    print(f"\nModel for {anchor}:")
    print(f"  Coefficient: {model.coef_[0]:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")

# 6. Generate anchor forecasts from GDP projections
# Convert quarterly projections to monthly
gdp_handler.parse_projections(gdp_projections)
monthly_gdp = gdp_handler.get_monthly_projections()

# Create dataframe with monthly GDP projections
forecast_data = pd.DataFrame({
    'date': monthly_gdp['date'],
    'real_gdp': monthly_gdp['gdp_growth']
})

# Generate forecasts for each anchor
for anchor, model in anchor_models.items():
    forecast_data[anchor] = model.predict(forecast_data[['real_gdp']])

print("\n=== ANCHOR FORECASTS ===")
print(forecast_data.head())

# Save results
forecast_data.to_csv('tier1_anchor_forecasts.csv', index=False)
print("\nFull Tier 1 forecast saved to 'tier1_anchor_forecasts.csv'")