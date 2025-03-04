"""
Database configuration settings for the predictive model.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Supabase connection parameters
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Dataset configurations - mapping friendly names to database table names
DATASETS = {
    # Economic Development Bank (EDB) Puerto Rico datasets
    'auto_sales': {
        'table': 'auto_sales',
        'date_col': 'date',
        'value_col': 'sales',
        'rename_to': 'auto_sales_value',
        'frequency': 'monthly'
    },
    'bankruptcies': {
        'table': 'bankruptcies',
        'date_col': 'date',
        'value_col': 'filings',
        'rename_to': 'bankruptcies_value',
        'frequency': 'monthly'
    },
    'cement_production': {
        'table': 'cement_production',
        'date_col': 'date',
        'value_col': 'production',
        'rename_to': 'cement_production_value',
        'frequency': 'monthly'
    },
    'electricity_consumption': {
        'table': 'electricity_consumption',
        'date_col': 'date',
        'value_col': 'consumption',
        'rename_to': 'electricity_consumption_value',
        'frequency': 'monthly'
    },
    'gas_price': {
        'table': 'gas_price',
        'date_col': 'date',
        'value_col': 'price',
        'rename_to': 'gas_price_value',
        'frequency': 'monthly'
    },
    'gas_consumption': {
        'table': 'gas_consumption',
        'date_col': 'date',
        'value_col': 'consumption',
        'rename_to': 'gas_consumption_value',
        'frequency': 'monthly'
    },
    'labor_participation': {
        'table': 'labor_participation',
        'date_col': 'date',
        'value_col': 'rate',
        'rename_to': 'labor_participation_value',
        'frequency': 'monthly'
    },
    'unemployment_rate': {
        'table': 'unemployment_rate',
        'date_col': 'date',
        'value_col': 'rate',
        'rename_to': 'unemployment_rate_value',
        'frequency': 'monthly'
    },
    'employment_rate': {
        'table': 'employment_rate',
        'date_col': 'date',
        'value_col': 'rate',
        'rename_to': 'employment_rate_value',
        'frequency': 'monthly'
    },
    'unemployment_claims': {
        'table': 'unemployment_claims',
        'date_col': 'date',
        'value_col': 'claims',
        'rename_to': 'unemployment_claims_value',
        'frequency': 'monthly'
    },
    'trade_employment': {
        'table': 'trade_employment',
        'date_col': 'date',
        'value_col': 'employment',
        'rename_to': 'trade_employment_value',
        'frequency': 'monthly'
    },
    'consumer_price_index': {
        'table': 'consumer_price_index',
        'date_col': 'date',
        'value_col': 'index',
        'rename_to': 'cpi_value',
        'frequency': 'monthly'
    },
    'transportation_price_index': {
        'table': 'transportation_price_index',
        'date_col': 'date',
        'value_col': 'index',
        'rename_to': 'transport_price_index_value',
        'frequency': 'monthly'
    },
    'imports': {
        'table': 'imports',
        'date_col': 'date',
        'value_col': 'value',
        'rename_to': 'imports_value',
        'frequency': 'monthly'
    },
    
    # FRED API datasets
    'federal_funds_rate': {
        'table': 'federal_funds_rate',
        'date_col': 'date',
        'value_col': 'rate',
        'rename_to': 'fed_funds_rate_value',
        'frequency': 'monthly'
    },
    'auto_manufacturing_orders': {
        'table': 'auto_manufacturing_orders',
        'date_col': 'date',
        'value_col': 'orders',
        'rename_to': 'auto_orders_value',
        'frequency': 'monthly'
    },
    'used_car_retail_sales': {
        'table': 'used_car_retail_sales',
        'date_col': 'date',
        'value_col': 'sales',
        'rename_to': 'used_car_sales_value',
        'frequency': 'monthly'
    },
    'domestic_auto_inventories': {
        'table': 'domestic_auto_inventories',
        'date_col': 'date',
        'value_col': 'inventories',
        'rename_to': 'auto_inventory_value',
        'frequency': 'monthly'
    },
    'domestic_auto_production': {
        'table': 'domestic_auto_production',
        'date_col': 'date',
        'value_col': 'production',
        'rename_to': 'auto_production_value',
        'frequency': 'monthly'
    },
    'liquidty_credit_facilities': {
        'table': 'liquidty_credit_facilities',
        'date_col': 'date',
        'value_col': 'facilities',
        'rename_to': 'liquidity_facilities_value',
        'frequency': 'monthly'
    },
    'semiconductor_manufacturing_units': {
        'table': 'semiconductor_manufacturing_units',
        'date_col': 'date',
        'value_col': 'units',
        'rename_to': 'semiconductor_units_value',
        'frequency': 'monthly'
    },
    'aluminum_new_orders': {
        'table': 'aluminum_new_orders',
        'date_col': 'date',
        'value_col': 'orders',
        'rename_to': 'aluminum_orders_value',
        'frequency': 'monthly'
    },
    'real_gdp': {
        'table': 'real_gdp',
        'date_col': 'date',
        'value_col': 'growth',
        'rename_to': 'gdp_growth_value',
        'frequency': 'quarterly'
    },
    'gdp_now_forecast': {
        'table': 'gdp_now_forecast',
        'date_col': 'date',
        'value_col': 'forecast',
        'rename_to': 'gdp_forecast_value',
        'frequency': 'quarterly'
    },

    # NYU Stern dataset - split into separate virtual datasets
    'tbond_rate': {
        'table': 'equity_risk_premium',
        'date_col': 'date',
        'value_col': 'tbond_rate',
        'rename_to': 'tbond_rate_value',
        'frequency': 'monthly'
    },
    'erp_sustainable': {
        'table': 'equity_risk_premium',
        'date_col': 'date',
        'value_col': 'erp_sustainable',
        'rename_to': 'erp_sustainable_value',
        'frequency': 'monthly'
    },
    'erp_t12m': {
        'table': 'equity_risk_premium',
        'date_col': 'date',
        'value_col': 'erp_t12m',
        'rename_to': 'erp_t12m_value',
        'frequency': 'monthly'
    }
}

# Default date range for fetching data
DEFAULT_START_DATE = "2014-01-01"
DEFAULT_END_DATE = None  # None means fetch up to the latest available data