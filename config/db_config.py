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
        'frequency': 'monthly'
    },
    'bankruptcies': {
        'table': 'bankruptcies',
        'date_col': 'date',
        'value_col': 'filings',
        'frequency': 'monthly'
    },
    'cement_production': {
        'table': 'cement_production',
        'date_col': 'date',
        'value_col': 'production',
        'frequency': 'monthly'
    },
    'electricity_consumption': {
        'table': 'electricity_consumption',
        'date_col': 'date',
        'value_col': 'consumption',
        'frequency': 'monthly'
    },
    'gas_price': {
        'table': 'gas_price',
        'date_col': 'date',
        'value_col': 'price',
        'frequency': 'monthly'
    },
    'gas_consumption': {
        'table': 'gas_consumption',
        'date_col': 'date',
        'value_col': 'consumption',
        'frequency': 'monthly'
    },
    'labor_participation': {
        'table': 'labor_participation',
        'date_col': 'date',
        'value_col': 'rate',
        'frequency': 'monthly'
    },
    'unemployment_rate': {
        'table': 'unemployment_rate',
        'date_col': 'date',
        'value_col': 'rate',
        'frequency': 'monthly'
    },
    'employment_rate': {
        'table': 'employment_rate',
        'date_col': 'date',
        'value_col': 'rate',
        'frequency': 'monthly'
    },
    'unemployment_claims': {
        'table': 'unemployment_claims',
        'date_col': 'date',
        'value_col': 'claims',
        'frequency': 'monthly'
    },
    'trade_employment': {
        'table': 'trade_employment',
        'date_col': 'date',
        'value_col': 'employment',
        'frequency': 'monthly'
    },
    'consumer_price_index': {
        'table': 'consumer_price_index',
        'date_col': 'date',
        'value_col': 'index',
        'frequency': 'monthly'
    },
    'transportation_price_index': {
        'table': 'transportation_price_index',
        'date_col': 'date',
        'value_col': 'index',
        'frequency': 'monthly'
    },
    # 'retail_sales': {
    #     'table': 'retail_sales',
    #     'date_col': 'date',
    #     'value_col': 'sales',
    #     'frequency': 'monthly'
    # },
    'imports': {
        'table': 'imports',
        'date_col': 'date',
        'value_col': 'value',
        'frequency': 'monthly'
    },
    # 'commercial_banking': {
    #     'table': 'commercial_banking',
    #     'date_col': 'date',
    #     'value_col': 'individual_loans',
    #     'frequency': 'quarterly'
    # },
    
    # FRED API datasets
    'federal_funds_rate': {
        'table': 'federal_funds_rate',
        'date_col': 'date',
        'value_col': 'rate',
        'frequency': 'monthly'
    },
    'auto_manufacturing_orders': {
        'table': 'auto_manufacturing_orders',
        'date_col': 'date',
        'value_col': 'orders',
        'frequency': 'monthly'
    },
    'used_car_retail_sales': {
        'table': 'used_car_retail_sales',
        'date_col': 'date',
        'value_col': 'sales',
        'frequency': 'monthly'
    },
    'domestic_auto_inventories': {
        'table': 'domestic_auto_inventories',
        'date_col': 'date',
        'value_col': 'inventories',
        'frequency': 'monthly'
    },
    'domestic_auto_production': {
        'table': 'domestic_auto_production',
        'date_col': 'date',
        'value_col': 'production',
        'frequency': 'monthly'
    },
    'liquidty_credit_facilities': {
        'table': 'liquidty_credit_facilities',
        'date_col': 'date',
        'value_col': 'facilities',
        'frequency': 'monthly'
    },
    'semiconductor_manufacturing_units': {
        'table': 'semiconductor_manufacturing_units',
        'date_col': 'date',
        'value_col': 'units',
        'frequency': 'monthly'
    },
    'aluminum_new_orders': {
        'table': 'aluminum_new_orders',
        'date_col': 'date',
        'value_col': 'orders',
        'frequency': 'monthly'
    },
    'real_gdp': {
        'table': 'real_gdp',
        'date_col': 'date',
        'value_col': 'growth',
        'frequency': 'quarterly'
    },
    'gdp_now_forecast': {
        'table': 'gdp_now_forecast',
        'date_col': 'date',
        'value_col': 'forecast',
        'frequency': 'quarterly'
    },

    # NYU Stern dataset
    'equity_risk_premium': {
        'table': 'equity_risk_premium',
        'date_col': 'date',
        'value_cols': ['tbond_rate', 'erp_sustainable', 'erp_t12m'],
        'frequency': 'monthly'
    }
}

# Default date range for fetching data
DEFAULT_START_DATE = "2014-01-01"
DEFAULT_END_DATE = None  # None means fetch up to the latest available data