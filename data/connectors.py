"""
Database connection and data retrieval functionality.
This revised version fetches data directly via table queries rather than via an RPC call,
and it handles None values for date parameters.
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime
from supabase import create_client, Client

from config.db_config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    DATASETS,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE
)

logger = logging.getLogger(__name__)


class DataConnector:
    """Handles database connections and data retrieval from Supabase."""

    def __init__(self):
        """Initialize the database connection."""
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        try:
            self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("Successfully connected to Supabase")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            raise

    def get_client(self) -> Client:
        """Return the Supabase client for direct access if needed."""
        return self.client

    def fetch_dataset(
        self, 
        dataset_name: str, 
        start_date: Optional[str] = DEFAULT_START_DATE,
        end_date: Optional[str] = DEFAULT_END_DATE
    ) -> pd.DataFrame:
        """
        Fetch a dataset from the database using direct table queries.
        
        Args:
            dataset_name: Name of the dataset as defined in db_config.DATASETS
            start_date: Start date in 'YYYY-MM-DD' format (defaults to DEFAULT_START_DATE if None)
            end_date: Optional end date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with the requested data.
        """
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(DATASETS.keys())}")

        # Ensure start_date is set to a valid value
        if start_date is None:
            start_date = DEFAULT_START_DATE

        dataset_config = DATASETS[dataset_name]
        table_name = dataset_config['table']
        date_col = dataset_config['date_col']

        try:
            # Build the query using direct table query methods.
            query = self.client.table(table_name).select("*").gte(date_col, start_date)
            if end_date:
                query = query.lte(date_col, end_date)
            response = query.execute()

            if response.data:
                df = pd.DataFrame(response.data)
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col])
                logger.info(f"Fetched {len(df)} records for dataset '{dataset_name}'")
                return df
            else:
                logger.warning(
                    f"No data found for dataset '{dataset_name}' for the date range {start_date} to {end_date or 'now'}"
                )
                # Return an empty DataFrame with expected columns.
                if "value_cols" in dataset_config:
                    columns = [date_col] + dataset_config["value_cols"]
                else:
                    columns = [date_col, dataset_config.get("value_col", "value")]
                return pd.DataFrame(columns=columns)
        except Exception as e:
            logger.error(f"Error fetching dataset '{dataset_name}': {e}")
            raise

    def fetch_multiple_datasets(
        self, 
        dataset_names: List[str], 
        start_date: Optional[str] = DEFAULT_START_DATE,
        end_date: Optional[str] = DEFAULT_END_DATE
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch multiple datasets and return them as a dictionary.
        
        Args:
            dataset_names: List of dataset names to fetch.
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: Optional end date in 'YYYY-MM-DD' format.
            
        Returns:
            Dictionary of DataFrames with dataset names as keys.
        """
        result = {}
        for name in dataset_names:
            try:
                result[name] = self.fetch_dataset(name, start_date, end_date)
            except Exception as e:
                logger.error(f"Error fetching dataset '{name}': {e}")
                result[name] = pd.DataFrame()
        return result

    def fetch_all_datasets(
        self, 
        start_date: Optional[str] = DEFAULT_START_DATE,
        end_date: Optional[str] = DEFAULT_END_DATE
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all available datasets.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: Optional end date in 'YYYY-MM-DD' format.
            
        Returns:
            Dictionary of DataFrames with dataset names as keys.
        """
        return self.fetch_multiple_datasets(list(DATASETS.keys()), start_date, end_date)

    def get_dataset_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get metadata for a specific dataset including date range and record count.
        
        Args:
            dataset_name: Name of the dataset.
            
        Returns:
            Dictionary with metadata.
        """
        try:
            df = self.fetch_dataset(dataset_name)
            if df.empty:
                logger.warning(f"No data available for metadata extraction for dataset '{dataset_name}'")
                return {}
            date_col = DATASETS[dataset_name]["date_col"]
            min_date = df[date_col].min()
            max_date = df[date_col].max()
            record_count = len(df)
            return {
                "min_date": min_date,
                "max_date": max_date,
                "record_count": record_count
            }
        except Exception as e:
            logger.error(f"Error retrieving metadata for dataset '{dataset_name}': {e}")
            return {}

    def get_all_dataset_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all datasets.
        
        Returns:
            Dictionary with dataset names as keys and metadata dictionaries as values
        """
        result = {}
        for name in DATASETS.keys():
            result[name] = self.get_dataset_metadata(name)
        return result
    
    def get_data_revisions(
        self, 
        dataset_name: str,
        limit: int = 10
    ) -> pd.DataFrame:
        """
        Get revision history for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            limit: Maximum number of revisions to return
            
        Returns:
            DataFrame with revision history
        """
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        sql = f"""
        SELECT * FROM data_revisions
        WHERE dataset = '{DATASETS[dataset_name]['table']}'
        ORDER BY revision_date DESC
        LIMIT {limit}
        """
        
        try:
            response = self.client.postgrest.rpc('exec_sql', {'query': sql}).execute()
            if not response.data:
                return pd.DataFrame(columns=[
                    'id', 'dataset', 'data_date', 'value_field', 
                    'old_value', 'new_value', 'revision_date'
                ])
            
            df = pd.DataFrame(response.data)
            # Convert date columns to datetime
            if 'data_date' in df.columns:
                df['data_date'] = pd.to_datetime(df['data_date'])
            if 'revision_date' in df.columns:
                df['revision_date'] = pd.to_datetime(df['revision_date'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching revision history for {dataset_name}: {e}")
            raise
    
    def check_database_connection(self) -> bool:
        """
        Check if the database connection is working.
        
        Returns:
            True if connection is working, False otherwise
        """
        try:
            # Simple query to check connection
            self.client.postgrest.rpc('exec_sql', {'query': 'SELECT 1'}).execute()
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False