"""
GDP projection handler for economic forecasting.

This module processes user-provided GDP projections, validates them,
and performs temporal disaggregation to convert quarterly projections
to monthly values for use in the forecasting pipeline.
"""

import logging
import re
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from data.multi_frequency import MultiFrequencyHandler

# Set up logging
logger = logging.getLogger(__name__)

class GDPProjectionHandler:
    """
    Handler for processing and validating GDP projections.
    
    This class processes user-provided GDP projections in various formats,
    validates them for consistency, and converts quarterly projections
    to monthly values using temporal disaggregation methods.
    """
    
    def __init__(
        self,
        base_gdp_data: Optional[pd.DataFrame] = None,
        gdp_column: str = 'real_gdp',
        date_column: str = 'date',
        disaggregation_method: str = 'chow_lin'
    ):
        """
        Initialize the GDP projection handler.
        
        Args:
            base_gdp_data: Historical GDP data (quarterly)
            gdp_column: Name of GDP column in data
            date_column: Name of date column in data
            disaggregation_method: Method for temporal disaggregation
                ('chow_lin', 'denton', 'state_space')
        """
        self.base_gdp_data = base_gdp_data
        self.gdp_column = gdp_column
        self.date_column = date_column
        self.disaggregation_method = disaggregation_method
        
        # Initialize multi-frequency handler for temporal disaggregation
        self.multi_freq_handler = MultiFrequencyHandler()
        
        # Store the parsed projections
        self.quarterly_projections = None
        self.monthly_projections = None
        self.forecast_start_date = None
        self.forecast_end_date = None
    
    def parse_projections(
        self, 
        projections_input: str,
        base_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Parse user-provided GDP projections in the format '2025.1: -1.8, 2025.2: 2.0'.
        
        Args:
            projections_input: String with GDP projections
            base_date: Base date for relative projections (default: last date in base_gdp_data)
            
        Returns:
            DataFrame with parsed quarterly projections
        """
        # Determine base date if not provided
        if base_date is None:
            if self.base_gdp_data is not None and not self.base_gdp_data.empty:
                last_date = self.base_gdp_data[self.date_column].max()
                if pd.notna(last_date):
                    base_date = pd.to_datetime(last_date)
                else:
                    base_date = datetime.now()
            else:
                base_date = datetime.now()
        
        # Compile regex pattern for different formats
        # Format: '2025.1: -1.8, 2025.2: 2.0'
        year_quarter_pattern = re.compile(r'(\d{4})\.(\d)\s*:\s*(-?\d+\.?\d*)')
        
        # Format: 'Q1: -1.8, Q2: 2.0' (relative to base_date year)
        relative_quarter_pattern = re.compile(r'Q(\d)\s*:\s*(-?\d+\.?\d*)', re.IGNORECASE)
        
        # Format: '+1: -1.8, +2: 2.0' (quarters ahead of base_date)
        ahead_quarter_pattern = re.compile(r'\+(\d+)\s*:\s*(-?\d+\.?\d*)')
        
        # Extract projections using the patterns
        projections = []
        
        # Try year.quarter format
        year_quarter_matches = year_quarter_pattern.findall(projections_input)
        
        if year_quarter_matches:
            for year, quarter, value in year_quarter_matches:
                year = int(year)
                quarter = int(quarter)
                value = float(value)
                
                # Create date for the start of the quarter
                date = datetime(year, ((quarter-1) * 3) + 1, 1)
                
                projections.append({
                    'date': date,
                    'year': year,
                    'quarter': quarter,
                    'value': value
                })
        
        # Try relative quarter format
        elif relative_quarter_pattern.findall(projections_input):
            matches = relative_quarter_pattern.findall(projections_input)
            base_year = base_date.year
            if base_date.month > 9:  # If we're in Q4, relative quarters likely refer to next year
                base_year += 1
            
            for quarter, value in matches:
                quarter = int(quarter)
                value = float(value)
                
                # Create date for the start of the quarter
                date = datetime(base_year, ((quarter-1) * 3) + 1, 1)
                
                projections.append({
                    'date': date,
                    'year': base_year,
                    'quarter': quarter,
                    'value': value
                })
        
        # Try quarters ahead format
        elif ahead_quarter_pattern.findall(projections_input):
            matches = ahead_quarter_pattern.findall(projections_input)
            
            # Determine the current quarter of base_date
            current_quarter = (base_date.month - 1) // 3 + 1
            
            for quarters_ahead, value in matches:
                quarters_ahead = int(quarters_ahead)
                value = float(value)
                
                # Calculate target year and quarter
                total_quarters = current_quarter + quarters_ahead
                years_ahead = (total_quarters - 1) // 4
                target_quarter = ((total_quarters - 1) % 4) + 1
                target_year = base_date.year + years_ahead
                
                # Create date for the start of the quarter
                date = datetime(target_year, ((target_quarter-1) * 3) + 1, 1)
                
                projections.append({
                    'date': date,
                    'year': target_year,
                    'quarter': target_quarter,
                    'value': value
                })
        
        else:
            raise ValueError(
                "Invalid GDP projections format. Expected formats:\n"
                "  - Year.Quarter: '2025.1: -1.8, 2025.2: 2.0'\n"
                "  - Relative quarter: 'Q1: -1.8, Q2: 2.0'\n"
                "  - Quarters ahead: '+1: -1.8, +2: 2.0'"
            )
        
        # Convert to DataFrame and sort by date
        projections_df = pd.DataFrame(projections)
        projections_df = projections_df.sort_values('date')
        
        # Store projections and set forecast date range
        self.quarterly_projections = projections_df
        if not projections_df.empty:
            self.forecast_start_date = projections_df['date'].min()
            
            # End date is the end of the last projected quarter
            last_date = projections_df['date'].max()
            self.forecast_end_date = last_date + relativedelta(months=3) - relativedelta(days=1)
        
        return projections_df
    
    def validate_projections(self) -> Tuple[bool, str]:
        """
        Validate the parsed GDP projections for consistency and completeness.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.quarterly_projections is None or self.quarterly_projections.empty:
            return False, "No GDP projections provided"
        
        # Check for sequential quarters with no gaps
        dates = self.quarterly_projections['date'].dt.to_period('Q')
        quarters = self.quarterly_projections[['year', 'quarter']].apply(
            lambda x: x['year'] * 4 + x['quarter'], axis=1
        )
        
        # Check if quarters are sequential
        is_sequential = np.all(np.diff(quarters) == 1)
        if not is_sequential:
            return False, "GDP projections must be for sequential quarters with no gaps"
        
        # Check for reasonable GDP growth rates (warn but don't fail on extreme values)
        extreme_values = self.quarterly_projections[
            (self.quarterly_projections['value'] > 10) | 
            (self.quarterly_projections['value'] < -10)
        ]
        
        if not extreme_values.empty:
            extreme_desc = ", ".join([
                f"{row['year']}.{row['quarter']}: {row['value']:.1f}%"
                for _, row in extreme_values.iterrows()
            ])
            logger.warning(f"Extreme GDP growth projections detected: {extreme_desc}")
        
        return True, ""
    
    def get_monthly_projections(
        self,
        monthly_indicator: Optional[pd.DataFrame] = None,
        indicator_column: Optional[str] = None,
        indicator_date_column: Optional[str] = 'date'
    ) -> pd.DataFrame:
        """
        Convert quarterly GDP projections to monthly using temporal disaggregation.
        
        Args:
            monthly_indicator: Optional monthly indicator series related to GDP
            indicator_column: Name of indicator column
            indicator_date_column: Name of date column in indicator DataFrame
            
        Returns:
            DataFrame with monthly GDP projections
        """
        if self.quarterly_projections is None or self.quarterly_projections.empty:
            raise ValueError("No GDP projections available. Call parse_projections first.")
        
        # Create a proper quarterly DataFrame with date index
        quarterly_df = self.quarterly_projections.copy()
        quarterly_df[self.date_column] = quarterly_df['date']
        quarterly_df = quarterly_df.set_index(self.date_column)
        
        # Strip out metadata and keep only date and value
        quarterly_series = quarterly_df['value']
        
        # Perform temporal disaggregation
        if monthly_indicator is not None and indicator_column is not None:
            # Prepare monthly indicator
            monthly_indicator = monthly_indicator.copy()
            monthly_indicator = monthly_indicator.set_index(indicator_date_column)
            indicator_series = monthly_indicator[indicator_column]
            
            # Use indicator-based disaggregation
            monthly_values = self.multi_freq_handler.denton_disaggregation(
                quarterly_series,
                indicator_series,
                low_freq='QS',
                high_freq='MS'
            )
        else:
            # Use simple disaggregation without indicator
            monthly_df = self.multi_freq_handler.disaggregate_simple(
                quarterly_df[['value']],
                target_frequency='MS',
                method='cubic'
            )
            monthly_values = monthly_df['value']
        
        # Create output DataFrame
        result = pd.DataFrame({
            self.date_column: monthly_values.index,
            'gdp_growth': monthly_values.values
        })
        
        # Store monthly projections
        self.monthly_projections = result
        
        return result
    
    def get_forecast_dates(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get the start and end dates of the forecast period.
        
        Returns:
            Tuple of (start_date, end_date)
        """
        return self.forecast_start_date, self.forecast_end_date
    
    def get_forecast_horizon(self) -> Optional[int]:
        """
        Get the forecast horizon in months.
        
        Returns:
            Number of months in forecast horizon or None if no projections
        """
        if self.forecast_start_date is None or self.forecast_end_date is None:
            return None
        
        delta = relativedelta(self.forecast_end_date, self.forecast_start_date)
        return delta.years * 12 + delta.months + (1 if delta.days > 0 else 0)
    
    def append_to_historical(
        self,
        historical_data: pd.DataFrame, 
        projection_type: str = 'monthly'
    ) -> pd.DataFrame:
        """
        Append projections to historical GDP data.
        
        Args:
            historical_data: DataFrame with historical GDP data
            projection_type: Type of projections to append ('monthly' or 'quarterly')
            
        Returns:
            DataFrame with historical data and appended projections
        """
        if projection_type == 'monthly':
            if self.monthly_projections is None:
                # Generate monthly projections from quarterly if needed
                self.get_monthly_projections()
            
            projections = self.monthly_projections
            freq = 'MS'
        elif projection_type == 'quarterly':
            projections = self.quarterly_projections
            projections = projections.rename(columns={'date': self.date_column, 'value': 'gdp_growth'})
            projections = projections[[self.date_column, 'gdp_growth']]
            freq = 'QS'
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")
        
        # Ensure historical data has the right index and frequency
        historical = historical_data.copy()
        if self.date_column in historical.columns and not isinstance(historical.index, pd.DatetimeIndex):
            historical = historical.set_index(self.date_column)
        
        # Resample to ensure consistent frequency
        historical = historical.resample(freq).last().reset_index()
        
        # Combine and drop duplicates if any
        combined = pd.concat([historical, projections], ignore_index=True)
        combined = combined.drop_duplicates(subset=[self.date_column], keep='last')
        combined = combined.sort_values(self.date_column)
        
        return combined
    
    def convert_growth_to_levels(
        self, 
        growth_data: pd.DataFrame,
        base_level: Optional[float] = None,
        growth_column: str = 'gdp_growth',
        level_column: str = 'gdp_level',
        frequency: str = 'quarterly'
    ) -> pd.DataFrame:
        """
        Convert GDP growth rates to level values.
        
        Args:
            growth_data: DataFrame with GDP growth rates
            base_level: Base GDP level to start from (if None, set to 100)
            growth_column: Name of column with growth rates
            level_column: Name of column to store level values
            frequency: Data frequency ('quarterly', 'monthly', 'annual')
            
        Returns:
            DataFrame with original data and added level column
        """
        # Make a copy to avoid modifying input
        data = growth_data.copy()
        
        # Set default base level if not provided
        if base_level is None:
            base_level = 100.0
        
        # Determine growth rate adjustment based on frequency
        if frequency == 'quarterly':
            # Quarterly growth rates are typically annualized
            adjustment = 0.25  # Divide by 4 to get quarterly rate
        elif frequency == 'monthly':
            # Convert annualized growth to monthly
            adjustment = 1/12
        elif frequency == 'annual':
            adjustment = 1.0
        else:
            raise ValueError(f"Unknown frequency: {frequency}")
        
        # Initialize level column with base value
        data[level_column] = base_level
        
        # Convert growth rates to level values
        for i in range(1, len(data)):
            growth_rate = data.iloc[i-1][growth_column] * adjustment
            prev_level = data.iloc[i-1][level_column]
            data.iloc[i, data.columns.get_loc(level_column)] = prev_level * (1 + growth_rate/100)
        
        return data


def parse_gdp_input(
    gdp_input: str,
    base_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Utility function to parse GDP input and provide formatted description.
    
    Args:
        gdp_input: String with GDP projections
        base_date: Base date for relative projections
        
    Returns:
        Tuple of (projections_df, description_string)
    """
    handler = GDPProjectionHandler()
    projections = handler.parse_projections(gdp_input, base_date)
    
    # Create formatted description of the projections
    description = "GDP Projections:\n"
    
    for _, row in projections.iterrows():
        year = row['year']
        quarter = row['quarter']
        value = row['value']
        quarter_name = f"Q{quarter}"
        
        description += f"  {year} {quarter_name}: {value:+.1f}%\n"
    
    # Add information about forecast horizon
    start_date, end_date = handler.get_forecast_dates()
    if start_date and end_date:
        description += f"\nForecast period: {start_date.strftime('%b %Y')} to {end_date.strftime('%b %Y')}"
        
        horizon = handler.get_forecast_horizon()
        if horizon:
            description += f" ({horizon} months)"
    
    return projections, description