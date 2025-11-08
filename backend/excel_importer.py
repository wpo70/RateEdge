"""
Excel Import Utility for IRS Swap Rates
Handles importing data from Excel files into the database
Supports: AUD, NZD, USD, EUR, GBP, JPY
"""
import pandas as pd
from datetime import datetime
from database_models import DatabaseManager
import sys
import os

# Add parent directory to path to import currency_config
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from currency_config import get_fixing_reference, SUPPORTED_CURRENCIES


class ExcelImporter:
    """Handle Excel file imports"""
    
    def __init__(self, db_manager=None):
        """Initialize with database manager"""
        self.db_manager = db_manager or DatabaseManager()
    
    def import_from_excel(self, file_path, sheet_name=0, currency='AUD'):
        """
        Import data from Excel file
        
        Expected format (4 columns - legacy):
        - Column 1: Date (YYYY-MM-DD or any recognizable date format)
        - Column 2: Currency (AUD or NZD)
        - Column 3: Tenor (1Y, 2Y, 5Y, 10Y, 30Y, etc.)
        - Column 4: Rate (numeric, can be percentage or decimal)
        
        Expected format (5 columns - with floating rate):
        - Column 1: Date
        - Column 2: Currency (AUD or NZD)
        - Column 3: FloatingRate (1M, 3M, 6M, 1Y)
        - Column 4: Tenor (1Y, 2Y, 5Y, 10Y, 30Y, etc.)
        - Column 5: Rate (numeric, can be percentage or decimal)
        
        Alternative format (wide format):
        - Column 1: Date
        - Columns 2+: Tenor labels (1Y, 2Y, 5Y, etc.) with rates as values
        - Currency specified separately or inferred
        
        Returns: dict with success status and statistics
        """
        try:
            # Read Excel file
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Check if we have unnamed columns (indicates title/blank rows before headers)
            if any('Unnamed' in str(col) for col in df.columns) or any('to' in str(col).lower() for col in df.columns):
                # Re-read skipping the title and blank rows
                df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=2)
            
            # Detect format
            if self._is_long_format(df):
                return self._import_long_format(df)
            else:
                return self._import_wide_format(df, currency=currency)
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'records_imported': 0
            }
    
    def _is_long_format(self, df):
        """Detect if data is in long format (date, currency, tenor, rate) or (date, currency, floating_rate, tenor, rate)"""
        # Check if we have 4 or 5 columns
        if len(df.columns) in [4, 5]:
            return True
        # Check for specific column names
        cols_lower = [str(col).lower() for col in df.columns]
        if 'tenor' in cols_lower and 'currency' in cols_lower:
            return True
        return False
    
    def _import_long_format(self, df):
        """Import data in long format"""
        records_imported = 0
        errors = []
        
        # Check if we have 4 or 5 columns (with or without FloatingRate)
        num_cols = len(df.columns)
        
        if num_cols == 5:
            # Has floating_rate column
            df.columns = ['date', 'currency', 'floating_rate', 'tenor', 'rate']
        elif num_cols == 4:
            # No floating_rate column - add default based on currency using currency_config
            df.columns = ['date', 'currency', 'tenor', 'rate']
            # Default to 6M with appropriate fixing reference based on currency
            df['floating_rate'] = df['currency'].apply(
                lambda c: get_fixing_reference(str(c).strip().upper(), '6M')
            )
        else:
            return {
                'success': False,
                'error': f'Expected 4 or 5 columns, got {num_cols}',
                'records_imported': 0
            }
        
        for idx, row in df.iterrows():
            try:
                # Parse date
                if pd.isna(row['date']):
                    continue
                    
                date_obj = pd.to_datetime(row['date']).date()
                
                # Validate currency
                currency = str(row['currency']).strip().upper()
                if currency not in SUPPORTED_CURRENCIES:
                    errors.append(f"Row {idx}: Invalid currency '{currency}' (supported: {', '.join(SUPPORTED_CURRENCIES)})")
                    continue
                
                # Parse floating rate and add fixing reference using currency_config
                floating_rate_raw = str(row['floating_rate']).strip().upper()
                
                # Use currency_config to get proper fixing reference
                floating_rate = get_fixing_reference(currency, floating_rate_raw)
                
                # Parse tenor
                tenor = str(row['tenor']).strip().upper()
                
                # Parse rate (handle percentage format)
                rate = float(row['rate'])
                
                # Convert to decimal format (0.04 for 4%)
                # Assume rates >= 0.1 are in percentage format and need /100
                # This handles: 4.5% → 0.045, 0.9% → 0.009, 0.04 (already decimal) → 0.04
                if rate > 100:  # Likely in basis points (450 → 4.5%)
                    rate = rate / 10000
                elif rate >= 0.1:  # Likely in percentage (4.5 → 0.045, or 0.9 → 0.009)
                    rate = rate / 100
                # else: rate < 0.1, assume already in decimal format (0.045 stays 0.045)
                
                # Add to database with floating_rate
                success = self.db_manager.add_rate(date_obj, currency, tenor, rate, floating_rate)
                if success:
                    records_imported += 1
                else:
                    errors.append(f"Row {idx}: Failed to insert into database")
                    
            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")
        
        return {
            'success': records_imported > 0,
            'records_imported': records_imported,
            'errors': errors[:10],  # Limit errors shown
            'total_errors': len(errors)
        }
    
    def _import_wide_format(self, df, currency='AUD'):
        """
        Import data in wide format
        First column is date, remaining columns are tenors
        
        Args:
            df: DataFrame to import
            currency: Currency code
        """
        records_imported = 0
        errors = []
        
        # First column is date
        date_col = df.columns[0]
        tenor_cols = df.columns[1:]
        
        for idx, row in df.iterrows():
            try:
                # Parse date
                if pd.isna(row[date_col]):
                    continue
                    
                date_obj = pd.to_datetime(row[date_col]).date()
                
                # Import each tenor
                for tenor_col in tenor_cols:
                    if pd.isna(row[tenor_col]):
                        continue
                    
                    tenor = str(tenor_col).strip().upper()
                    rate = float(row[tenor_col])
                    
                    # Handle percentage format
                    # Convert to decimal format (0.04 for 4%)
                    if rate > 100:
                        rate = rate / 10000
                    elif rate >= 0.1:
                        rate = rate / 100
                    # else: already in decimal format
                    
                    # Add to database with appropriate fixing reference using currency_config
                    floating_rate = get_fixing_reference(currency, '6M')
                    
                    success = self.db_manager.add_rate(date_obj, currency, tenor, rate, floating_rate)
                    if success:
                        records_imported += 1
                        
            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")
        
        return {
            'success': records_imported > 0,
            'records_imported': records_imported,
            'errors': errors[:10],
            'total_errors': len(errors)
        }
    
    def import_multiple_currencies(self, file_path, currency_sheet_map):
        """
        Import data from multiple sheets, one per currency
        
        Args:
            file_path: Path to Excel file
            currency_sheet_map: Dict like {'AUD': 'Sheet1', 'NZD': 'Sheet2'}
        """
        results = {}
        
        for currency, sheet_name in currency_sheet_map.items():
            result = self.import_from_excel(file_path, sheet_name)
            results[currency] = result
        
        return results


def create_sample_excel(output_path='sample_swap_rates.xlsx'):
    """Create a sample Excel file showing the expected format"""
    
    # Long format example
    data_long = {
        'Date': ['2024-01-15', '2024-01-15', '2024-01-15', '2024-01-16', '2024-01-16'],
        'Currency': ['AUD', 'AUD', 'AUD', 'NZD', 'NZD'],
        'Tenor': ['1Y', '5Y', '10Y', '1Y', '5Y'],
        'Rate': [4.25, 4.50, 4.75, 5.10, 5.35]
    }
    
    # Wide format example
    dates = pd.date_range('2024-01-15', periods=5, freq='D')
    data_wide = {
        'Date': dates,
        '1Y': [4.25, 4.26, 4.27, 4.28, 4.29],
        '2Y': [4.35, 4.36, 4.37, 4.38, 4.39],
        '5Y': [4.50, 4.51, 4.52, 4.53, 4.54],
        '10Y': [4.75, 4.76, 4.77, 4.78, 4.79],
        '30Y': [5.00, 5.01, 5.02, 5.03, 5.04]
    }
    
    # Create Excel file with both formats
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        pd.DataFrame(data_long).to_excel(writer, sheet_name='Long Format', index=False)
        pd.DataFrame(data_wide).to_excel(writer, sheet_name='Wide Format', index=False)
    
    print(f"Sample Excel file created: {output_path}")
    return output_path


if __name__ == '__main__':
    # Create sample file for testing
    create_sample_excel()
