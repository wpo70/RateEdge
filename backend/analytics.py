"""
Statistics and Analytics Module for Swap Rates
Provides calculations for rate analysis
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class SwapRateAnalytics:
    """Calculate statistics and analytics for swap rates"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    def get_rate_statistics(self, currency, tenor, start_date=None, end_date=None):
        """
        Calculate comprehensive statistics for a tenor
        Returns: dict with various statistics
        """
        rates = self.db_manager.get_rates(
            currency=currency,
            tenor=tenor,
            start_date=start_date,
            end_date=end_date
        )
        
        if not rates:
            return None
        
        # Convert to list of values
        values = [r.rate * 100 for r in rates]  # Convert to percentage
        dates = [r.date for r in rates]
        
        stats = {
            'count': len(values),
            'current': values[0] if values else None,
            'mean': np.mean(values),
            'median': np.median(values),
            'std_dev': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values),
            'first_date': dates[-1] if dates else None,
            'last_date': dates[0] if dates else None,
            'percentile_25': np.percentile(values, 25),
            'percentile_75': np.percentile(values, 75),
        }
        
        # Calculate rate changes
        if len(values) >= 2:
            stats['change_1d'] = values[0] - values[1] if len(values) > 1 else 0
            stats['change_1w'] = values[0] - values[min(5, len(values)-1)] if len(values) > 5 else 0
            stats['change_1m'] = values[0] - values[min(21, len(values)-1)] if len(values) > 21 else 0
            stats['change_3m'] = values[0] - values[min(63, len(values)-1)] if len(values) > 63 else 0
            stats['change_ytd'] = self._calculate_ytd_change(dates, values)
        
        return stats
    
    def _calculate_ytd_change(self, dates, values):
        """Calculate year-to-date change"""
        if not dates:
            return 0
        
        current_year = dates[0].year
        year_start_idx = None
        
        for i, date in enumerate(dates):
            if date.year < current_year:
                year_start_idx = i - 1
                break
        
        if year_start_idx and year_start_idx > 0:
            return values[0] - values[year_start_idx]
        return 0
    
    def calculate_spread(self, currency, tenor1, tenor2, start_date=None, end_date=None):
        """
        Calculate spread between two tenors
        Returns DataFrame with dates and spreads
        """
        rates1 = self.db_manager.get_rates(currency, tenor1, start_date, end_date)
        rates2 = self.db_manager.get_rates(currency, tenor2, start_date, end_date)
        
        if not rates1 or not rates2:
            return None
        
        # Convert to DataFrames
        df1 = pd.DataFrame([{
            'date': r.date,
            'rate1': r.rate * 100
        } for r in rates1])
        
        df2 = pd.DataFrame([{
            'date': r.date,
            'rate2': r.rate * 100
        } for r in rates2])
        
        # Merge on date
        df = pd.merge(df1, df2, on='date', how='inner')
        df['spread'] = df['rate2'] - df['rate1']
        
        # Calculate spread statistics
        spread_stats = {
            'mean_spread': df['spread'].mean(),
            'median_spread': df['spread'].median(),
            'std_spread': df['spread'].std(),
            'min_spread': df['spread'].min(),
            'max_spread': df['spread'].max(),
            'current_spread': df['spread'].iloc[0] if len(df) > 0 else None,
        }
        
        return {
            'data': df,
            'stats': spread_stats,
            'tenor1': tenor1,
            'tenor2': tenor2
        }
    
    def calculate_volatility(self, currency, tenor, window=30, start_date=None, end_date=None):
        """
        Calculate rolling volatility (standard deviation of rate changes)
        """
        rates = self.db_manager.get_rates(currency, tenor, start_date, end_date)
        
        if not rates or len(rates) < window:
            return None
        
        df = pd.DataFrame([{
            'date': r.date,
            'rate': r.rate * 100
        } for r in rates])
        
        # Sort by date ascending for proper calculation
        df = df.sort_values('date')
        
        # Calculate daily changes
        df['rate_change'] = df['rate'].diff()
        
        # Calculate rolling volatility
        df['volatility'] = df['rate_change'].rolling(window=window).std()
        
        # Annualize volatility (assuming 252 trading days)
        df['volatility_annualized'] = df['volatility'] * np.sqrt(252)
        
        return df
    
    def detect_outliers(self, currency, tenor, threshold=3, start_date=None, end_date=None):
        """
        Detect outliers using z-score method
        threshold: number of standard deviations (default 3)
        """
        rates = self.db_manager.get_rates(currency, tenor, start_date, end_date)
        
        if not rates:
            return None
        
        values = np.array([r.rate * 100 for r in rates])
        dates = [r.date for r in rates]
        
        # Calculate z-scores
        mean = np.mean(values)
        std = np.std(values)
        z_scores = np.abs((values - mean) / std)
        
        # Find outliers
        outliers = []
        for i, (date, value, z_score) in enumerate(zip(dates, values, z_scores)):
            if z_score > threshold:
                outliers.append({
                    'date': date,
                    'rate': value,
                    'z_score': z_score,
                    'deviation_from_mean': value - mean
                })
        
        return {
            'outliers': outliers,
            'count': len(outliers),
            'mean': mean,
            'std': std,
            'threshold': threshold
        }
    
    def find_missing_dates(self, currency, tenor, expected_frequency='daily'):
        """
        Find missing dates in the data series
        """
        rates = self.db_manager.get_rates(currency, tenor)
        
        if not rates or len(rates) < 2:
            return None
        
        dates = sorted([r.date for r in rates])
        
        missing_dates = []
        
        if expected_frequency == 'daily':
            # Check for missing business days (Mon-Fri)
            for i in range(len(dates) - 1):
                current = dates[i]
                next_date = dates[i + 1]
                
                # Calculate expected business days between
                expected = current + timedelta(days=1)
                while expected < next_date:
                    # Skip weekends
                    if expected.weekday() < 5:  # Mon=0, Fri=4
                        missing_dates.append(expected)
                    expected += timedelta(days=1)
        
        return {
            'missing_dates': missing_dates,
            'count': len(missing_dates),
            'first_date': dates[0],
            'last_date': dates[-1],
            'total_dates': len(dates)
        }
    
    def calculate_rate_changes(self, currency, tenor, periods=[1, 5, 21, 63, 252]):
        """
        Calculate rate changes over different periods
        periods: list of days (default: 1day, 1week, 1month, 3months, 1year)
        """
        rates = self.db_manager.get_rates(currency, tenor)
        
        if not rates:
            return None
        
        values = [r.rate * 100 for r in rates]
        dates = [r.date for r in rates]
        
        changes = {}
        current = values[0]
        
        for period in periods:
            if len(values) > period:
                change = current - values[period]
                pct_change = (change / values[period]) * 100 if values[period] != 0 else 0
                
                period_name = f"{period}d"
                if period == 1:
                    period_name = "1 Day"
                elif period == 5:
                    period_name = "1 Week"
                elif period == 21:
                    period_name = "1 Month"
                elif period == 63:
                    period_name = "3 Months"
                elif period == 252:
                    period_name = "1 Year"
                
                changes[period_name] = {
                    'absolute_change': change,
                    'percent_change': pct_change,
                    'from_date': dates[period],
                    'from_rate': values[period],
                    'to_date': dates[0],
                    'to_rate': current
                }
        
        return changes
    
    def calculate_correlation(self, currency, tenor1, tenor2, start_date=None, end_date=None):
        """
        Calculate correlation between two tenors
        """
        spread_data = self.calculate_spread(currency, tenor1, tenor2, start_date, end_date)
        
        if not spread_data or spread_data['data'].empty:
            return None
        
        df = spread_data['data']
        correlation = df['rate1'].corr(df['rate2'])
        
        return {
            'correlation': correlation,
            'tenor1': tenor1,
            'tenor2': tenor2,
            'count': len(df)
        }
