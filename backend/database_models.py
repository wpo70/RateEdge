"""
Database models for IRS Swap Rate Storage
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()


def tenor_sort_key(tenor):
    """
    Convert tenor string to months for proper numerical sorting
    Examples: '6M' -> 6, '1Y' -> 12, '18M' -> 18, '5Y' -> 60
    """
    tenor = str(tenor).upper().strip()
    
    if tenor.endswith('M'):
        # Months
        return int(tenor.replace('M', ''))
    elif tenor.endswith('Y'):
        # Years - convert to months
        return int(tenor.replace('Y', '')) * 12
    else:
        # Fallback - try to extract number
        import re
        match = re.search(r'\d+', tenor)
        if match:
            return int(match.group()) * 12
        return 0

class SwapRate(Base):
    """Model for storing interest rate swap data"""
    __tablename__ = 'swap_rates'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    currency = Column(String(3), nullable=False, index=True)  # AUD, NZD, USD, EUR, GBP, JPY
    tenor = Column(String(10), nullable=False, index=True)  # e.g., '1Y', '2Y', '5Y', '10Y', '30Y'
    floating_rate = Column(String(20), nullable=False, default='6M', index=True)  # '3M BBSW', '6M SOFR', etc.
    rate = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Ensure unique combination of date, currency, tenor, and floating_rate
    __table_args__ = (
        UniqueConstraint('date', 'currency', 'tenor', 'floating_rate', name='unique_rate_entry'),
    )
    
    def __repr__(self):
        return f"<SwapRate(date={self.date}, currency={self.currency}, floating_rate={self.floating_rate}, tenor={self.tenor}, rate={self.rate})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'date': self.date.isoformat() if self.date else None,
            'currency': self.currency,
            'floating_rate': self.floating_rate,
            'tenor': self.tenor,
            'rate': self.rate,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class DatabaseManager:
    """Manager class for database operations"""
    
    def __init__(self, db_url='sqlite:///./database/swap_rates.db'):
        """Initialize database connection"""
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def add_rate(self, date, currency, tenor, rate, floating_rate='6M'):
        """Add or update a single swap rate"""
        try:
            # Check if entry exists
            existing = self.session.query(SwapRate).filter_by(
                date=date, currency=currency, tenor=tenor, floating_rate=floating_rate
            ).first()
            
            if existing:
                # Update existing rate
                existing.rate = rate
                existing.updated_at = datetime.utcnow()
            else:
                # Create new entry
                new_rate = SwapRate(
                    date=date,
                    currency=currency,
                    tenor=tenor,
                    floating_rate=floating_rate,
                    rate=rate
                )
                self.session.add(new_rate)
            
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            print(f"Error adding rate: {e}")
            return False
    
    def bulk_add_rates(self, rates_data):
        """Add multiple rates at once (faster for imports)"""
        try:
            for rate_dict in rates_data:
                self.add_rate(
                    date=rate_dict['date'],
                    currency=rate_dict['currency'],
                    tenor=rate_dict['tenor'],
                    rate=rate_dict['rate']
                )
            return True
        except Exception as e:
            self.session.rollback()
            print(f"Error in bulk add: {e}")
            return False
    
    def get_rates(self, currency=None, tenor=None, start_date=None, end_date=None, floating_rate=None):
        """Query swap rates with optional filters"""
        query = self.session.query(SwapRate)
        
        if currency:
            query = query.filter(SwapRate.currency == currency)
        if tenor:
            query = query.filter(SwapRate.tenor == tenor)
        if start_date:
            query = query.filter(SwapRate.date >= start_date)
        if end_date:
            query = query.filter(SwapRate.date <= end_date)
        if floating_rate:
            query = query.filter(SwapRate.floating_rate == floating_rate)
        
        query = query.order_by(SwapRate.date.desc())
        return query.all()
    
    def get_latest_rates(self, currency=None):
        """Get the most recent rates for all tenors, sorted numerically"""
        from sqlalchemy import func
        
        # Get the latest date
        subquery = self.session.query(func.max(SwapRate.date)).scalar_subquery()
        query = self.session.query(SwapRate).filter(SwapRate.date == subquery)
        
        if currency:
            query = query.filter(SwapRate.currency == currency)
        
        rates = query.all()
        
        # Sort by tenor numerically
        rates_sorted = sorted(rates, key=lambda x: tenor_sort_key(x.tenor))
        
        return rates_sorted
    
    def delete_rates(self, currency=None, start_date=None, end_date=None):
        """Delete rates based on filters"""
        query = self.session.query(SwapRate)
        
        if currency:
            query = query.filter(SwapRate.currency == currency)
        if start_date:
            query = query.filter(SwapRate.date >= start_date)
        if end_date:
            query = query.filter(SwapRate.date <= end_date)
        
        count = query.count()
        query.delete()
        self.session.commit()
        return count
    
    def get_available_dates(self, currency=None):
        """Get list of all dates with data"""
        from sqlalchemy import func, distinct
        
        query = self.session.query(distinct(SwapRate.date))
        
        if currency:
            query = query.filter(SwapRate.currency == currency)
        
        dates = [row[0] for row in query.order_by(SwapRate.date.desc()).all()]
        return dates
    
    def get_available_tenors(self, currency=None):
        """Get list of all available tenors, sorted numerically"""
        from sqlalchemy import distinct
        
        query = self.session.query(distinct(SwapRate.tenor))
        
        if currency:
            query = query.filter(SwapRate.currency == currency)
        
        tenors = [row[0] for row in query.all()]
        
        # Sort tenors numerically using our helper function
        tenors_sorted = sorted(tenors, key=tenor_sort_key)
        
        return tenors_sorted
    
    def get_available_floating_rates(self, currency=None):
        """Get list of all available floating rates for a currency"""
        from sqlalchemy import distinct
        
        query = self.session.query(distinct(SwapRate.floating_rate))
        
        if currency:
            query = query.filter(SwapRate.currency == currency)
        
        floating_rates = [row[0] for row in query.order_by(SwapRate.floating_rate).all()]
        return floating_rates
    
    def get_benchmark_rates(self, currency=None, rate_type=None, start_date=None, end_date=None, limit=None):
        """Get benchmark rates with optional filters"""
        from sqlalchemy import text
        
        # Force date to be returned as string in YYYY-MM-DD format
        query = "SELECT DATE(date) as date, currency, rate_type, rate FROM benchmark_rates WHERE 1=1"
        params = {}
        
        if currency:
            query += " AND currency = :currency"
            params['currency'] = currency
        
        if rate_type:
            query += " AND rate_type = :rate_type"
            params['rate_type'] = rate_type
        
        if start_date:
            query += " AND date >= :start_date"
            params['start_date'] = start_date
        
        if end_date:
            query += " AND date <= :end_date"
            params['end_date'] = end_date
        
        query += " ORDER BY date DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        result = self.session.execute(text(query), params)
        return result.fetchall()
    
    def get_benchmark_rate_types(self, currency=None):
        """Get list of available benchmark rate types"""
        from sqlalchemy import text
        
        query = "SELECT DISTINCT rate_type FROM benchmark_rates"
        params = {}
        
        if currency:
            query += " WHERE currency = :currency"
            params['currency'] = currency
        
        query += " ORDER BY rate_type"
        
        result = self.session.execute(text(query), params)
        return [row[0] for row in result.fetchall()]
    
    def get_ois_rates(self, currency=None, rate_type=None, start_date=None, end_date=None, limit=None):
        """Get OIS rates with optional filters"""
        from sqlalchemy import text
        
        # Force date to be returned as string in YYYY-MM-DD format
        query = "SELECT DATE(date) as date, currency, rate_type, rate FROM ois_rates WHERE 1=1"
        params = {}
        
        if currency:
            query += " AND currency = :currency"
            params['currency'] = currency
        
        if rate_type:
            query += " AND rate_type = :rate_type"
            params['rate_type'] = rate_type
        
        if start_date:
            query += " AND date >= :start_date"
            params['start_date'] = start_date
        
        if end_date:
            query += " AND date <= :end_date"
            params['end_date'] = end_date
        
        query += " ORDER BY date DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        result = self.session.execute(text(query), params)
        return result.fetchall()
    
    def get_ois_rate_types(self, currency=None):
        """Get list of available OIS rate types"""
        from sqlalchemy import text
        
        query = "SELECT DISTINCT rate_type FROM ois_rates"
        params = {}
        
        if currency:
            query += " WHERE currency = :currency"
            params['currency'] = currency
        
        query += " ORDER BY rate_type"
        
        result = self.session.execute(text(query), params)
        return [row[0] for row in result.fetchall()]
    
    def close(self):
        """Close database connection"""
        self.session.close()

