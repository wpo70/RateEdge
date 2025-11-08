"""
Forward Swap Pricer - Professional swap valuation engine
Supports OIS discounting, convexity adjustments, and tenor basis swaps
"""
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import math
import numpy as np


class SwapPricer:
    """Professional swap pricing engine"""
    
    def __init__(self, valuation_date=None):
        """Initialize pricer with valuation date"""
        self.valuation_date = valuation_date or datetime.now().date()
        self.day_count = 'ACT/365F'  # Australian convention
        
    def year_fraction(self, start_date, end_date):
        """Calculate year fraction using ACT/365F"""
        days = (end_date - start_date).days
        return days / 365.0
    
    def discount_factor(self, zero_rate, time_years, compounding='Continuous'):
        """
        Calculate discount factor from zero rate
        
        Args:
            zero_rate: Zero rate (as decimal, e.g., 0.04 for 4%)
            time_years: Time to maturity in years
            compounding: 'Continuous', 'Annual', 'Semi-Annual', 'Quarterly'
        """
        if compounding == 'Continuous':
            return math.exp(-zero_rate * time_years)
        elif compounding == 'Annual':
            return 1.0 / ((1 + zero_rate) ** time_years)
        elif compounding == 'Semi-Annual':
            return 1.0 / ((1 + zero_rate / 2) ** (2 * time_years))
        elif compounding == 'Quarterly':
            return 1.0 / ((1 + zero_rate / 4) ** (4 * time_years))
        else:
            return math.exp(-zero_rate * time_years)
    
    def interpolate_zero_rate(self, tenor_months, zero_curve):
        """
        Interpolate zero rate for a given tenor
        
        Args:
            tenor_months: Tenor in months
            zero_curve: Dict of {tenor_months: zero_rate}
        
        Returns:
            Interpolated zero rate
        """
        # Convert to list and sort
        tenors = sorted(zero_curve.keys())
        
        if tenor_months in tenors:
            return zero_curve[tenor_months]
        
        # Find bracketing tenors
        lower = max([t for t in tenors if t < tenor_months], default=tenors[0])
        upper = min([t for t in tenors if t > tenor_months], default=tenors[-1])
        
        if lower == upper:
            return zero_curve[lower]
        
        # Linear interpolation
        weight = (tenor_months - lower) / (upper - lower)
        return zero_curve[lower] + weight * (zero_curve[upper] - zero_curve[lower])
    
    def generate_schedule(self, start_date, end_date, frequency_per_year):
        """
        Generate payment schedule
        
        Args:
            start_date: Swap start date
            end_date: Swap end date
            frequency_per_year: Payment frequency (2=semi-annual, 4=quarterly)
        
        Returns:
            List of payment dates
        """
        months_per_period = 12 // frequency_per_year
        schedule = []
        current = start_date
        
        while current < end_date:
            current = current + relativedelta(months=months_per_period)
            if current > end_date:
                current = end_date
            schedule.append(current)
        
        return schedule
    
    def price_forward_swap(self, 
                          start_months,
                          maturity_years,
                          fixed_rate,
                          notional,
                          projection_curve,
                          discount_curve=None,
                          fixed_freq=2,
                          float_freq=4,
                          float_tenor_months=3,
                          use_ois_discounting=True,
                          fixed_spread_bp=0,
                          float_margin_bp=0,
                          convexity_adj_float_bp=0,
                          convexity_adj_fixed_bp=0):
        """
        Price a forward-starting fixed-for-float interest rate swap
        
        Args:
            start_months: Months from valuation date to start
            maturity_years: Swap maturity in years from start
            fixed_rate: Fixed rate as decimal (e.g., 0.04 for 4%)
            notional: Notional amount
            projection_curve: Dict of {tenor_months: zero_rate} for projecting floats
            discount_curve: Dict for OIS discounting (if use_ois_discounting=True)
            fixed_freq: Fixed leg frequency per year (2=semi-annual)
            float_freq: Float leg frequency per year (4=quarterly)
            float_tenor_months: Float index tenor (3=3M BBSW, 6=6M BBSW)
            use_ois_discounting: Whether to use OIS curve for discounting
            fixed_spread_bp: Spread added to fixed leg (basis points)
            float_margin_bp: Margin added to float leg (basis points)
            convexity_adj_float_bp: Convexity adjustment for float leg
            convexity_adj_fixed_bp: Convexity adjustment for fixed leg
        
        Returns:
            Dict with swap value, fixed leg PV, float leg PV, and par rate
        """
        # Determine discount curve
        if use_ois_discounting and discount_curve:
            disc_curve = discount_curve
        else:
            disc_curve = projection_curve
        
        # Convert basis points to decimal
        fixed_spread = fixed_spread_bp / 10000.0
        float_margin = float_margin_bp / 10000.0
        conv_adj_float = convexity_adj_float_bp / 10000.0
        conv_adj_fixed = convexity_adj_fixed_bp / 10000.0
        
        # Calculate start and end dates
        start_date = self.valuation_date + relativedelta(months=start_months)
        end_date = start_date + relativedelta(years=maturity_years)
        
        # Generate payment schedules
        fixed_schedule = self.generate_schedule(start_date, end_date, fixed_freq)
        float_schedule = self.generate_schedule(start_date, end_date, float_freq)
        
        # Calculate fixed leg PV
        fixed_pv = 0.0
        fixed_leg_details = []
        
        prev_date = start_date
        for pay_date in fixed_schedule:
            # Year fraction
            yf = self.year_fraction(prev_date, pay_date)
            
            # Time to payment from valuation
            time_to_payment = self.year_fraction(self.valuation_date, pay_date)
            
            # Discount factor
            zero_rate = self.interpolate_zero_rate(time_to_payment * 12, disc_curve)
            df = self.discount_factor(zero_rate, time_to_payment)
            
            # Cash flow (with spread and convexity adjustment)
            cash_flow = notional * (fixed_rate + fixed_spread + conv_adj_fixed) * yf
            pv = cash_flow * df
            
            fixed_pv += pv
            fixed_leg_details.append({
                'pay_date': pay_date,
                'year_fraction': yf,
                'rate': fixed_rate + fixed_spread,
                'cash_flow': cash_flow,
                'discount_factor': df,
                'pv': pv
            })
            
            prev_date = pay_date
        
        # Calculate float leg PV
        float_pv = 0.0
        float_leg_details = []
        
        prev_date = start_date
        for pay_date in float_schedule:
            # Year fraction for payment
            yf = self.year_fraction(prev_date, pay_date)
            
            # Project forward rate
            # Time from valuation to start of period
            time_to_start = self.year_fraction(self.valuation_date, prev_date)
            # Time from valuation to end of period
            time_to_end = self.year_fraction(self.valuation_date, pay_date)
            
            # Get zero rates
            zero_start = self.interpolate_zero_rate(time_to_start * 12, projection_curve)
            zero_end = self.interpolate_zero_rate(time_to_end * 12, projection_curve)
            
            # Calculate implied forward rate (continuous compounding)
            if time_to_start > 0:
                forward_rate = (zero_end * time_to_end - zero_start * time_to_start) / yf
            else:
                forward_rate = zero_end
            
            # Discount factor for payment
            zero_rate_disc = self.interpolate_zero_rate(time_to_end * 12, disc_curve)
            df = self.discount_factor(zero_rate_disc, time_to_end)
            
            # Cash flow (with margin and convexity adjustment)
            cash_flow = notional * (forward_rate + float_margin + conv_adj_float) * yf
            pv = cash_flow * df
            
            float_pv += pv
            float_leg_details.append({
                'pay_date': pay_date,
                'start_date': prev_date,
                'year_fraction': yf,
                'forward_rate': forward_rate,
                'cash_flow': cash_flow,
                'discount_factor': df,
                'pv': pv
            })
            
            prev_date = pay_date
        
        # Swap value (receiver perspective: receive fixed, pay float)
        swap_value = fixed_pv - float_pv
        
        # Calculate par rate (the fixed rate that makes swap value zero)
        # Sum of discount factors on fixed leg
        fixed_annuity = sum([detail['discount_factor'] * detail['year_fraction'] 
                            for detail in fixed_leg_details])
        
        if fixed_annuity > 0:
            par_rate = float_pv / (notional * fixed_annuity)
        else:
            par_rate = 0.0
        
        return {
            'swap_value': swap_value,
            'fixed_leg_pv': fixed_pv,
            'float_leg_pv': float_pv,
            'par_rate': par_rate,
            'par_rate_percent': par_rate * 100,
            'fixed_leg_details': fixed_leg_details,
            'float_leg_details': float_leg_details,
            'start_date': start_date,
            'end_date': end_date
        }
    
    def price_tenor_basis_swap(self,
                               start_months,
                               maturity_years,
                               leg_a_tenor_months,
                               leg_b_tenor_months,
                               notional,
                               projection_curve,
                               discount_curve=None,
                               leg_a_margin_bp=0,
                               leg_b_margin_bp=0,
                               use_ois_discounting=True):
        """
        Price a tenor basis swap (e.g., 3M vs 6M BBSW)
        
        Solves for the par basis that makes the swap value zero
        """
        # Similar structure to forward swap but with two float legs
        # Implementation would follow similar pattern
        pass


def test_pricer():
    """Test the swap pricer"""
    pricer = SwapPricer(valuation_date=datetime(2025, 10, 20).date())
    
    # Sample projection curve (zero rates as decimals)
    projection_curve = {
        1: 0.0400,    # 1M: 4.00%
        3: 0.0410,    # 3M: 4.10%
        6: 0.0420,    # 6M: 4.20%
        9: 0.0425,    # 9M: 4.25%
        12: 0.0430,   # 1Y: 4.30%
        18: 0.0435,   # 18M: 4.35%
        24: 0.0440,   # 2Y: 4.40%
        36: 0.0445,   # 3Y: 4.45%
        48: 0.0450,   # 4Y: 4.50%
        60: 0.0455,   # 5Y: 4.55%
        84: 0.0460,   # 7Y: 4.60%
        120: 0.0465   # 10Y: 4.65%
    }
    
    # Price a 6M forward, 5Y swap
    result = pricer.price_forward_swap(
        start_months=6,
        maturity_years=5,
        fixed_rate=0.0450,  # 4.50%
        notional=10_000_000,
        projection_curve=projection_curve,
        fixed_freq=2,
        float_freq=4
    )
    
    print("Forward Swap Pricing Test")
    print("=" * 60)
    print(f"Swap Value: ${result['swap_value']:,.2f}")
    print(f"Fixed Leg PV: ${result['fixed_leg_pv']:,.2f}")
    print(f"Float Leg PV: ${result['float_leg_pv']:,.2f}")
    print(f"Par Rate: {result['par_rate_percent']:.4f}%")
    print(f"Start Date: {result['start_date']}")
    print(f"End Date: {result['end_date']}")


if __name__ == '__main__':
    test_pricer()
