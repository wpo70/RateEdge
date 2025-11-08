"""
Currency Configuration for IRS Manager
Defines fixing references and conventions for each currency
"""

# Currency fixing references and conventions
CURRENCY_CONFIG = {
    'AUD': {
        'name': 'Australian Dollar',
        'fixing_reference': 'BBSW',
        'full_name': 'Bank Bill Swap Rate',
        'administrator': 'ASX',
        'common_tenors': ['1M', '2M', '3M', '4M', '5M', '6M'],
        'color': '#0066CC',
        'symbol': 'A$'
    },
    'NZD': {
        'name': 'New Zealand Dollar',
        'fixing_reference': 'BKBM',
        'full_name': 'Bank Bill Benchmark',
        'administrator': 'NZFMA',
        'common_tenors': ['1M', '2M', '3M', '4M', '5M', '6M'],
        'color': '#000000',
        'symbol': 'NZ$'
    },
    'USD': {
        'name': 'US Dollar',
        'fixing_reference': 'SOFR',
        'full_name': 'Secured Overnight Financing Rate',
        'administrator': 'Federal Reserve Bank of NY',
        'common_tenors': ['1M', '3M', '6M', '12M'],
        'legacy_reference': 'LIBOR',  # Historical data might use LIBOR
        'color': '#006400',
        'symbol': '$'
    },
    'EUR': {
        'name': 'Euro',
        'fixing_reference': 'EURIBOR',
        'full_name': 'Euro Interbank Offered Rate',
        'administrator': 'EMMI',
        'common_tenors': ['1W', '1M', '3M', '6M', '12M'],
        'alternative_reference': '€STR',  # Euro Short-Term Rate
        'color': '#003399',
        'symbol': '€'
    },
    'GBP': {
        'name': 'British Pound',
        'fixing_reference': 'SONIA',
        'full_name': 'Sterling Overnight Index Average',
        'administrator': 'Bank of England',
        'common_tenors': ['1M', '3M', '6M', '12M'],
        'legacy_reference': 'LIBOR',  # Historical data might use LIBOR
        'color': '#C8102E',
        'symbol': '£'
    },
    'JPY': {
        'name': 'Japanese Yen',
        'fixing_reference': 'TONA',
        'full_name': 'Tokyo Overnight Average Rate',
        'administrator': 'Bank of Japan',
        'common_tenors': ['1M', '3M', '6M', '12M'],
        'alternative_reference': 'TIBOR',  # Tokyo Interbank Offered Rate
        'legacy_reference': 'LIBOR',  # Historical data might use LIBOR
        'color': '#BC002D',
        'symbol': '¥'
    },
    'CAD': {
        'name': 'Canadian Dollar',
        'fixing_reference': 'CORRA',
        'full_name': 'Canadian Overnight Repo Rate Average',
        'administrator': 'Bank of Canada',
        'common_tenors': ['1M', '3M', '6M', '12M'],
        'legacy_reference': 'CDOR',
        'color': '#FF0000',
        'symbol': 'C$'
    }
}

# Supported currencies list
SUPPORTED_CURRENCIES = list(CURRENCY_CONFIG.keys())

def get_fixing_reference(currency, floating_rate):
    """
    Get the full fixing reference for a currency and floating rate
    
    Args:
        currency: Currency code (e.g., 'AUD', 'USD')
        floating_rate: Floating rate tenor (e.g., '3M', '6M')
        
    Returns:
        Full fixing reference (e.g., '3M BBSW', '3M SOFR')
    """
    if currency not in CURRENCY_CONFIG:
        return floating_rate
    
    config = CURRENCY_CONFIG[currency]
    fixing_ref = config['fixing_reference']
    
    # Clean floating rate (remove existing reference if present)
    period = floating_rate.split()[0] if ' ' in floating_rate else floating_rate
    
    # Check if it already has a fixing reference
    if fixing_ref in floating_rate or 'LIBOR' in floating_rate or 'TIBOR' in floating_rate:
        return floating_rate
    
    # Add fixing reference
    return f"{period} {fixing_ref}"

def get_currency_info(currency):
    """Get full currency configuration"""
    return CURRENCY_CONFIG.get(currency, {
        'name': currency,
        'fixing_reference': 'Unknown',
        'color': '#666666',
        'symbol': currency
    })

def get_currency_name(currency):
    """Get full name of currency"""
    return CURRENCY_CONFIG.get(currency, {}).get('name', currency)

def get_currency_color(currency):
    """Get color for currency in charts"""
    return CURRENCY_CONFIG.get(currency, {}).get('color', '#666666')

def parse_floating_rate(floating_rate):
    """
    Parse floating rate to extract period (3M, 6M) and reference (BBSW, SOFR)
    
    Args:
        floating_rate: Rate string like '3M BBSW' or '3M'
        
    Returns:
        tuple: (period, reference) e.g., ('3M', 'BBSW') or ('3M', None)
    """
    parts = floating_rate.strip().split()
    if len(parts) >= 2:
        return parts[0], ' '.join(parts[1:])
    return parts[0], None
