"""
Flask API Backend for IRS Swap Rate Application
Provides RESTful endpoints for data access
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from database_models import DatabaseManager, SwapRate
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for desktop app to connect

# Initialize database
db_manager = DatabaseManager()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'API is running'}), 200


@app.route('/api/rates', methods=['GET'])
def get_rates():
    """
    Get swap rates with optional filters
    Query parameters:
        - currency: AUD or NZD
        - tenor: e.g., 1Y, 5Y, 10Y
        - start_date: YYYY-MM-DD
        - end_date: YYYY-MM-DD
    """
    try:
        currency = request.args.get('currency')
        tenor = request.args.get('tenor')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Convert dates if provided
        if start_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if end_date:
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        rates = db_manager.get_rates(currency, tenor, start_date, end_date)
        
        return jsonify({
            'success': True,
            'count': len(rates),
            'data': [rate.to_dict() for rate in rates]
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/rates/latest', methods=['GET'])
def get_latest_rates():
    """
    Get most recent rates for all tenors
    Query parameters:
        - currency: AUD or NZD (optional)
    """
    try:
        currency = request.args.get('currency')
        rates = db_manager.get_latest_rates(currency)
        
        return jsonify({
            'success': True,
            'count': len(rates),
            'data': [rate.to_dict() for rate in rates]
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/rates', methods=['POST'])
def add_rate():
    """
    Add a single swap rate
    Body: JSON with date, currency, tenor, rate
    """
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['date', 'currency', 'tenor', 'rate']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing field: {field}'}), 400
        
        # Convert date string to date object
        date_obj = datetime.strptime(data['date'], '%Y-%m-%d').date()
        
        # Validate currency
        if data['currency'] not in ['AUD', 'NZD']:
            return jsonify({'success': False, 'error': 'Currency must be AUD or NZD'}), 400
        
        success = db_manager.add_rate(
            date=date_obj,
            currency=data['currency'],
            tenor=data['tenor'],
            rate=float(data['rate'])
        )
        
        if success:
            return jsonify({'success': True, 'message': 'Rate added successfully'}), 201
        else:
            return jsonify({'success': False, 'error': 'Failed to add rate'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/rates/bulk', methods=['POST'])
def bulk_add_rates():
    """
    Add multiple rates at once
    Body: JSON array of rate objects
    """
    try:
        data = request.json
        
        if not isinstance(data, list):
            return jsonify({'success': False, 'error': 'Data must be an array'}), 400
        
        # Convert dates and validate
        rates_data = []
        for item in data:
            date_obj = datetime.strptime(item['date'], '%Y-%m-%d').date()
            rates_data.append({
                'date': date_obj,
                'currency': item['currency'],
                'tenor': item['tenor'],
                'rate': float(item['rate'])
            })
        
        success = db_manager.bulk_add_rates(rates_data)
        
        if success:
            return jsonify({
                'success': True, 
                'message': f'{len(rates_data)} rates added successfully'
            }), 201
        else:
            return jsonify({'success': False, 'error': 'Failed to add rates'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/rates', methods=['DELETE'])
def delete_rates():
    """
    Delete rates based on filters
    Query parameters:
        - currency: AUD or NZD
        - start_date: YYYY-MM-DD
        - end_date: YYYY-MM-DD
    """
    try:
        currency = request.args.get('currency')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Convert dates if provided
        if start_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if end_date:
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        count = db_manager.delete_rates(currency, start_date, end_date)
        
        return jsonify({
            'success': True,
            'message': f'{count} rates deleted'
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/metadata/dates', methods=['GET'])
def get_available_dates():
    """Get list of all dates with data"""
    try:
        currency = request.args.get('currency')
        dates = db_manager.get_available_dates(currency)
        
        return jsonify({
            'success': True,
            'count': len(dates),
            'dates': [d.isoformat() for d in dates]
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/metadata/tenors', methods=['GET'])
def get_available_tenors():
    """Get list of all available tenors"""
    try:
        currency = request.args.get('currency')
        tenors = db_manager.get_available_tenors(currency)
        
        return jsonify({
            'success': True,
            'count': len(tenors),
            'tenors': tenors
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/export', methods=['GET'])
def export_data():
    """
    Export data to Excel format
    Query parameters: same as /api/rates
    """
    try:
        currency = request.args.get('currency')
        tenor = request.args.get('tenor')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Convert dates if provided
        if start_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if end_date:
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        rates = db_manager.get_rates(currency, tenor, start_date, end_date)
        
        # Convert to DataFrame
        data_list = [rate.to_dict() for rate in rates]
        df = pd.DataFrame(data_list)
        
        # Create Excel file
        output_path = '/tmp/swap_rates_export.xlsx'
        df.to_excel(output_path, index=False, engine='openpyxl')
        
        from flask import send_file
        return send_file(output_path, as_attachment=True, 
                        download_name='swap_rates_export.xlsx')
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    # For local development
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
