from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import sys
import os

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from database_models import DatabaseManager

app = Flask(__name__)
CORS(app)

# Database setup - ABSOLUTE PATH for Azure
db_path = '/home/site/wwwroot/database/swap_rates.db'
os.makedirs(os.path.dirname(db_path), exist_ok=True)
db = DatabaseManager(f'sqlite:///{db_path}')

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# ===== PAGE ROUTES =====
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analytics/forward-pricer')
def forward_pricer_page():
    return render_template('forward_pricer.html')

@app.route('/analytics/relative-value')
def relative_value_page():
    return render_template('relative_value.html')

@app.route('/analytics/cross-currency')
def cross_currency_page():
    return render_template('cross_currency.html')

@app.route('/analytics/butterfly')
def butterfly_page():
    return render_template('butterfly.html')

@app.route('/analytics/yield-curve')
def yield_curve_page():
    return render_template('yield_curve.html')

@app.route('/data/view')
def data_view_page():
    return render_template('data_view.html')

@app.route('/data/import')
def data_import_page():
    return render_template('data_import.html')

@app.route('/data/pivot')
def data_pivot_page():
    return render_template('pivot_table.html')

@app.route('/analytics/charts')
def analytics_charts_page():
    return render_template('analytics_charts.html')

# ===== API ROUTES =====
@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'RateEdge API', 'version': '1.0'})

@app.route('/api/statistics')
def statistics():
    try:
        from database_models import SwapRate
        from sqlalchemy import func
        
        total = db.session.query(SwapRate).count()
        currencies = db.session.query(SwapRate.currency, func.count(SwapRate.id).label('count')).group_by(SwapRate.currency).all()
        currency_counts = {curr: count for curr, count in currencies}
        
        min_date = db.session.query(func.min(SwapRate.date)).scalar()
        max_date = db.session.query(func.max(SwapRate.date)).scalar()
        
        return jsonify({
            'success': True,
            'data': {
                'total_records': total,
                'currencies': len(currencies),
                'currency_breakdown': currency_counts,
                'date_range': {
                    'start': str(min_date) if min_date else None,
                    'end': str(max_date) if max_date else None
                }
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/currencies')
def get_currencies():
    try:
        from database_models import SwapRate
        currencies = db.session.query(SwapRate.currency).distinct().all()
        currency_list = sorted([c[0] for c in currencies])
        return jsonify({'success': True, 'data': currency_list})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/tenors/<currency>')
def get_tenors(currency):
    try:
        tenors = db.get_available_tenors(currency=currency)
        return jsonify({'success': True, 'data': tenors})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/rates')
def get_rates():
    try:
        currency = request.args.get('currency')
        tenor = request.args.get('tenor')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = request.args.get('limit', 1000, type=int)
        
        rates = db.get_rates(currency=currency, tenor=tenor, start_date=start_date, end_date=end_date)
        rates = rates[:limit]
        
        rate_data = []
        for rate in rates:
            rate_data.append({
                'date': rate.date.isoformat(),
                'currency': rate.currency,
                'tenor': rate.tenor,
                'floating_rate': rate.floating_rate,
                'rate': float(rate.rate),
                'rate_percent': float(rate.rate * 100)
            })
        
        return jsonify({'success': True, 'data': rate_data, 'count': len(rate_data)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/latest/<currency>')
def get_latest(currency):
    try:
        rates = db.get_latest_rates(currency=currency)
        
        rate_data = []
        for rate in rates:
            rate_data.append({
                'date': rate.date.isoformat(),
                'currency': rate.currency,
                'tenor': rate.tenor,
                'floating_rate': rate.floating_rate,
                'rate': float(rate.rate),
                'rate_percent': float(rate.rate * 100)
            })
        
        return jsonify({'success': True, 'data': rate_data, 'count': len(rate_data)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/forward-pricing', methods=['POST'])
def forward_pricing():
    try:
        data = request.json
        from swap_pricer import SwapPricer
        pricer = SwapPricer(db)
        result = pricer.calculate_forward_rate(
            currency=data['currency'],
            start_tenor=data['start_tenor'],
            end_tenor=data['end_tenor']
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/import', methods=['POST'])
def import_data():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        # Import the file
        from excel_importer import ExcelImporter
        importer = ExcelImporter(db)
        result = importer.import_from_excel(tmp_path)
        
        # Clean up temp file
        import os
        os.unlink(tmp_path)
        
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
