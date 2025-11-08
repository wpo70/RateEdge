from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import sys
import os

# Add backend to sys.path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Import with error handling
try:
    from database_models import DatabaseManager, SwapRate
except ImportError as e:
    raise RuntimeError(f"Could not import from backend/database_models.py: {e}")

# Initialize Flask with explicit paths
app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
CORS(app)

# Database setup - flexible path
db_path = os.environ.get("RATEEDGE_DB_PATH") or '/home/site/wwwroot/database/swap_rates.db'
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# Initialize database
db = DatabaseManager(f'sqlite:///{db_path}')

# Create tables if they don't exist
try:
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    if not inspector.has_table('swap_rates'):
        print("Creating database tables...")
        SwapRate.__table__.create(db.engine, checkfirst=True)
        print("Tables created successfully")
except Exception as e:
    print(f"Database initialization: {e}")

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

@app.route('/data/view')
def data_view_page():
    return render_template('data_view.html')

@app.route('/data/import')
def data_import_page():
    return render_template('data_import.html')

# ===== API ROUTES =====
@app.route('/api/data', methods=['GET'])
def get_data():
    try:
        currency = request.args.get('currency')
        tenor = request.args.get('tenor')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        rates = db.get_rates(currency, tenor, start_date, end_date)
        
        data = [{
            'id': r.id,
            'date': r.date.isoformat(),
            'currency': r.currency,
            'tenor': r.tenor,
            'floating_rate': r.floating_rate,
            'rate': r.rate
        } for r in rates]
        
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        print(f"Error in get_data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        stats = db.get_statistics()
        return jsonify({'success': True, 'data': stats})
    except Exception as e:
        print(f"Error in get_stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/currencies', methods=['GET'])
def get_currencies():
    try:
        currencies = db.get_currencies()
        return jsonify({'success': True, 'data': currencies})
    except Exception as e:
        print(f"Error in get_currencies: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/tenors', methods=['GET'])
def get_tenors():
    try:
        currency = request.args.get('currency')
        tenors = db.get_tenors(currency)
        return jsonify({'success': True, 'data': tenors})
    except Exception as e:
        print(f"Error in get_tenors: {e}")
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
        
        print(f"Importing file: {file.filename} from {tmp_path}")
        
        # Import the file
        from excel_importer import ExcelImporter
        importer = ExcelImporter(db)
        result = importer.import_from_excel(tmp_path)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        print(f"Import result: {result}")
        
        # Verify data was saved
        from database_models import SwapRate
        count = db.session.query(SwapRate).count()
        print(f"Total records in database after import: {count}")
        
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        print(f"Error in import_data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/forward-rate', methods=['POST'])
def calculate_forward_rate():
    try:
        data = request.json
        currency = data.get('currency')
        start_tenor = data.get('start_tenor')
        end_tenor = data.get('end_tenor')
        
        from swap_pricer import ForwardSwapPricer
        pricer = ForwardSwapPricer(db)
        
        result = pricer.calculate_forward_rate(currency, start_tenor, end_tenor)
        
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        print(f"Error in calculate_forward_rate: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/spread', methods=['GET'])
def get_spread_analysis():
    try:
        currency = request.args.get('currency')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        from analytics import SpreadAnalyzer
        analyzer = SpreadAnalyzer(db)
        
        result = analyzer.analyze_spreads(currency, start_date, end_date)
        
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        print(f"Error in get_spread_analysis: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 8000))
        print(f"Starting RateEdge on port {port}")
        print(f"Database path: {db_path}")
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as exc:
        print(f"Failed to launch RateEdge: {exc}")
        import traceback
        traceback.print_exc()
