"""
Flask Web Application for Land Price Predictor
Modern web interface for Katunayake land price predictions
"""

from flask import Flask, render_template, request, jsonify
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from predict import LandPricePredictor, AREA_NAMES, AREA_LIST

# Configure Flask to use fontend directory for templates and static files
app = Flask(__name__, 
            template_folder='fontend',
            static_folder='fontend',
            static_url_path='/static')
app.config['SECRET_KEY'] = 'land-price-predictor-secret-key'

# Initialize predictor globally
predictor = None

def init_predictor():
    """Initialize the predictor"""
    global predictor
    try:
        if predictor is None:
            predictor = LandPricePredictor()
        return True
    except Exception as e:
        print(f"Error initializing predictor: {str(e)}")
        return False


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', area_names=AREA_NAMES, area_list=AREA_LIST)


@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle price prediction requests"""
    try:
        if not init_predictor():
            return jsonify({
                'success': False,
                'error': 'Predictor not initialized. Please ensure models are trained.'
            }), 500
        
        data = request.get_json()
        area_type = int(data.get('area_type'))
        predict_year = int(data.get('predict_year'))
        
        # Validate inputs
        if area_type < 0 or area_type >= len(AREA_LIST):
            return jsonify({
                'success': False,
                'error': f'Invalid area type. Must be between 0 and {len(AREA_LIST)-1}'
            }), 400
        
        if predict_year < 1994:
            return jsonify({
                'success': False,
                'error': 'Year must be 1994 or later'
            }), 400
        
        # Make prediction
        result = predictor.predict_price(area_type, predict_year)
        
        # Format response
        response = {
            'success': True,
            'data': {
                'area_name': result['area_name'].replace('_', ' ').title(),
                'predict_year': result['predict_year'],
                'predicted_price': round(result['predicted_price'], 2),
                'confidence_lower': round(result['confidence_lower'], 2) if result['confidence_lower'] else None,
                'confidence_upper': round(result['confidence_upper'], 2) if result['confidence_upper'] else None,
                'confidence_level': result['confidence_level'],
                'latest_price': round(result['latest_price'], 2),
                'latest_year': result['latest_year'],
                'is_extrapolation': result['is_extrapolation'],
                'extrapolation_years': result['extrapolation_years'],
                'reliability_score': result['reliability_score'],
                'price_change_percent': round(((result['predicted_price'] / result['latest_price']) - 1) * 100, 2) if result['predict_year'] > result['latest_year'] else 0,
                'years_diff': result['predict_year'] - result['latest_year']
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/investment', methods=['POST'])
def investment_analysis():
    """Handle investment analysis requests"""
    try:
        if not init_predictor():
            return jsonify({
                'success': False,
                'error': 'Predictor not initialized. Please ensure models are trained.'
            }), 500
        
        data = request.get_json()
        area_type = int(data.get('area_type'))
        purchase_price = float(data.get('purchase_price'))
        purchase_year = int(data.get('purchase_year'))
        sell_year = int(data.get('sell_year'))
        num_perches = float(data.get('num_perches'))
        
        # Validate inputs
        if area_type < 0 or area_type >= len(AREA_LIST):
            return jsonify({
                'success': False,
                'error': f'Invalid area type. Must be between 0 and {len(AREA_LIST)-1}'
            }), 400
        
        if purchase_price <= 0:
            return jsonify({
                'success': False,
                'error': 'Purchase price must be positive'
            }), 400
        
        if sell_year <= purchase_year:
            return jsonify({
                'success': False,
                'error': 'Sell year must be after purchase year'
            }), 400
        
        if num_perches <= 0:
            return jsonify({
                'success': False,
                'error': 'Number of perches must be positive'
            }), 400
        
        # Calculate investment returns
        result = predictor.calculate_investment_return(
            area_type, purchase_price, purchase_year, sell_year, num_perches
        )
        
        # Format response
        response = {
            'success': True,
            'data': {
                'area_name': result['area_name'].replace('_', ' ').title(),
                'num_perches': result['num_perches'],
                'purchase_year': result['purchase_year'],
                'sell_year': result['sell_year'],
                'holding_period_years': result['holding_period_years'],
                'purchase_price': round(result['purchase_price'], 2),
                'total_investment': round(result['total_investment'], 2),
                'predicted_price': round(result['predicted_price'], 2),
                'total_return': round(result['total_return'], 2),
                'profit': round(result['profit'], 2),
                'roi': round(result['roi'], 2),
                'cagr': round(result['cagr'], 2),
                'profit_lower': round(result['profit_lower'], 2) if result['profit_lower'] else None,
                'profit_upper': round(result['profit_upper'], 2) if result['profit_upper'] else None,
                'roi_lower': round(result['roi_lower'], 2) if result['roi_lower'] else None,
                'roi_upper': round(result['roi_upper'], 2) if result['roi_upper'] else None,
                'is_extrapolation': result['is_extrapolation'],
                'extrapolation_years': result['extrapolation_years'],
                'reliability_score': result['reliability_score'],
                'confidence_level': result['confidence_level']
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'predictor_initialized': predictor is not None
    })


if __name__ == '__main__':
    print("\n" + "="*80)
    print("KATUNAYAKE LAND PRICE PREDICTOR - WEB APPLICATION")
    print("="*80)
    print("\nInitializing predictor...")
    
    if init_predictor():
        print("\nPredictor initialized successfully!")
        print("\n" + "-"*80)
        print("Server starting on http://127.0.0.1:5000")
        print("-"*80)
        print("\nOpen your browser and navigate to: http://127.0.0.1:5000")
        print("Press CTRL+C to stop the server\n")
        
        app.run(debug=True, host='127.0.0.1', port=5000)
    else:
        print("\nERROR: Failed to initialize predictor!")
        print("Please ensure:")
        print("  1. You have run train_model.py first")
        print("  2. Models directory exists with trained model files")
        print("  3. Data file is accessible")
