
"""
Katunayake Land Price Prediction - Prediction Module
Production-ready prediction interface with confidence intervals
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from preprocess import load_and_preprocess

# Configuration
START_YEAR = 1994
DATA_MAX_YEAR = 2024
CURRENT_YEAR = 2025  # Update this annually
CONFIDENCE_LEVEL = 0.90

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# If script is in project root, use SCRIPT_DIR; otherwise use parent
if os.path.exists(os.path.join(SCRIPT_DIR, "data")):
    PROJECT_ROOT = SCRIPT_DIR
else:
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Area mapping
AREA_LIST = ["urban", "little_away", "2km", "10km", "30km"]
AREA_NAMES = {
    0: "Urban (City Center)",
    1: "Little Away from City",
    2: "2km from City",
    3: "10km from City",
    4: "30km from City"
}


class LandPricePredictor:
    """Production-ready land price predictor with confidence intervals"""
    
    def __init__(self):
        """Initialize predictor by loading trained model and preprocessors"""
        
        # Load model
        model_path = os.path.join(MODELS_DIR, "best_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ERROR: Trained model not found at {model_path}\n"
                f"Please run train_model.py first!"
            )
        
        self.model = joblib.load(model_path)
        print(f"Model loaded: {type(self.model).__name__}")
        
        # Load scaler
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"ERROR: Scaler not found at {scaler_path}")
        
        self.scaler = joblib.load(scaler_path)
        print(f"Scaler loaded")
        
        # Load encoder
        encoder_path = os.path.join(MODELS_DIR, "encoder.pkl")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"ERROR: Encoder not found at {encoder_path}")
        
        self.encoder = joblib.load(encoder_path)
        print(f"Encoder loaded")
        
        # Load historical data
        csv_path = os.path.join(DATA_DIR, "katunayake_land_prices.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(SCRIPT_DIR, "katunayake_land_prices.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(PROJECT_ROOT, "katunayake_land_prices.csv")
        
        _, _, self.df, _, _ = load_and_preprocess(csv_path, add_features=True)
        print(f"Historical data loaded ({len(self.df)} records)\n")
    
    
    def _calculate_confidence_interval(self, area_encoded, predict_year):
        """Calculate confidence interval based on historical volatility"""
        area_name = AREA_LIST[int(area_encoded)]
        area_data = self.df[self.df["area_type"] == area_name]
        
        # Calculate historical volatility
        yoy_changes = area_data["price"].pct_change().dropna()
        volatility = yoy_changes.std()
        
        # Years beyond training data
        years_beyond = max(0, predict_year - DATA_MAX_YEAR)
        
        # Uncertainty calculation with aggressive scaling for long-term predictions
        base_uncertainty = volatility * 1.645  # 90% CI
        
        # Exponential penalty for far-future predictions (realistic for real estate)
        if years_beyond <= 5:
            extrapolation_penalty = 0.03 * years_beyond  # Modest increase for near-term
        elif years_beyond <= 10:
            extrapolation_penalty = 0.15 + 0.05 * (years_beyond - 5)  # Medium increase
        else:
            # For 10+ years: Significant uncertainty (common in 20+ year holdings)
            extrapolation_penalty = 0.40 + 0.08 * (years_beyond - 10)
        
        total_uncertainty = base_uncertainty + extrapolation_penalty
        
        return -total_uncertainty, total_uncertainty
    
    
    def predict_price(self, area_type, predict_year, return_confidence=True):
        """
        Predict land price per perch for given area and year
        
        Args:
            area_type: Area type (0-4 or name string)
            predict_year: Year to predict
            return_confidence: Whether to return confidence intervals
        
        Returns:
            dict: Prediction results with confidence intervals
        """
        
        # Convert area name to encoded value if needed
        if isinstance(area_type, str):
            try:
                area_encoded = AREA_LIST.index(area_type)
            except ValueError:
                raise ValueError(f"Invalid area type. Must be one of: {AREA_LIST}")
        else:
            area_encoded = int(area_type)
            if area_encoded < 0 or area_encoded >= len(AREA_LIST):
                raise ValueError(f"Area must be between 0 and {len(AREA_LIST)-1}")
        
        # Validate year
        if predict_year < START_YEAR:
            raise ValueError(f"Year must be >= {START_YEAR}")
        
        # Warning for far-future predictions (no hard limit)
        if predict_year > DATA_MAX_YEAR + 15:
            print(f"\nWARNING - EXTREME EXTRAPOLATION:")
            print(f"   Predicting {predict_year - DATA_MAX_YEAR} years beyond training data.")
            print(f"   Accuracy significantly decreases with longer time horizons.")
            print(f"   Use these predictions with extreme caution!\n")
        
        # Get latest price for this area
        area_name = AREA_LIST[area_encoded]
        latest_data = self.df[self.df["area_type"] == area_name].iloc[-1]
        latest_price = latest_data["price"]
        latest_year = int(latest_data["year"])
        
        # Prepare features
        years_since_1994 = predict_year - START_YEAR
        
        # Get historical data for feature engineering
        area_history = self.df[self.df["area_type"] == area_name].copy()
        
        # Use latest available values for lag features
        price_lag_1 = area_history["price"].iloc[-1]
        price_lag_2 = area_history["price"].iloc[-2] if len(area_history) > 1 else price_lag_1
        price_rolling_3 = area_history["price"].tail(3).mean()
        price_rolling_5 = area_history["price"].tail(5).mean()
        
        # Growth rates
        price_yoy_growth = (area_history["price"].iloc[-1] / area_history["price"].iloc[-2] - 1) if len(area_history) > 1 else 0
        growth_rolling_3 = area_history["price"].pct_change().tail(3).mean()
        
        # Polynomial features
        years_squared = years_since_1994 ** 2
        years_cubed = years_since_1994 ** 3
        
        # Interactions
        time_area_interaction = years_since_1994 * area_encoded
        time_squared_area = years_squared * area_encoded
        
        # Price ratio to urban
        urban_price = self.df[self.df["area_type"] == "urban"].iloc[-1]["price"]
        price_ratio_to_urban = latest_price / urban_price if area_name != "urban" else 1.0
        
        # Volatility
        price_volatility = area_history["price"].tail(3).std()
        
        # Create feature dataframe
        features = pd.DataFrame([[
            years_since_1994,
            area_encoded,
            price_lag_1,
            price_lag_2,
            price_rolling_3,
            price_rolling_5,
            price_yoy_growth,
            growth_rolling_3,
            years_squared,
            years_cubed,
            time_area_interaction,
            time_squared_area,
            price_ratio_to_urban,
            price_volatility
        ]], columns=[
            "years_since_1994",
            "area_encoded",
            "price_lag_1",
            "price_lag_2",
            "price_rolling_3",
            "price_rolling_5",
            "price_yoy_growth",
            "growth_rolling_3",
            "years_squared",
            "years_cubed",
            "time_area_interaction",
            "time_squared_area",
            "price_ratio_to_urban",
            "price_volatility"
        ])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        predicted_price = self.model.predict(features_scaled)[0]
        
        # Calculate confidence interval
        if return_confidence:
            lower_pct, upper_pct = self._calculate_confidence_interval(area_encoded, predict_year)
            price_lower = predicted_price * (1 + lower_pct)
            price_upper = predicted_price * (1 + upper_pct)
        else:
            price_lower = None
            price_upper = None
        
        # Determine extrapolation warning
        is_extrapolation = predict_year > DATA_MAX_YEAR
        extrapolation_years = max(0, predict_year - DATA_MAX_YEAR)
        
        # Reliability score
        reliability = max(0, 100 - (extrapolation_years * 10))
        
        return {
            "area_name": area_name,
            "area_encoded": area_encoded,
            "predict_year": predict_year,
            "predicted_price": predicted_price,
            "confidence_lower": price_lower,
            "confidence_upper": price_upper,
            "confidence_level": CONFIDENCE_LEVEL,
            "latest_price": latest_price,
            "latest_year": latest_year,
            "is_extrapolation": is_extrapolation,
            "extrapolation_years": extrapolation_years,
            "reliability_score": reliability
        }
    
    
    def calculate_investment_return(self, area_type, purchase_price, purchase_year, 
                                   sell_year, num_perches=1):
        """
        Calculate investment returns for land purchase
        
        Args:
            area_type: Area type (0-4 or name)
            purchase_price: Purchase price per perch
            purchase_year: Year of purchase
            sell_year: Year of sale
            num_perches: Number of perches
        
        Returns:
            dict: Investment analysis with profit, ROI, CAGR
        """
        
        if sell_year <= purchase_year:
            raise ValueError("Sell year must be after purchase year")
        
        # Predict future price
        prediction = self.predict_price(area_type, sell_year)
        
        # Calculate returns
        predicted_sell_price = prediction["predicted_price"]
        total_purchase = purchase_price * num_perches
        total_sell = predicted_sell_price * num_perches
        profit = total_sell - total_purchase
        roi = (profit / total_purchase) * 100
        
        # Calculate CAGR
        years = sell_year - purchase_year
        cagr = (((predicted_sell_price / purchase_price) ** (1/years)) - 1) * 100
        
        # With confidence intervals
        if prediction["confidence_lower"] and prediction["confidence_upper"]:
            profit_lower = (prediction["confidence_lower"] * num_perches) - total_purchase
            profit_upper = (prediction["confidence_upper"] * num_perches) - total_purchase
            roi_lower = (profit_lower / total_purchase) * 100
            roi_upper = (profit_upper / total_purchase) * 100
        else:
            profit_lower = profit_upper = None
            roi_lower = roi_upper = None
        
        return {
            **prediction,
            "purchase_price": purchase_price,
            "purchase_year": purchase_year,
            "sell_year": sell_year,
            "num_perches": num_perches,
            "total_investment": total_purchase,
            "total_return": total_sell,
            "profit": profit,
            "roi": roi,
            "cagr": cagr,
            "profit_lower": profit_lower,
            "profit_upper": profit_upper,
            "roi_lower": roi_lower,
            "roi_upper": roi_upper,
            "holding_period_years": years
        }
    
    
    def plot_prediction(self, area_type, predict_year, save_path=None):
        """Create visualization of historical trend and prediction"""
        
        # Get prediction
        prediction = self.predict_price(area_type, predict_year)
        area_name = prediction["area_name"]
        
        # Get historical data
        area_data = self.df[self.df["area_type"] == area_name].copy()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot historical data
        ax.plot(area_data["year"], area_data["price"], 
               marker='o', linewidth=2, markersize=6, 
               label='Historical Prices', color='blue')
        
        # Plot prediction
        ax.plot([DATA_MAX_YEAR, predict_year], 
               [area_data["price"].iloc[-1], prediction["predicted_price"]], 
               'r--', linewidth=2, label='Predicted Price')
        
        ax.scatter([predict_year], [prediction["predicted_price"]], 
                  color='red', s=200, marker='*', 
                  label=f'{predict_year} Prediction', zorder=5)
        
        # Add confidence interval
        if prediction["confidence_lower"] and prediction["confidence_upper"]:
            ax.fill_between(
                [DATA_MAX_YEAR, predict_year],
                [area_data["price"].iloc[-1], prediction["confidence_lower"]],
                [area_data["price"].iloc[-1], prediction["confidence_upper"]],
                alpha=0.2, color='red',
                label=f'{int(CONFIDENCE_LEVEL*100)}% Confidence Interval'
            )
        
        # Formatting
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price per Perch (Rs.)', fontsize=12, fontweight='bold')
        ax.set_title(f'Land Price Prediction - {area_name.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rs. {x/1e6:.1f}M'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()


def print_prediction_report(result):
    """Print formatted prediction report"""
    
    print("\n" + "="*80)
    print("LAND PRICE PREDICTION REPORT")
    print("="*80)
    
    print(f"\nLocation: {result['area_name'].replace('_', ' ').title()}")
    print(f"Prediction Year: {result['predict_year']}")
    
    if result['is_extrapolation']:
        print(f"WARNING - EXTRAPOLATION: {result['extrapolation_years']} years beyond training data")
        print(f"   Reliability Score: {result['reliability_score']}%")
    
    print(f"\nPRICE PREDICTION:")
    print(f"   Predicted Price per Perch: Rs. {result['predicted_price']:,.2f}")
    
    if result['confidence_lower'] and result['confidence_upper']:
        print(f"   {int(CONFIDENCE_LEVEL*100)}% Confidence Interval:")
        print(f"      Lower Bound: Rs. {result['confidence_lower']:,.2f}")
        print(f"      Upper Bound: Rs. {result['confidence_upper']:,.2f}")
    
    print(f"\nCONTEXT:")
    print(f"   Latest Known Price ({result['latest_year']}): Rs. {result['latest_price']:,.2f}")
    
    if result['predict_year'] > result['latest_year']:
        price_change = ((result['predicted_price'] / result['latest_price']) - 1) * 100
        years_diff = result['predict_year'] - result['latest_year']
        print(f"   Expected Change: {price_change:+.2f}% over {years_diff} years")


def print_investment_report(result):
    """Print formatted investment analysis report"""
    
    print("\n" + "="*80)
    print("INVESTMENT ANALYSIS REPORT")
    print("="*80)
    
    print(f"\nLocation: {result['area_name'].replace('_', ' ').title()}")
    print(f"Land Size: {result['num_perches']} perch(es)")
    
    print(f"\nTIMELINE:")
    print(f"   Purchase Year: {result['purchase_year']}")
    print(f"   Selling Year: {result['sell_year']}")
    print(f"   Holding Period: {result['holding_period_years']} years")
    
    print(f"\nINVESTMENT DETAILS:")
    print(f"   Purchase Price per Perch: Rs. {result['purchase_price']:,.2f}")
    print(f"   Total Investment: Rs. {result['total_investment']:,.2f}")
    
    print(f"\nPROJECTED RETURNS:")
    print(f"   Predicted Selling Price per Perch: Rs. {result['predicted_price']:,.2f}")
    print(f"   Total Return: Rs. {result['total_return']:,.2f}")
    print(f"   Estimated Profit: Rs. {result['profit']:,.2f}")
    print(f"   ROI: {result['roi']:.2f}%")
    print(f"   CAGR: {result['cagr']:.2f}% per year")
    
    if result['profit_lower'] and result['profit_upper']:
        print(f"\n{int(CONFIDENCE_LEVEL*100)}% CONFIDENCE INTERVAL:")
        print(f"   Profit Range: Rs. {result['profit_lower']:,.2f} to Rs. {result['profit_upper']:,.2f}")
        print(f"   ROI Range: {result['roi_lower']:.2f}% to {result['roi_upper']:.2f}%")
    
    if result['is_extrapolation']:
        years_beyond = result['extrapolation_years']
        print(f"\nWARNING - EXTRAPOLATION:")
        print(f"   This prediction extends {years_beyond} years beyond training data (2024)")
        print(f"   Reliability Score: {result['reliability_score']}%")
        
        if years_beyond >= 20:
            print(f"\nLONG-TERM INVESTMENT CONSIDERATIONS (20+ years):")
            print(f"   \u2022 Economic cycles: Multiple recessions/booms expected")
            print(f"   \u2022 Infrastructure: Major projects may alter land values")
            print(f"   \u2022 Policy changes: Zoning, taxes, regulations unpredictable")
            print(f"   \u2022 Market disruptions: Technology, climate, demographics")
            print(f"\n   RECOMMENDATION:")
            print(f"   1. Use this as ONE data point, not the only indicator")
            print(f"   2. Re-run predictions every 2-3 years with updated data")
            print(f"   3. Consult local real estate experts and urban planners")
            print(f"   4. Consider worst-case scenarios (confidence interval lower bound)")
            print(f"   5. Build in safety margins (assume 20-30% lower returns)")
        elif years_beyond >= 10:
            print(f"\n   RECOMMENDATION: Re-check predictions every 2-3 years")
            print(f"      Retrain model annually with new data for better accuracy")
        else:
            print(f"   Consider this a reasonable estimate with moderate uncertainty")
    
    print("\n" + "="*80)


def interactive_cli():
    """Interactive command-line interface for predictions"""
    
    print("\n" + "="*80)
    print("KATUNAYAKE LAND PRICE PREDICTOR")
    print("="*80)
    
    try:
        # Initialize predictor
        predictor = LandPricePredictor()
        
        while True:
            print("\n" + "-"*80)
            print("SELECT ANALYSIS TYPE:")
            print("  1. Price Prediction Only")
            print("  2. Investment Analysis (Buy & Sell)")
            print("  3. Exit")
            print("-"*80)
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "3":
                print("\nThank you for using Land Price Predictor!")
                break
            
            if choice not in ["1", "2"]:
                print("ERROR: Invalid choice. Please enter 1, 2, or 3.")
                continue
            
            # Get area input
            print("\nSELECT AREA:")
            for idx, area in enumerate(AREA_LIST):
                print(f"  {idx} → {AREA_NAMES[idx]}")
            
            while True:
                try:
                    area_idx = int(input("\nEnter area number (0-4): "))
                    if 0 <= area_idx < len(AREA_LIST):
                        break
                    print(f"ERROR: Please enter a number between 0 and {len(AREA_LIST)-1}")
                except ValueError:
                    print("ERROR: Invalid input. Please enter a number.")
            
            if choice == "1":
                # Price prediction only
                while True:
                    try:
                        predict_year = int(input(f"\nEnter prediction year ({START_YEAR} onwards): "))
                        if predict_year >= START_YEAR:
                            break
                        print(f"ERROR: Year must be {START_YEAR} or later")
                    except ValueError:
                        print("ERROR: Invalid input. Please enter a valid year.")
                
                # Make prediction
                result = predictor.predict_price(area_idx, predict_year)
                print_prediction_report(result)
                
                # Ask to plot
                plot_choice = input("\nGenerate visualization? (y/n): ").strip().lower()
                if plot_choice == 'y':
                    predictor.plot_prediction(area_idx, predict_year)
            
            elif choice == "2":
                # Investment analysis
                while True:
                    try:
                        purchase_price = float(input("\nEnter purchase price per perch (Rs.): "))
                        if purchase_price > 0:
                            break
                        print("ERROR: Price must be positive")
                    except ValueError:
                        print("ERROR: Invalid input. Please enter a number.")
                
                while True:
                    try:
                        purchase_year = int(input(f"Enter purchase year ({START_YEAR}-{CURRENT_YEAR + 2}): "))
                        if START_YEAR <= purchase_year <= CURRENT_YEAR + 2:
                            break
                        print(f"ERROR: Year must be between {START_YEAR} and {CURRENT_YEAR + 2}")
                    except ValueError:
                        print("ERROR: Invalid input.")
                
                while True:
                    try:
                        sell_year = int(input(f"Enter selling year (after {purchase_year}): "))
                        if sell_year > purchase_year:
                            break
                        print(f"ERROR: Year must be after {purchase_year}")
                    except ValueError:
                        print("ERROR: Invalid input.")
                
                while True:
                    try:
                        num_perches = float(input("Enter number of perches: "))
                        if num_perches > 0:
                            break
                        print("ERROR: Number of perches must be positive")
                    except ValueError:
                        print("ERROR: Invalid input.")
                
                # Calculate investment returns
                result = predictor.calculate_investment_return(
                    area_idx, purchase_price, purchase_year, sell_year, num_perches
                )
                print_investment_report(result)
            
            # Ask to continue
            continue_choice = input("\nMake another prediction? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("\nThank you for using Land Price Predictor!")
                break
    
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nPlease ensure:")
        print("  1. You have run train_model.py first")
        print("  2. Models directory exists with trained model files")
        print("  3. Data file is accessible")


if __name__ == "__main__":
    interactive_cli()