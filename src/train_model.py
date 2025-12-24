"""
Katunayake Land Price Prediction - Model Training Module
Production-ready model training with cross-validation and comprehensive evaluation
"""

import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from preprocess import load_and_preprocess

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Configuration
TRAIN_CUTOFF_YEAR = 2018
TEST_START_YEAR = 2019
RANDOM_STATE = 42
N_CV_SPLITS = 5

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# If script is in project root, use SCRIPT_DIR; otherwise use parent
if os.path.exists(os.path.join(SCRIPT_DIR, "data")):
    PROJECT_ROOT = SCRIPT_DIR
else:
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_data():
    """Load and preprocess data"""
    csv_path = os.path.join(DATA_DIR, "katunayake_land_prices.csv")
    
    # Try alternative paths if not found
    if not os.path.exists(csv_path):
        csv_path = os.path.join(SCRIPT_DIR, "katunayake_land_prices.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, "katunayake_land_prices.csv")
    
    return load_and_preprocess(csv_path, add_features=True)


def split_data(X, y, df, cutoff_year=TRAIN_CUTOFF_YEAR):
    """
    Time-based train/test split
    
    Args:
        X: Features
        y: Target
        df: Full dataframe
        cutoff_year: Year to split on
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    train_mask = df["year"] <= cutoff_year
    
    X_train = X[train_mask]
    X_test = X[~train_mask]
    y_train = y[train_mask]
    y_test = y[~train_mask]
    
    print(f"\n📊 Train/Test Split:")
    print(f"   Training: {len(X_train)} samples (years ≤ {cutoff_year})")
    print(f"   Testing:  {len(X_test)} samples (years > {cutoff_year})")
    
    return X_train, X_test, y_train, y_test


def get_models():
    """
    Define models to train and compare
    
    Returns:
        dict: Model name -> model object
    """
    models = {
        "Linear Regression": LinearRegression(),
        
        "Ridge Regression": Ridge(
            alpha=10.0,
            random_state=RANDOM_STATE
        ),
        
        "Lasso Regression": Lasso(
            alpha=100.0,
            random_state=RANDOM_STATE,
            max_iter=10000
        ),
        
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=RANDOM_STATE
        ),
        
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }
    
    return models


def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained model
        X_train, X_test, y_train, y_test: Train/test data
        name: Model name
    
    Returns:
        dict: Evaluation metrics
    """
    # Predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "train_mae": mean_absolute_error(y_train, train_preds),
        "train_rmse": np.sqrt(mean_squared_error(y_train, train_preds)),
        "train_r2": r2_score(y_train, train_preds),
        "test_mae": mean_absolute_error(y_test, test_preds),
        "test_rmse": np.sqrt(mean_squared_error(y_test, test_preds)),
        "test_r2": r2_score(y_test, test_preds),
        "test_mape": mean_absolute_percentage_error(y_test, test_preds) * 100,
        "predictions": test_preds
    }
    
    # Cross-validation on training set
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    metrics["cv_mae_mean"] = -cv_scores.mean()
    metrics["cv_mae_std"] = cv_scores.std()
    
    return metrics


def print_results(results):
    """Print formatted model comparison results"""
    
    print("\n" + "="*80)
    print("📊 MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Create comparison table
    comparison = []
    for name, data in results.items():
        comparison.append({
            "Model": name,
            "Test MAE": f"Rs. {data['test_mae']:,.0f}",
            "Test RMSE": f"Rs. {data['test_rmse']:,.0f}",
            "Test R²": f"{data['test_r2']:.4f}",
            "MAPE": f"{data['test_mape']:.2f}%",
            "CV MAE": f"Rs. {data['cv_mae_mean']:,.0f}"
        })
    
    df_comparison = pd.DataFrame(comparison)
    print("\n" + df_comparison.to_string(index=False))
    
    # Find best model
    best_name = min(results, key=lambda x: results[x]["test_rmse"])
    best_metrics = results[best_name]
    
    print("\n" + "="*80)
    print(f"✅ BEST MODEL: {best_name}")
    print("="*80)
    print(f"Test MAE  : Rs. {best_metrics['test_mae']:,.2f}")
    print(f"Test RMSE : Rs. {best_metrics['test_rmse']:,.2f}")
    print(f"Test R²   : {best_metrics['test_r2']:.4f}")
    print(f"Test MAPE : {best_metrics['test_mape']:.2f}%")
    print(f"CV MAE    : Rs. {best_metrics['cv_mae_mean']:,.2f} (±{best_metrics['cv_mae_std']:,.2f})")
    
    return best_name


def analyze_by_area(y_test, predictions, df, test_mask):
    """Analyze prediction accuracy by area"""
    
    test_df = df[test_mask].copy()
    test_df["predicted_price"] = predictions
    test_df["error"] = test_df["predicted_price"] - test_df["price"]
    test_df["abs_error"] = np.abs(test_df["error"])
    test_df["pct_error"] = (test_df["abs_error"] / test_df["price"]) * 100
    
    print("\n" + "="*80)
    print("📍 PREDICTION ACCURACY BY AREA")
    print("="*80)
    
    area_analysis = []
    for area in ["urban", "little_away", "2km", "10km", "30km"]:
        area_data = test_df[test_df["area_type"] == area]
        if len(area_data) > 0:
            area_analysis.append({
                "Area": area,
                "MAE": f"Rs. {area_data['abs_error'].mean():,.0f}",
                "RMSE": f"Rs. {np.sqrt((area_data['error']**2).mean()):,.0f}",
                "MAPE": f"{area_data['pct_error'].mean():.2f}%",
                "Max Error": f"Rs. {area_data['abs_error'].max():,.0f}"
            })
    
    df_area = pd.DataFrame(area_analysis)
    print("\n" + df_area.to_string(index=False))
    
    return test_df


def save_model(model, scaler, encoder, model_name):
    """Save trained model and preprocessing objects"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save best model
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    encoder_path = os.path.join(MODELS_DIR, "encoder.pkl")
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.txt")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoder, encoder_path)
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        f.write(f"Model Training Metadata\n")
        f.write(f"="*50 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model Type: {model_name}\n")
        f.write(f"Training Data: {TRAIN_CUTOFF_YEAR} and earlier\n")
        f.write(f"Test Data: {TEST_START_YEAR} and later\n")
        f.write(f"Random State: {RANDOM_STATE}\n")
        f.write(f"CV Splits: {N_CV_SPLITS}\n")
    
    print(f"\n💾 Model saved to: {model_path}")
    print(f"💾 Scaler saved to: {scaler_path}")
    print(f"💾 Encoder saved to: {encoder_path}")
    print(f"💾 Metadata saved to: {metadata_path}")


def create_visualizations(results, test_df, best_name):
    """Create and save visualization plots"""
    
    print("\n📊 Creating visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Model Comparison Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # MAE Comparison
    models = list(results.keys())
    test_mae = [results[m]['test_mae'] for m in models]
    axes[0, 0].barh(models, test_mae, color='skyblue')
    axes[0, 0].set_xlabel('Mean Absolute Error (Rs.)')
    axes[0, 0].set_title('Test MAE Comparison')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # R² Comparison
    test_r2 = [results[m]['test_r2'] for m in models]
    axes[0, 1].barh(models, test_r2, color='lightgreen')
    axes[0, 1].set_xlabel('R² Score')
    axes[0, 1].set_title('Test R² Comparison')
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # MAPE Comparison
    test_mape = [results[m]['test_mape'] for m in models]
    axes[1, 0].barh(models, test_mape, color='salmon')
    axes[1, 0].set_xlabel('Mean Absolute Percentage Error (%)')
    axes[1, 0].set_title('Test MAPE Comparison')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # CV MAE
    cv_mae = [results[m]['cv_mae_mean'] for m in models]
    axes[1, 1].barh(models, cv_mae, color='plum')
    axes[1, 1].set_xlabel('Cross-Validation MAE (Rs.)')
    axes[1, 1].set_title('CV MAE Comparison')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    model_comp_path = os.path.join(REPORTS_DIR, 'model_comparison.png')
    plt.savefig(model_comp_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {model_comp_path}")
    plt.close()
    
    # 2. Actual vs Predicted
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(test_df['price'], test_df['predicted_price'], alpha=0.6, s=100)
    
    # Perfect prediction line
    min_val = min(test_df['price'].min(), test_df['predicted_price'].min())
    max_val = max(test_df['price'].max(), test_df['predicted_price'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Price (Rs.)', fontsize=12)
    ax.set_ylabel('Predicted Price (Rs.)', fontsize=12)
    ax.set_title(f'Actual vs Predicted Prices - {best_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    actual_pred_path = os.path.join(REPORTS_DIR, 'actual_vs_predicted.png')
    plt.savefig(actual_pred_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {actual_pred_path}")
    plt.close()
    
    # 3. Error Distribution by Area
    fig, ax = plt.subplots(figsize=(12, 6))
    
    areas = ["urban", "little_away", "2km", "10km", "30km"]
    area_errors = [test_df[test_df['area_type'] == area]['pct_error'].values for area in areas]
    
    bp = ax.boxplot(area_errors, labels=areas, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.set_xlabel('Area Type', fontsize=12)
    ax.set_ylabel('Percentage Error (%)', fontsize=12)
    ax.set_title('Prediction Error Distribution by Area', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    error_dist_path = os.path.join(REPORTS_DIR, 'error_distribution.png')
    plt.savefig(error_dist_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {error_dist_path}")
    plt.close()
    
    print("✅ Visualizations created successfully")


def main():
    """Main training pipeline"""
    
    print("\n" + "="*80)
    print("🏗️  KATUNAYAKE LAND PRICE PREDICTION - MODEL TRAINING")
    print("="*80)
    
    # Load data
    print("\n📂 Loading and preprocessing data...")
    X, y, df, scaler, encoder = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, df)
    test_mask = df["year"] > TRAIN_CUTOFF_YEAR
    
    # Get models
    models = get_models()
    
    print(f"\n🤖 Training {len(models)} models...")
    print("   This may take a few minutes...")
    
    # Train and evaluate all models
    results = {}
    for name, model in models.items():
        print(f"\n   Training {name}...", end=" ")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results[name] = {**metrics, "model": model}
        print("✓")
    
    # Print results
    best_name = print_results(results)
    
    # Analyze by area
    best_predictions = results[best_name]["predictions"]
    test_df = analyze_by_area(y_test, best_predictions, df, test_mask)
    
    # Save best model
    best_model = results[best_name]["model"]
    save_model(best_model, scaler, encoder, best_name)
    
    # Create visualizations
    create_visualizations(results, test_df, best_name)
    
    # Final recommendations
    print("\n" + "="*80)
    print("⚠️  IMPORTANT RECOMMENDATIONS")
    print("="*80)
    print("1. Model trained on data up to 2018, tested on 2019-2024")
    print("2. Predictions beyond 2024 are EXTRAPOLATIONS - use with caution")
    print("3. Recommended: Limit predictions to max 5 years in the future")
    print("4. Update model annually with new data for best accuracy")
    print("5. Monitor prediction errors and retrain if MAPE > 15%")
    print("6. Consider external factors: economic conditions, infrastructure projects")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"  1. Review reports in: {REPORTS_DIR}")
    print(f"  2. Use predict.py to make predictions")
    print(f"  3. Monitor model performance over time")


if __name__ == "__main__":
    main()