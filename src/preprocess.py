"""
Katunayake Land Price Prediction - Data Preprocessing Module
Production-ready preprocessing with advanced feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

START_YEAR = 1994

def load_and_preprocess(csv_path, add_features=True):
    """
    Load and preprocess land price data with advanced feature engineering
    
    Args:
        csv_path (str): Path to CSV file
        add_features (bool): Whether to add engineered features
    
    Returns:
        tuple: (X, y, df, scaler, encoder)
            - X: Feature DataFrame (scaled)
            - y: Target prices (Series)
            - df: Full processed DataFrame
            - scaler: Fitted StandardScaler
            - encoder: Fitted OrdinalEncoder
    """
    
    print(f"📂 Loading data from: {csv_path}")
    
    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Data file not found: {csv_path}")
    except Exception as e:
        raise Exception(f"❌ Error reading CSV: {str(e)}")
    
    # Validate required columns
    required_cols = ['year', 'area_type', 'price']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Missing required columns: {missing_cols}")
    
    # Remove any duplicates
    df = df.drop_duplicates(subset=['year', 'area_type'])
    
    # Sort by time (CRITICAL for time-series)
    df = df.sort_values(["year", "area_type"]).reset_index(drop=True)
    
    print(f"✅ Loaded {len(df)} records from {df['year'].min()} to {df['year'].max()}")
    
    # Ordinal encoding based on closeness to town
    area_order = [["urban", "little_away", "2km", "10km", "30km"]]
    encoder = OrdinalEncoder(categories=area_order)
    df["area_encoded"] = encoder.fit_transform(df[["area_type"]])
    
    # Base time feature
    df["years_since_1994"] = df["year"] - START_YEAR
    
    # Feature Engineering
    if add_features:
        print("🔧 Engineering features...")
        
        # 1. LAG FEATURES (previous year's price per area)
        df["price_lag_1"] = df.groupby("area_type")["price"].shift(1)
        df["price_lag_2"] = df.groupby("area_type")["price"].shift(2)
        
        # 2. ROLLING AVERAGES (capture trends)
        df["price_rolling_3"] = df.groupby("area_type")["price"].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df["price_rolling_5"] = df.groupby("area_type")["price"].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        
        # 3. GROWTH RATE (year-over-year change)
        df["price_yoy_growth"] = df.groupby("area_type")["price"].pct_change()
        
        # 4. ROLLING GROWTH RATE (smoothed growth)
        df["growth_rolling_3"] = df.groupby("area_type")["price_yoy_growth"].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # 5. POLYNOMIAL TIME FEATURES (non-linear trends)
        df["years_squared"] = df["years_since_1994"] ** 2
        df["years_cubed"] = df["years_since_1994"] ** 3
        
        # 6. INTERACTION FEATURES (different growth rates per area)
        df["time_area_interaction"] = df["years_since_1994"] * df["area_encoded"]
        df["time_squared_area"] = df["years_squared"] * df["area_encoded"]
        
        # 7. PRICE RATIOS (relative to urban area)
        urban_prices = df[df["area_type"] == "urban"].set_index("year")["price"]
        df["price_ratio_to_urban"] = df.apply(
            lambda row: row["price"] / urban_prices[row["year"]] if row["area_type"] != "urban" else 1.0,
            axis=1
        )
        
        # 8. VOLATILITY (standard deviation of recent prices)
        df["price_volatility"] = df.groupby("area_type")["price"].transform(
            lambda x: x.rolling(window=3, min_periods=1).std()
        )
        
        # Fill NaN values intelligently
        df["price_lag_1"] = df["price_lag_1"].fillna(df["price"])
        df["price_lag_2"] = df["price_lag_2"].fillna(df["price"])
        df["price_yoy_growth"] = df["price_yoy_growth"].fillna(0)
        df["growth_rolling_3"] = df["growth_rolling_3"].fillna(0)
        df["price_volatility"] = df["price_volatility"].fillna(0)
        
        print("✅ Features engineered successfully")
    
    # Select features for modeling
    feature_cols = ["years_since_1994", "area_encoded"]
    
    if add_features:
        feature_cols.extend([
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
    
    X = df[feature_cols].copy()
    y = df["price"].copy()
    
    # Scale features (CRITICAL for model performance)
    print("📊 Scaling features...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    print(f"✅ Final feature set: {len(X_scaled.columns)} features")
    
    return X_scaled, y, df, scaler, encoder


def detect_outliers(df, column="price", method="iqr", threshold=1.5):
    """
    Detect outliers in price data using IQR or Z-score method
    
    Args:
        df (DataFrame): Input dataframe
        column (str): Column to check for outliers
        method (str): 'iqr' or 'zscore'
        threshold (float): Threshold for outlier detection
    
    Returns:
        DataFrame: Original dataframe with 'is_outlier' column added
    """
    
    df = df.copy()
    
    if method == "iqr":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df["is_outlier"] = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == "zscore":
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column]))
        df["is_outlier"] = z_scores > threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    n_outliers = df["is_outlier"].sum()
    print(f"🔍 Detected {n_outliers} outliers using {method.upper()} method")
    
    return df


def get_data_summary(df):
    """
    Generate summary statistics for the dataset
    
    Args:
        df (DataFrame): Input dataframe
    
    Returns:
        dict: Summary statistics
    """
    
    summary = {
        "total_records": len(df),
        "year_range": (int(df["year"].min()), int(df["year"].max())),
        "num_years": df["year"].nunique(),
        "num_areas": df["area_type"].nunique(),
        "areas": df["area_type"].unique().tolist(),
        "price_stats": {
            "min": float(df["price"].min()),
            "max": float(df["price"].max()),
            "mean": float(df["price"].mean()),
            "median": float(df["price"].median()),
            "std": float(df["price"].std())
        }
    }
    
    return summary


if __name__ == "__main__":
    import os
    
    # Construct path to data file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try data directory relative to script location
    csv_path = os.path.join(script_dir, "data", "katunayake_land_prices.csv")
    
    # Alternative: if data is in same directory as script
    if not os.path.exists(csv_path):
        csv_path = os.path.join(script_dir, "katunayake_land_prices.csv")
    
    # Alternative: if script is in src/ and data is in ../data/
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(script_dir), "data", "katunayake_land_prices.csv")
    
    print("="*70)
    print("KATUNAYAKE LAND PRICE PREDICTION - DATA PREPROCESSING")
    print("="*70 + "\n")
    
    try:
        # Load and preprocess
        X, y, df, scaler, encoder = load_and_preprocess(csv_path, add_features=True)
        
        # Get summary
        summary = get_data_summary(df)
        
        print("\n" + "="*70)
        print("📊 DATA SUMMARY")
        print("="*70)
        print(f"Total Records    : {summary['total_records']}")
        print(f"Year Range       : {summary['year_range'][0]} - {summary['year_range'][1]}")
        print(f"Number of Years  : {summary['num_years']}")
        print(f"Number of Areas  : {summary['num_areas']}")
        print(f"Areas            : {', '.join(summary['areas'])}")
        print(f"\nPrice Statistics:")
        print(f"  Min            : Rs. {summary['price_stats']['min']:,.2f}")
        print(f"  Max            : Rs. {summary['price_stats']['max']:,.2f}")
        print(f"  Mean           : Rs. {summary['price_stats']['mean']:,.2f}")
        print(f"  Median         : Rs. {summary['price_stats']['median']:,.2f}")
        print(f"  Std Dev        : Rs. {summary['price_stats']['std']:,.2f}")
        
        # Detect outliers
        print("\n" + "="*70)
        print("🔍 OUTLIER DETECTION")
        print("="*70)
        df_with_outliers = detect_outliers(df, column="price", method="iqr")
        outliers = df_with_outliers[df_with_outliers["is_outlier"] == True]
        
        if len(outliers) > 0:
            print("\nOutliers found:")
            print(outliers[["year", "area_type", "price"]].to_string(index=False))
        else:
            print("No outliers detected")
        
        # Show first few rows
        print("\n" + "="*70)
        print("📋 SAMPLE DATA (First 10 rows)")
        print("="*70)
        print(df[["year", "area_type", "price", "years_since_1994", "area_encoded"]].head(10).to_string(index=False))
        
        print("\n" + "="*70)
        print("✅ PREPROCESSING COMPLETED SUCCESSFULLY")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nPlease ensure:")
        print("  1. CSV file exists at the correct path")
        print("  2. CSV has columns: year, area_type, price")
        print("  3. Data is in correct format")