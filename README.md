# Land Value Predictor - Katunayake Region

Machine learning system for predicting land prices in Katunayake, Sri Lanka using time-series regression with multiple ML algorithms.

## 🌟 Features

- **Modern Web Interface** - Beautiful, responsive web application with interactive forms
- **Command Line Interface** - Traditional terminal-based predictions
- Time-series land price prediction (1994-present)
- Investment analysis with ROI, CAGR, and profit projections
- Multiple ML models: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, Extra Trees
- Advanced feature engineering
- 90% confidence intervals for predictions
- 5 geographic zones: urban, little_away, 2km, 10km, 30km
- Cross-validation and model comparison

---

## System Requirements

- **Python**: 3.8 or higher
- **OS**: Windows, Linux, or macOS
- **RAM**: Minimum 4GB recommended
- **Disk Space**: ~100MB

---

## Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd Land_Value_Predictor-ml
```

### 2. Create Virtual Environment

**Windows (PowerShell):**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```bash
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
cd src
python train_model.py
```

**Output:**
- Trained models saved to `models/` directory
- Performance reports in `reports/` directory
- Model metadata in `models/model_metadata.txt`

### 5. Run the Application

**Option A: Web Application (Recommended)** ⭐

```bash
python app.py
```

Then open your browser and navigate to: **http://127.0.0.1:5000**

**Features:**
- Modern, responsive web interface
- Interactive forms with real-time validation
- Beautiful result displays with charts and cards
- Price prediction and investment analysis
- Works on desktop, tablet, and mobile

**Option B: Command Line Interface**

```bash
cd src
python predict.py
```

**Example:**
```
SELECT ANALYSIS TYPE:
  1. Price Prediction Only
  2. Investment Analysis (Buy & Sell)
  3. Exit

Enter your choice (1-3): 

SELECT AREA:
  0 → Urban (City Center)
  1 → Little Away from City
  2 → 2km from City
  3 → 10km from City
  4 → 30km from City

Enter area number (0-4): 
Enter prediction year (1994 onwards): 

================================================================================
LAND PRICE PREDICTION REPORT
================================================================================

Location: 
Prediction Year: 
Predicted Price per Perch: 
90% Confidence Interval:
   Lower Bound: 
   Upper Bound: 
```

---

## Project Structure

```
Land_Value_Predictor-ml/
│
├── app.py                             # Flask web application 
├── fontend/                           # Web interface files 
│   ├── index.html                     # Main webpage
│   ├── css/
│   │   └── style.css                  # Modern styling
│   └── js/
│       └── script.js                  # Interactive features
│
├── data/
│   └── katunayake_land_prices.csv    # Dataset (1994-2024)
│
├── models/
│   ├── best_model.pkl                 # Trained model
│   ├── scaler.pkl                     # Feature scaler
│   ├── encoder.pkl                    # Categorical encoder
│   └── model_metadata.txt             # Training info
│
├── reports/                           # Performance reports
│
├── src/
│   ├── preprocess.py                  # Data preprocessing
│   ├── train_model.py                 # Model training
│   └── predict.py                     # Prediction engine (CLI & API)
│
├── requirements.txt                   # Dependencies
├── README.md                          # This file
├── WEB_README.md                      # Web app documentation 
└── HOW_TO_RUN_WEBSITE.md             # Quick start guide 
```

---

## Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
joblib>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
flask>=2.0.0           # For web application 
flask-cors>=3.0.10     # For API support 
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Web Application (Recommended)

1. Start the server:
   ```bash
   python app.py
   ```

2. Open in browser: http://127.0.0.1:5000

3. Choose analysis type:
   - **Price Prediction Only**: Get predicted land prices for any year
   - **Investment Analysis**: Calculate ROI, profit, and CAGR for land investments

4. Fill in the form and submit

5. View beautiful, interactive results with:
   - Predicted prices
   - Confidence intervals
   - Reliability scores
   - Historical context
   - Investment returns

### Command Line Interface

1. Navigate to src directory:
   ```bash
   cd src
   ```

2. Run predictor:
   ```bash
   python predict.py
   ```

3. Follow the interactive prompts

---

## 📊 Analysis Types

### 1. Price Prediction Only
- Select area and year
- Get predicted price per perch
- View 90% confidence interval
- See reliability score
- Compare with historical data

### 2. Investment Analysis
- Input purchase details (price, year, land size)
- Enter projected selling year
- Get comprehensive analysis:
  - Total investment and returns
  - Estimated profit
  - ROI (Return on Investment)
  - CAGR (Compound Annual Growth Rate)
  - Confidence intervals for projections
  - Extrapolation warnings

---

## 🎨 Web Interface Features

- **Modern Design**: Gradient backgrounds, smooth animations, professional styling
- **Responsive Layout**: Works on desktop, tablet, and mobile devices
- **Interactive Forms**: Real-time validation and error handling
- **Visual Results**: Color-coded cards with icons and charts
- **Warning Systems**: Clear indicators for extrapolated predictions
- **Confidence Intervals**: Visual representation of prediction ranges
- **Easy Sharing**: Perfect for presentations and stakeholder reports

---

## Dataset

**File:** `data/katunayake_land_prices.csv`

| Column | Description |
|--------|-------------|
| year | Year (1994-2024) |
| area_type | Geographic zone |
| price | Land price per perch (LKR) |

**Geographic Zones:**
- `urban` - Central urban areas
- `little_away` - Peri-urban zones
- `2km` - 2km from center
- `10km` - 10km from center
- `30km` - 30km from center

---

## Model Information

**Training:**
- Training Data: 1994-2018
- Test Data: 2019+
- Validation: 5-fold time-series cross-validation
- Metrics: MAE, RMSE, R², MAPE

**Algorithms Evaluated:**
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Random Forest
5. Gradient Boosting
6. Extra Trees

---

## Troubleshooting

### Web Application Issues

**Port Already in Use:**
```bash
# Stop other Flask apps or change port in app.py
# Look for: app.run(debug=True, host='127.0.0.1', port=5000)
# Change 5000 to 5001 or 8080
```

**Models Not Found:**
```bash
# Train models first
cd src
python train_model.py
cd ..
python app.py
```

**Browser Not Opening:**
- Manually open: http://127.0.0.1:5000 or http://localhost:5000
- Check firewall settings

**Styling Issues:**
- Clear browser cache (Ctrl+Shift+Delete)
- Hard refresh (Ctrl+F5)

### Virtual Environment Issues

**Windows PowerShell execution policy error:**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Can't find python command:**
```bash
# Try python3 instead
python3 -m venv venv
```

### Import Errors

```bash
# Ensure virtual environment is activated
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Module Not Found

```bash
# Run from correct directory
cd src
python train_model.py
```


**Both interfaces use the same prediction engine!**

---

## Deactivating Virtual Environment

When finished:
```bash
deactivate
```

---

## License

MIT License