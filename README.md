# Land Value Predictor - Katunayake Region

Machine learning system for predicting land prices in Katunayake, Sri Lanka using time-series regression with multiple ML algorithms.

## Features

- Time-series land price prediction (1994-present)
- Multiple ML models: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, Extra Trees
- Advanced feature engineering
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

### 5. Make Predictions

```bash
python predict.py
```

**Example:**
```
Enter year (1994-2030): 2025
Select area type:
  1. urban
  2. little_away
  3. 2km
  4. 10km
  5. 30km
Enter choice (1-5): 1

Predicted Price: LKR 1,234,567
```

---

## Project Structure

```
Land_Value_Predictor-ml/
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
│   └── predict.py                     # Prediction interface
│
├── requirements.txt                   # Dependencies
└── README.md
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
```

Install all at once:
```bash
pip install -r requirements.txt
```

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

---

## Deactivating Virtual Environment

When finished:
```bash
deactivate
```

---

## License

MIT License