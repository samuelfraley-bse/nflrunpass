# NFL Play Caller

Machine learning model to predict run vs pass plays in NFL games based on pre-snap situational features. Built using logistic regression on 2021-2023 play-by-play data.

## Project Overview

This project implements a binary classification model that predicts whether an NFL offensive play will be a run or pass based on 18 pre-snap features. The model uses logistic regression with hyperparameter tuning via GridSearchCV.

### Key Features
- 50+ engineered pre-snap features (field position, down/distance, score state, time, formation)
- Logistic regression with GridSearchCV hyperparameter optimization
- FastAPI backend for real-time predictions
- Streamlit UI for interactive predictions
- Comprehensive unit tests for features and preprocessing
- Modular, scalable library architecture

## Project Structure

```
nflrunpass/
├── src/nfl_run_pass/          # Core ML library
│   ├── config.py              # Central configuration
│   ├── data_loading.py        # Data ingestion and filtering
│   ├── preprocessing.py       # Train/test split, scaling
│   ├── features.py            # Feature engineering
│   ├── models.py              # Model training (logistic regression)
│   ├── evaluation.py          # Metrics and evaluation
│   ├── pipeline.py            # End-to-end training pipeline
│   └── tuning.py              # Hyperparameter tuning utilities
├── api/                       # FastAPI backend
│   └── main.py                # REST API for predictions
├── streamlit/                 # Frontend applications
│   └── streamlit_app.py       # UI
├── models/                    # Saved artifacts
│   ├── log_reg_model.pkl      # Trained model
│   ├── scaler.pkl             # Fitted StandardScaler
│   └── feature_cols.json      # Feature column names (ordered)
├── data/raw/                  # Data files (gitignored)
│   └── pbp_2021_2023.csv      # NFL play-by-play dataset
├── tests/                     # Unit tests
│   ├── test_data_loading.py
│   ├── test_features.py
│   └── test_preprocessing.py
├── run_pipeline.py            # Script to train model
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project metadata
└── README.md                  # This file
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone <repository-url>
cd nflrunpass
```

### 2. Create virtual environment
```bash
python -m venv .venv
```

### 3. Activate environment
**Windows:**
```bash
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

Or using uv:
```bash
pip install uv
uv pip install -e .
```

### 5. Prepare data
Place your play-by-play CSV at `data/raw/pbp_2021_2023.csv`. The dataset should contain nflfastR-style columns including:
- season, week, game_id, play_id
- play_type (run/pass)
- down, ydstogo, yardline_100
- shotgun, no_huddle, qtr
- posteam_score, defteam_score, score_differential
- half_seconds_remaining, game_seconds_remaining
- posteam, home_team, away_team

Data source: [nflfastR](https://www.nflfastr.com/)

## Running the Project

### Train the model
```bash
python run_pipeline.py
```

This will:
- Load and filter 2023 season data
- Engineer 50+ pre-snap features
- Split train/test (80/20, stratified)
- Scale features with StandardScaler
- Train logistic regression with GridSearchCV
- Evaluate on multiple metrics
- Save model artifacts to `models/`

### Run Streamlit UI
```bash
streamlit run streamlit/streamlit_app.py
```

### Run FastAPI backend
```bash
cd api
uvicorn main:app --reload
```

API will be available at `http://localhost:8000`. Interactive documentation at `http://localhost:8000/docs`.

### Run tests
```bash
pytest tests/ -v
```

## Model Details

### Features (50+ total)

**Base numeric (3):**
- down, ydstogo, yardline_100

**Engineered features (47+):**
- **Field Position:** red zone, goal line, distance buckets (short/medium/long)
- **Formation:** shotgun, no_huddle
- **Score State:** trailing/tied/leading, multi-score deficit, blowout situations
- **Time Context:** quarter flags, 2-minute drill (context-aware), late-game scenarios
- **Short Yardage:** 3rd & 1/2/3, 4th & 1/2/3
- **Down × Distance:** interaction terms, long yardage flags
- **Advanced:** score × time pressure, home/away

### Model Architecture
- **Algorithm:** Logistic Regression with L2 regularization
- **Preprocessing:** StandardScaler, stratified 80/20 split
- **Hyperparameter Tuning:** GridSearchCV (3-fold CV, F1 scoring)
  - C: [0.01, 0.1, 1.0, 10.0]
  - class_weight: [None, "balanced"]

### Performance
- Test accuracy: 70.1%
- Test F1: 0.764
- Test ROC-AUC: 0.745

## API Usage

### Endpoints

**Health Check:**
- `GET /` - Basic API status
- `GET /health` - Model status and feature information

**Prediction:**
- `POST /predict` - Predict run or pass play

**Interactive Documentation:**
- Visit `http://localhost:8000/docs` for Swagger UI with try-it-out functionality

### Request Format
```json
{
  "down": 3,
  "ydstogo": 7,
  "yardline_100": 35,
  "offense_score": 14,
  "defense_score": 17,
  "qtr": 4,
  "seconds_remaining_half": 180,
  "shotgun": true,
  "no_huddle": false,
  "is_home_offense": true
}
```

**Input Validation:**
- `down`: 1-4
- `ydstogo`: 0-99
- `yardline_100`: 0-100
- `qtr`: 1-4
- `seconds_remaining_half`: 0-1800
- Scores: non-negative integers
- Booleans: true/false

### Response Format
```json
{
  "prediction": "PASS",
  "prob_pass": 0.73,
  "prob_run": 0.27,
  "confidence": "High"
}
```

**Confidence Levels:**
- High: probability ≥ 0.70
- Medium: probability 0.55-0.69
- Low: probability < 0.55

### Example with curl
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "down": 3,
    "ydstogo": 7,
    "yardline_100": 35,
    "offense_score": 14,
    "defense_score": 17,
    "qtr": 4,
    "seconds_remaining_half": 180,
    "shotgun": true,
    "no_huddle": false,
    "is_home_offense": true
  }'
```

## Configuration

All settings are in `src/nfl_run_pass/config.py`:

```python
from nfl_run_pass.config import CONFIG

# Access configuration
CONFIG.data.season           # 2023
CONFIG.train_test.test_size  # 0.2
CONFIG.log_reg.param_grid    # GridSearchCV params
```

To modify:
```python
CONFIG.data.season = 2022
CONFIG.train_test.test_size = 0.25
```

## Library Scaling Guidelines

This library is designed to be extended by new contributors. Follow these guidelines to add new components.

### Adding a New Feature

1. **Edit `src/nfl_run_pass/features.py`** in the `add_engineered_features()` function:

```python
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    # ... existing features ...
    
    # Add your new feature
    df["my_new_feature"] = (df["some_column"] > threshold).astype(int)
    
    return df
```

2. **Register in `config.py`:**

```python
@dataclass
class FeatureConfig:
    engineered_feature_candidates: Tuple[str, ...] = (
        "is_red_zone",
        # ... existing features ...
        "my_new_feature",  # Add here
    )
```

3. **Write unit tests in `tests/test_features.py`:**

```python
def test_my_new_feature():
    df = pd.DataFrame({"some_column": [10, 20, 30]})
    df_feat = add_engineered_features(df)
    
    assert "my_new_feature" in df_feat.columns
    assert df_feat["my_new_feature"][0] == expected_value
```

4. **Retrain the model:**

```bash
python run_pipeline.py
```

5. **Update API** (`api/main.py`) to include new feature calculation in `build_feature_df()`.

### Adding a New Preprocessor

1. **Add function to `preprocessing.py`:**

```python
def my_new_preprocessor(X: pd.DataFrame) -> pd.DataFrame:
    """Apply new preprocessing step."""
    X = X.copy()
    # Your preprocessing logic
    return X
```

2. **Integrate into `prepare_train_test_data()`:**

```python
def prepare_train_test_data(df_model, target_col=None, scale=True):
    X, y, feature_cols = build_feature_matrix(df_model, target_col)
    X = handle_missing_values(X)
    X = my_new_preprocessor(X)  # Add here
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    # ...
```

3. **Write tests in `tests/test_preprocessing.py`:**

```python
def test_my_new_preprocessor():
    X = pd.DataFrame({"col": [1, 2, 3]})
    X_processed = my_new_preprocessor(X)
    assert "expected_column" in X_processed.columns
```

### Adding a New Model

1. **Create model functions in `models.py`:**

```python
from sklearn.ensemble import RandomForestClassifier

def create_random_forest_model(n_estimators=100):
    return RandomForestClassifier(n_estimators=n_estimators, random_state=42)

def train_random_forest_model(X_train, y_train, use_grid_search=False):
    model = create_random_forest_model()
    
    if use_grid_search:
        param_grid = {"n_estimators": [50, 100, 200]}
        gs = GridSearchCV(model, param_grid, cv=3, scoring="f1")
        gs.fit(X_train, y_train)
        return gs.best_estimator_, {"best_params": gs.best_params_}
    
    model.fit(X_train, y_train)
    return model, {}
```

2. **Add config in `config.py`:**

```python
@dataclass
class RandomForestConfig:
    n_estimators: int = 100
    use_grid_search: bool = True
```

3. **Update pipeline** to support model selection:

```python
def run_training_pipeline(model_type="logistic_regression"):
    # ... preprocessing ...
    
    if model_type == "logistic_regression":
        model, info = train_log_reg_model(X_train, y_train)
    elif model_type == "random_forest":
        model, info = train_random_forest_model(X_train, y_train)
```

### Adding a New Metric

1. **Edit `evaluation.py`** in `_compute_classification_metrics()`:

```python
from sklearn.metrics import matthews_corrcoef

def _compute_classification_metrics(y_true, y_pred, y_scores=None):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1"] = f1_score(y_true, y_pred)
    # Add new metric
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    return metrics
```

2. **No other changes needed.** The metric automatically appears in pipeline output and evaluation reports.

## Team

- Samuel Fraley
- Tizian Schenk
- Corneel Moons

## Acknowledgments

- Data from [nflfastR](https://www.nflfastr.com/)
- Course: Computing for Data Science, Barcelona School of Economics
