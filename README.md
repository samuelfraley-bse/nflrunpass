# NFL Play Caller

Machine learning model to predict run vs pass plays in NFL games based on pre-snap situational features. Built using logistic regression on 2021-2023 play-by-play data.

## Project Structure

```
nflplaycaller/
├── src/nfl_run_pass/           # Core ML library
│   ├── __init__.py
│   ├── config.py               # Settings, paths, hyperparameters
│   ├── data_loading.py         # Load and filter play-by-play data
│   ├── preprocessing.py        # Train/test split, scaling
│   ├── features.py             # Feature engineering
│   ├── models.py               # Model training (logistic regression)
│   ├── evaluation.py           # Metrics and evaluation
│   ├── pipeline.py             # End-to-end training pipeline
│   └── tuning.py               # Hyperparameter tuning
│
├── api/                        # FastAPI backend
│   └── main.py                 # REST API for predictions
│
├── streamlit/                  # Frontend applications
│   ├── st_app.py              # Main predictor UI
│   ├── steamlit_btm.py        # "Beat the Model" game
│   └── streamlit_app.py       # Alternative UI version
│
├── models/                     # Saved artifacts
│   ├── log_reg_model.pkl      # Trained logistic regression model
│   ├── scaler.pkl             # Fitted StandardScaler
│   └── feature_cols.json      # Feature column names (ordered)
│
├── data/                       # Data files (gitignored)
│   └── raw/
│       └── pbp_2021_2023.csv  # NFL play-by-play dataset
│
├── tests/                      # Unit and integration tests (TO BE CREATED)
│   ├── test_data_loading.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_pipeline.py
│   └── test_api.py
│
├── run_pipeline.py             # Script to train model
├── sample_plays.csv            # Sample data for demos
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project metadata (uv/pip)
└── README.md                   # This file
```

## Module Overview

### Core Library (`src/nfl_run_pass/`)

The library follows a modular pipeline design:

1. **`config.py`** - Central configuration
   - Data paths and filtering rules
   - Feature definitions
   - Model hyperparameters
   - Train/test split settings

tuning.py – hyperparameter search
