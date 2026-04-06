# Score Predictor Training Job

A minimal ML training job that trains a Linear Regression model to predict exam scores based on hours studied, and deploys it to [TrueFoundry](https://www.truefoundry.com/) as a Job.

## Overview

This project extracts the training logic from the [basic-ml-model-with-fastapi](https://github.com/sreeharikmarar/basic-ml-model-with-fastapi) Jupyter notebook into standalone Python scripts, following the TrueFoundry Job deployment pattern.

### What the Model Does

Given hours studied (1–10), predict the exam score. The model learns a simple linear relationship:

```
score = 6.42 * hours_studied + 31.47
```

**Dataset** (inline, 10 samples):

| Hours Studied | Score |
|---------------|-------|
| 1             | 35    |
| 2             | 45    |
| 3             | 50    |
| 4             | 60    |
| 5             | 65    |
| 6             | 70    |
| 7             | 78    |
| 8             | 82    |
| 9             | 88    |
| 10            | 95    |

**Metrics** (on training data):
- MAE: 1.30
- R²: 0.9928

## Project Structure

```
.
├── train.py           # Model training script
├── deploy.py          # TrueFoundry Job deployment script
├── requirements.txt   # Python dependencies
└── README.md
```

## Prerequisites

- Python 3.11+
- [TrueFoundry CLI/SDK](https://docs.truefoundry.com/) (for deployment only)

## Local Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Training

```bash
python train.py
```

This will:
1. Load the hours_studied vs score dataset
2. Train a `LinearRegression` model using scikit-learn
3. Print evaluation metrics (coefficient, intercept, MAE, R²)
4. Save the trained model to `score_predictor.joblib`
5. Verify by reloading the model and running a test prediction

**Expected output:**

```
10 samples loaded
Coefficient (slope): 6.42
Intercept:           31.47
Formula:             score = 6.42 * hours + 31.47
MAE:  1.30
R²:   0.9928

Model saved to score_predictor.joblib
Verification: 5.0 hours -> predicted score: 63.6
```

## Deploy to TrueFoundry

### 1. Log in to TrueFoundry

```bash
tfy login
```

### 2. Deploy the Training Job

```bash
python deploy.py --workspace_fqn <your-workspace-fqn>
```

This deploys a Job named `score-predictor-train-job` that:
- Uploads the local source code to TrueFoundry
- Builds a Python 3.11 container with dependencies from `requirements.txt`
- Runs `python train.py` inside the container

### 3. Monitor

Check the job status on the TrueFoundry dashboard.

## Serving the Model

The saved `score_predictor.joblib` file is compatible with the [basic-ml-model-with-fastapi](https://github.com/sreeharikm/basic-ml-model-with-fastapi) project, which serves predictions via a FastAPI endpoint:

```
GET /predict?hours=5  →  {"hours_studied": 5.0, "predicted_score": 63.6}
```