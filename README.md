# Model Regression Detector

### AI-Powered ML Model Reliability & Regression Detection Platform

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-Frontend-3178C6?logo=typescript)](https://www.typescriptlang.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-4169E1?logo=postgresql)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://www.docker.com/)

> Detect model performance regressions before they reach production — then use AI to investigate why they happened.

**Live Demo:** https://model-regression-detector-frontend.onrender.com/

---

## Overview

Machine learning models can silently become worse after a new model version, preprocessing change, data shift, or deployment.

Traditional monitoring often tells you that something changed.

**Model Regression Detector goes further.**

It provides an end-to-end workflow for evaluating model versions, detecting performance regressions, analyzing data drift and affected segments, and using an AI investigation layer to help identify possible root causes.

The platform supports:

* Binary and multiclass classification
* Regression
* Time-series forecasting
* Baseline vs candidate model comparison
* Configurable regression thresholds
* Data drift detection
* Segment-level analysis
* Automated evaluation reports
* LLM-powered investigation
* Historical incident retrieval
* Embeddings and vector search
* AI-generated recommendations
* REST API
* Interactive web dashboard
* Production deployment

---

## Why This Project?

Imagine a bank deploying a new credit-risk model.

### Model V1

```text
ROC-AUC: 0.91
F1:      0.90
```

### Model V2

```text
ROC-AUC: 0.88
F1:      0.84
```

The new model is deployed because it passed basic validation.

But it performs worse.

The important question is not only:

> "Did the model get worse?"

It is:

> "How much worse did it get, where did it get worse, and why?"

This project is designed to answer those questions.

---

# Core Workflow

```text
                MODEL ARTIFACTS
                      |
        +-------------+-------------+
        |                           |
        v                           v
   BASELINE MODEL             CANDIDATE MODEL
        |                           |
        +-------------+-------------+
                      |
                      v
              EVALUATION ENGINE
                      |
                      v
             PERFORMANCE METRICS
                      |
                      v
           BASELINE vs CANDIDATE
                      |
                      v
             REGRESSION DETECTOR
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
        DRIFT      SEGMENTS    REPORT
          |           |           |
          +-----------+-----------+
                      |
                      v
              AI INVESTIGATION
                      |
          +-----------+-----------+
          |                       |
          v                       v
   HISTORICAL INCIDENTS       RAG / RETRIEVAL
          |                       |
          +-----------+-----------+
                      |
                      v
             LLM INVESTIGATION
                      |
                      v
              RECOMMENDATIONS
```

---

# Key Features

## 1. Model Regression Detection

Compare a baseline model against a candidate model using task-specific
evaluation metrics.

Example:

```text
Baseline F1      = 0.91
Candidate F1     = 0.86
Regression       = 0.05
Allowed          = 0.02

Result:
REGRESSION DETECTED
```

The comparison logic understands whether a metric is:

```text
Higher is better
```

or:

```text
Lower is better
```

This prevents incorrect comparisons such as treating an increase
in MAE as an improvement.

---

## 2. Multi-Task ML Evaluation

The platform is designed around task-specific evaluation.

### Classification

Supported metrics include:

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC
* Average Precision
* Log Loss

### Regression

Supported metrics include:

* MAE
* MSE
* RMSE
* R²
* MAPE

### Time Series

Supports chronological evaluation of forecasting models and
forecast-specific performance analysis.

---

# 3. Configurable Regression Thresholds

Regression thresholds are configurable rather than hard-coded.

Example:

```yaml
classification:
  accuracy:
    direction: higher_is_better
    max_regression: 0.02

  precision:
    direction: higher_is_better
    max_regression: 0.02

  recall:
    direction: higher_is_better
    max_regression: 0.02

  f1:
    direction: higher_is_better
    max_regression: 0.02

  roc_auc:
    direction: higher_is_better
    max_regression: 0.01

  log_loss:
    direction: lower_is_better
    max_regression: 0.03
```

This makes the system adaptable to different business requirements.

---

# 4. Data Drift Detection

Model performance can degrade because the data distribution changed.

For example:

```text
Training income distribution
        |
        v
Mostly 30K - 80K


Production income distribution
        |
        v
Mostly 80K - 200K
```

The platform analyzes distribution changes separately from model
performance.

Example PSI interpretation:

```text
PSI < 0.10       Low
0.10 - 0.25      Moderate
> 0.25           Significant
```

These thresholds are configurable monitoring policies rather than
universal rules.

---

# 5. Segment-Level Analysis

Overall metrics can hide serious failures.

Example:

```text
Overall F1       = 0.90

Age 18-25        = 0.62
Age 25-40        = 0.87
Age 40-60        = 0.94
```

The overall score looks acceptable.

But the younger customer segment is experiencing a major regression.

The detector can analyze important segments to identify these hidden
failures.

---

# 6. AI-Powered Investigation

The deterministic ML engine answers:

> "Did the model regress?"

The AI layer helps answer:

> "Why might it have regressed?"

The investigation can consider:

* metric changes
* drift results
* segment-level regressions
* model versions
* evaluation context
* historical incidents
* retrieved knowledge

Example:

```text
Regression detected

F1:
0.91 -> 0.84

Income PSI:
0.31

Worst affected segment:
Age 18-25

Historical incident:
Similar regression occurred after preprocessing changes

AI investigation:

Possible causes:
1. Distribution shift in income
2. Segment-specific degradation
3. Possible preprocessing mismatch

Recommended actions:
- inspect income feature distribution
- compare preprocessing artifacts
- evaluate the affected segment independently
- compare against previous incident
```

---

# 7. RAG and Historical Incident Memory

The system can use embeddings and vector search to retrieve
similar historical incidents.

```text
CURRENT INCIDENT
       |
       v
    EMBEDDING
       |
       v
   VECTOR SEARCH
       |
       v
SIMILAR HISTORICAL INCIDENTS
       |
       v
       LLM
       |
       v
 INVESTIGATION
```

This allows the platform to turn previous ML failures into reusable
organizational knowledge.

---

# 8. Separation of Deterministic ML and Generative AI

A core architectural principle is:

```text
DETERMINISTIC CODE
        |
        v
"REGRESSION DETECTED"
```

Then:

```text
AI
 |
 v
"Here are evidence-backed possible causes."
```

The LLM is not responsible for calculating the official metrics.

This makes the system more reliable because exact numerical decisions
remain deterministic while the LLM is used for reasoning, investigation,
retrieval and recommendations.

---

# Architecture

```text
                         USER
                           |
                           v
                  REACT DASHBOARD
                           |
                           v
                 TYPESCRIPT + VITE
                           |
                           v
                      FASTAPI
                           |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
   MODEL EVALUATION   DRIFT DETECTION    POSTGRESQL
        |                  |                  |
        v                  v               PGVECTOR
 BASELINE/CANDIDATE    SEGMENT ANALYSIS      |
        |                  |                 v
        +------------------+--------------> RAG
                           |                 |
                           v                 v
                    REGRESSION REPORT       LLM
                                             |
                                             v
                                      AI INVESTIGATION
                                             |
                                             v
                                      RECOMMENDATIONS
```

---

# Technology Stack

## Backend

* Python
* FastAPI
* Uvicorn
* Pydantic
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* Joblib

## Machine Learning

* Classification
* Regression
* Time-series forecasting
* Model comparison
* Performance regression detection
* Data drift analysis
* Segment analysis

## Generative AI

* Large Language Models
* OpenRouter
* Embeddings
* Semantic retrieval
* RAG
* AI investigation
* AI recommendations

## Data / Storage

* PostgreSQL
* pgvector

## Frontend

* React
* TypeScript
* Vite

## Infrastructure

* Docker
* Render
* GitHub

---

# Project Structure

```text
model-regression-detector/
│
├── config/
│   └── thresholds.yaml
│
├── examples/
│   └── credit_risk/
│
├── frontend/
│
├── src/
│   └── mrd/
│       │
│       ├── __init__.py
│       ├── config.py
│       ├── schemas.py
│       │
│       ├── metrics/
│       │   ├── classification.py
│       │   ├── regression.py
│       │   └── forecasting.py
│       │
│       ├── evaluation/
│       │   ├── classification.py
│       │   ├── regression.py
│       │   ├── timeseries.py
│       │   ├── evaluator.py
│       │   ├── dataset_loader.py
│       │   ├── loader.py
│       │   └── model_input.py
│       │
│       ├── detection/
│       │
│       ├── drift/
│       │
│       ├── segments/
│       │
│       ├── reports/
│       │
│       ├── ai/
│       │
│       ├── api/
│       │   ├── app.py
│       │   └── schemas.py
│       │
│       └── pipeline/
│
├── tests/
│
├── pyproject.toml
└── requirements.txt
```

---

# Evaluation API

The main evaluation workflow is exposed through:

```http
POST /evaluations/run
```

The endpoint accepts:

```text
model_name
baseline_version
candidate_version
dataset_version
target_column
model_type
time_column
baseline_model
candidate_model
evaluation_dataset
```

The backend then:

```text
1. Validates the request
2. Stores uploaded artifacts temporarily
3. Loads baseline model
4. Loads candidate model
5. Loads evaluation dataset
6. Prepares model inputs
7. Evaluates baseline
8. Evaluates candidate
9. Compares metrics
10. Detects regression
11. Analyzes drift
12. Generates report
13. Returns structured JSON
```

---

# Example Regression Detection

Suppose the baseline produces:

```json
{
  "f1": 0.91,
  "roc_auc": 0.94,
  "log_loss": 0.21
}
```

The candidate produces:

```json
{
  "f1": 0.84,
  "roc_auc": 0.92,
  "log_loss": 0.27
}
```

The detector evaluates the direction of each metric.

```text
F1:
0.91 -> 0.84
Regression = 0.07

ROC-AUC:
0.94 -> 0.92
Regression = 0.02

Log Loss:
0.21 -> 0.27
Regression = 0.06
```

If the configured thresholds are exceeded, the candidate is flagged.

---

# Example Use Cases

## Financial Services

Monitor:

* credit-risk models
* fraud detection
* loan approval
* default prediction

## E-Commerce

Monitor:

* recommendation models
* customer churn
* conversion prediction
* demand forecasting

## Healthcare

Monitor:

* risk prediction
* diagnosis support models
* patient outcome prediction

## Operations

Monitor:

* demand forecasting
* price prediction
* inventory forecasting
* anomaly detection

---

# Engineering Principles

### 1. Evaluation Before Deployment

A candidate model should be evaluated against a trusted baseline.

### 2. Deterministic Metrics

Official regression decisions should be reproducible and
mathematically defined.

### 3. Explainability

A monitoring system should explain what changed and where.

### 4. AI as an Investigation Layer

LLMs should help investigate evidence rather than replace
deterministic evaluation.

### 5. Version Everything

Model versions and dataset versions should be tracked.

### 6. Production First

The system is designed around:

* APIs
* logging
* configuration
* deployment
* persistence
* monitoring
* error handling

---

# Local Development

## Clone

```bash
git clone https://github.com/Nehmyabiruk/model-regression-detector.git

cd model-regression-detector
```

## Create Environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

or install the project using the configured Python packaging setup.

## Start Backend

```bash
python -m uvicorn mrd.api.app:app --reload --port 8000
```

Backend:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## Start Frontend

From the frontend directory:

```bash
npm install
npm run dev
```

---

# Environment Variables

Example:

```env
DATABASE_URL=your_database_url
OPENROUTER_API_KEY=your_api_key
AI_MODEL=your_model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
FRONTEND_URL=http://localhost:5173
```

Never commit real secrets to the repository.

---

# Production Deployment

The project is structured for cloud deployment with:

```text
GitHub
   |
   v
Render
   |
   +---- Backend
   |
   +---- Frontend
   |
   +---- PostgreSQL
```

Live frontend:

https://model-regression-detector-frontend.onrender.com/

---

# Example: Credit Risk

The project includes a credit-risk example demonstrating how a
classification model can be evaluated.

Typical workflow:

```text
Credit Dataset
      |
      v
Feature Preparation
      |
      v
Preprocessing
      |
      v
XGBoost
      |
      v
Baseline Model
      |
      +--------------------+
                           |
                      Candidate Model
                           |
                           v
                    Model Comparison
                           |
                           v
                 Regression Detection
```

The evaluation pipeline preserves raw tabular DataFrames when a
model contains preprocessing so categorical features can be handled
correctly by the fitted pipeline.

---

# Future Roadmap

Potential future extensions include:

* CI/CD quality gates
* automated model promotion/rejection
* model registry integration
* statistical significance testing
* confidence intervals
* alerting
* Slack/email notifications
* experiment tracking
* richer model explainability
* automated retraining
* controlled rollback
* anomaly detection
* feature-level root-cause analysis
* LLM evaluation
* LLM behavioral regression testing
* multi-model monitoring
* model lineage
* audit logs
* human approval workflows
* self-healing ML workflows

---

# What This Project Demonstrates

This project goes beyond training a machine learning model.

It demonstrates experience with:

```text
Machine Learning
       +
ML Evaluation
       +
Model Monitoring
       +
Data Drift
       +
Model Regression Detection
       +
Time-Series Forecasting
       +
REST APIs
       +
React Applications
       +
PostgreSQL
       +
Vector Search
       +
LLMs
       +
RAG
       +
AI Agents / Investigation
       +
Docker
       +
Cloud Deployment
```

---

# Why It Matters

Most ML projects answer:

> "Can I train a model?"

Production ML systems must answer much harder questions:

> "Is the new model actually better?"

> "Did performance regress?"

> "Which users are affected?"

> "Did the data change?"

> "What caused the regression?"

> "Has this happened before?"

> "What should the engineering team do next?"

**Model Regression Detector is designed around those questions.**

---

# Author

**Nehmya Biruk**

AI/ML Engineer focused on building production-oriented machine
learning systems, LLM applications, RAG systems, and AI-powered
developer and ML infrastructure.

GitHub:
https://github.com/Nehmyabiruk

---

# License

Add your preferred open-source license here.

For example:

MIT License
