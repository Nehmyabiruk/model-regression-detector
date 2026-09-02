import io
import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression, LinearRegression

from mrd.api.app import app

client = TestClient(app)


def _model_bytes(model) -> io.BytesIO:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    return buf


def test_api_rejects_invalid_model_type() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression().fit(X, y)

    response = client.post(
        '/evaluations/run',
        data={
            'model_name': 'test-model',
            'baseline_version': '1.0.0',
            'candidate_version': '1.1.0',
            'dataset_version': 'v1',
            'target_column': 'target',
            'model_type': 'invalid_type',
        },
        files={
            'baseline_model': ('baseline.pkl', _model_bytes(model), 'application/octet-stream'),
            'candidate_model': ('candidate.pkl', _model_bytes(model), 'application/octet-stream'),
            'evaluation_dataset': ('eval.csv', b'feature,target\n1.0,0\n2.0,0\n3.0,1\n4.0,1\n', 'text/csv'),
        },
    )
    assert response.status_code == 400
    assert 'not supported' in response.json()['detail']


def test_api_timeseries_missing_time_column() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([10.0, 20.0, 30.0, 40.0])
    model = LinearRegression().fit(X, y)

    response = client.post(
        '/evaluations/run',
        data={
            'model_name': 'ts-model',
            'baseline_version': '1.0.0',
            'candidate_version': '1.1.0',
            'dataset_version': 'v1',
            'target_column': 'target',
            'model_type': 'timeseries',
        },
        files={
            'baseline_model': ('baseline.pkl', _model_bytes(model), 'application/octet-stream'),
            'candidate_model': ('candidate.pkl', _model_bytes(model), 'application/octet-stream'),
            'evaluation_dataset': ('eval.csv', b'date,feature,target\n2026-01-01,1.0,10.0\n', 'text/csv'),
        },
    )
    assert response.status_code == 400
    assert 'time_column is required' in response.json()['detail']


def test_api_regression_evaluation_run() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([10.0, 20.0, 30.0, 40.0])
    model = LinearRegression().fit(X, y)

    response = client.post(
        '/evaluations/run',
        data={
            'model_name': 'reg-model',
            'baseline_version': '1.0.0',
            'candidate_version': '1.1.0',
            'dataset_version': 'v1',
            'target_column': 'target',
            'model_type': 'regression',
        },
        files={
            'baseline_model': ('baseline.pkl', _model_bytes(model), 'application/octet-stream'),
            'candidate_model': ('candidate.pkl', _model_bytes(model), 'application/octet-stream'),
            'evaluation_dataset': ('eval.csv', b'feature,target\n1.0,10.0\n2.0,20.0\n3.0,30.0\n4.0,40.0\n', 'text/csv'),
        },
    )
    assert response.status_code == 200
    report = response.json()['report']
    assert report['task_type'] == 'regression'
    assert 'mae' in report['performance']
    assert 'r2' in report['performance']


def test_api_timeseries_evaluation_run() -> None:
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
    y = np.array([10.0, 20.0, 30.0, 40.0])
    model = LinearRegression().fit(X, y)

    response = client.post(
        '/evaluations/run',
        data={
            'model_name': 'ts-model',
            'baseline_version': '1.0.0',
            'candidate_version': '1.1.0',
            'dataset_version': 'v1',
            'target_column': 'target',
            'model_type': 'timeseries',
            'time_column': 'date',
        },
        files={
            'baseline_model': ('baseline.pkl', _model_bytes(model), 'application/octet-stream'),
            'candidate_model': ('candidate.pkl', _model_bytes(model), 'application/octet-stream'),
            'evaluation_dataset': ('eval.csv', b'date,feature,target\n2026-01-01,1.0,10.0\n2026-01-02,2.0,20.0\n2026-01-03,3.0,30.0\n2026-01-04,4.0,40.0\n', 'text/csv'),
        },
    )
    assert response.status_code == 200
    report = response.json()['report']
    assert report['task_type'] == 'timeseries'
    assert 'mae' in report['performance']
    assert 'r2' in report['performance']
