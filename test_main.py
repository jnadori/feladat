from fastapi.testclient import TestClient
from main import app
import pytest


client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() ==  {"message": "Welcome to Your Sentiment Classification FastAPI"}
    
def test_predict_salary_high():
    r = client.post("/predict", json = {
  "model_path": "model/model.pkl",
  "input_data": "{'age': 50,'workclass': 'Self-emp-not-inc','fnlgt': 83311,'education': 'Bachelors','education-num': 13,'marital-status': 'Married-civ-spouse','occupation': 'Exec-managerial','relationship': 'Husband','race': 'White','sex': 'Male','capital-gain': 0,'capital-loss': 0,'hours-per-week': 13,'native-country': 'United-States','salary': '<=50K'}"
})
    assert r.status_code == 200
    assert r.json() == {"salary": 0}
    
def test_predict_salary_low():
    r = client.post("/predict", json = {"model_path": "model/model.pkl",
  "input_data": "{'age': 44,'workclass': 'Private','fnlgt': 326232,'education': 'Bachelors','education-num': 13,'marital-status': 'Divorced','occupation': 'Exec-managerial','relationship': 'Unmarried','race': 'White','sex': 'Male','capital-gain': 0,'capital-loss': 2547,'hours-per-week': 50,'native-country': 'United-States','salary': '>50K'}"})
    assert r.status_code == 200
    assert r.json() == {"salary": 1}