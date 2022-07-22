# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union 
from starter.ml.data import process_data
from starter.ml.model import *
from starter import train_model as tm
from joblib import load
import pandas as pd 

from fastapi import FastAPI, Header
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel


# Initialize an instance of FastAPI
app = FastAPI()


class modelset(BaseModel):
    data_path : str
    model_path : str
    feature : str

class predictset(BaseModel):
    model_path : str
    input_data : str


@app.get("/")
def root():
    return {"message": "Welcome to Your Sentiment Classification FastAPI"}

@app.post("/start_model")
def start_model(settings: modelset):
    precision, recall, fbeta = tm.get_model(settings.data_path, settings.model_path, settings.feature)
    return {"precision": precision,
           "recall": recall,
           "f1 score": fbeta}    
@app.post("/predict")
def predict(settings: predictset):
    df = pd.DataFrame(eval(settings.input_data),index=[0])
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",]
    lb=load('model/lb.pkl')
    encoder=load('model/encoder.pkl')
    model=load(settings.model_path)
    X_infer, y_infer, encoder, lb = process_data(
    df, categorical_features=cat_features, label="salary", training=False, encoder=encoder,lb=lb)
    predictos=int(inference(model,X_infer)[0])
    return {"salary" : predictos}