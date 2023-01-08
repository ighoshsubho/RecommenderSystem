from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict_pipeline
from app.model import __version__ as model_version


app = FastAPI()


class data(BaseModel):
    movie_id: int

class PredictionOut(BaseModel):
    movies_result: list


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict")
def predict(payload: data):
    movies_result = predict_pipeline(payload)
    return {"Result": movies_result}