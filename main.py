from fastapi import FastAPI
from pydantic import BaseModel
import joblib
app=FastAPI()
model=joblib.load("model.pkl")
class Input(BaseModel):
	x:float
@app.post("/predict")
def predict(data:Input):
	return {"prediction": model.predict([[data.x]])[0]}
