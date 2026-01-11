from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from model_service import ModelService
import joblib
from typing import List
import numpy as np

app = FastAPI()

mean_time = joblib.load('data/mean_time.pkl')

class HistoryItem(BaseModel):
    status: float
    duration: float # Raw duration
    level: int

class PredictRequest(BaseModel):
    history: conlist(HistoryItem, min_length=10, max_length=10)
    target_levels: List[int]

model_service = ModelService()

@app.get("/")
def read_root():
    return {"message": "Hello, Falcon!"}

@app.post("/predict")
def predict(data: PredictRequest):
    raw_sequence = []
    
    for item in data.history:
        if not (0 < item.level <= len(mean_time)):
            raise HTTPException(status_code=400, detail=f"Invalid history level: {item.level}")
        
        mean_t = float(mean_time[item.level - 1])
        
        residual = item.duration - mean_t
        
        raw_sequence.append([item.status, residual])

    seq_array = np.array(raw_sequence, dtype=np.float32)
    mean_vals = seq_array.mean(axis=0)
    std_vals = seq_array.std(axis=0)
    
    std_vals[std_vals == 0] = 1.0
    
    normalized_seq = (seq_array - mean_vals) / (std_vals + 1e-6)
    
    flattened_history = normalized_seq.flatten().tolist()
    batch_features = []
    for level in data.target_levels:
        if not (0 < level <= len(mean_time)):
            raise HTTPException(status_code=400, detail=f"Invalid target_level: {level}")

        mean_time_for_target = float(mean_time[level - 1])

        row = flattened_history + [mean_time_for_target]
        
        batch_features.append(row)

    if not batch_features:
        return {"predictions": ""}
        
    try:
        all_predictions = model_service.predict(batch_features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {str(e)}")

    return {"predictions": all_predictions.tolist()}