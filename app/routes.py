from fastapi import APIRouter, HTTPException
from model.predict import ModelPredictor
from model.monitor import monitor_prediction_time

router = APIRouter()
predictor = ModelPredictor(
    "model/text_generation_model.h5",
    "model/tokenizer.pkl",
)
monitor = monitor_prediction_time()


@router.get("/predict/")
@monitor
def predict(text: str):
    try:
        result = predictor.predict(text.strip())
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
