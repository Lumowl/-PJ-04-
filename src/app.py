# FIX для joblib: нужно определить _do_preprocessing в __main__ перед загрузкой модели
import sys
import types

# Создаю или получаем модуль __main__
if '__main__' not in sys.modules:
    sys.modules['__main__'] = types.ModuleType('__main__')

# Импортирую функцию из preprocessing
try:
    from preprocessing import _do_preprocessing
    # Копирую её в __main__ модуль
    sys.modules['__main__']._do_preprocessing = _do_preprocessing
    print("✅ _do_preprocessing зарегистрирована в __main__")
except ImportError as e:
    print(f"⚠️ Не удалось импортировать _do_preprocessing: {e}")

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from datetime import datetime

from predictor import HousePricePredictor
from schemas import HouseInput, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальный объект предсказателя
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Обработчик жизненного цикла приложения"""
    global predictor
    try:
        predictor = HousePricePredictor()
        logger.info("✅ Модель загружена")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {e}")
        predictor = None
    yield


app = FastAPI(
    title="🏠 House Price Prediction API",
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "House Price Prediction API",
        "status": "running" if predictor and predictor.is_loaded else "error",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict"
        }
    }


@app.get("/health")
async def health():
    if predictor and predictor.is_loaded:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(house: HouseInput):
    if not predictor or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        house_data = house.model_dump(exclude_unset=True)
        price = predictor.predict(house_data)

        return PredictionResponse(
            success=True,
            predicted_price=price,
            predicted_price_formatted=f"${price:,.2f}",
            message="Предсказание успешно"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)