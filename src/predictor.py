
import os
import sys

# ФИКС ДЛЯ JOBLIB: создаю функцию в __main__ модуле
if '__main__' not in sys.modules:
    sys.modules['__main__'] = types.ModuleType('__main__')

# Импортирую функцию из preprocessing
try:
    from preprocessing import _do_preprocessing
    # Копирую её в __main__ модуль
    sys.modules['__main__']._do_preprocessing = _do_preprocessing
except ImportError:
    # Если файл не найден, создаю заглушку
    def _do_preprocessing_stub(df):
        return df
    sys.modules['__main__']._do_preprocessing = _do_preprocessing_stub

import joblib
import pandas as pd
import logging
from typing import Dict, Any, List

# Создаю fake модуль __main__
if not hasattr(sys.modules['__main__'], '_do_preprocessing'):
    # Импортирую из preprocessing
    from preprocessing import _do_preprocessing
    # Присваиваю в __main__
    setattr(sys.modules['__main__'], '_do_preprocessing', _do_preprocessing)

logger = logging.getLogger(__name__)

class HousePricePredictor:
    """Простой класс для предсказания цен"""

    def __init__(self, model_path: str = None):
        """
        Инициализация предсказателя с автопоиском модели
        """
        # Автоматически нахожу модель
        if model_path is None:
            model_path = self._find_model()

        logger.info(f"Попытка загрузки модели из: {model_path}")

        if not os.path.exists(model_path):
            error_msg = self._get_error_message(model_path)
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            self.model = joblib.load(model_path)
            self.is_loaded = True
            self.model_path = model_path
            logger.info("✅ Модель загружена успешно")

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            self.is_loaded = False
            raise

    def _find_model(self) -> str:
        """Автоматический поиск модели"""
        # Текущая директория где запущен код
        current_dir = os.getcwd()

        # Возможные пути к модели (в порядке приоритета)
        possible_paths = [
            # 1. В папке models рядом с текущей директорией
            os.path.join(current_dir, "models", "housing_model.pkl"),
            # 2. В корне проекта (если запускаем из корня)
            os.path.join(current_dir, "housing_model.pkl"),
            # 3. В папке housing_price_service/models
            os.path.join(current_dir, "housing_price_service", "models", "housing_model.pkl"),
            # 4. Рядом с этим файлом (если predictor.py в src/)
            os.path.join(os.path.dirname(__file__), "..", "models", "housing_model.pkl"),
            os.path.join(os.path.dirname(__file__), "housing_model.pkl"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"✅ Модель найдена: {path}")
                return path

        # Если не нашли, возвращаю наиболее вероятный путь для ошибки
        return os.path.join(current_dir, "models", "housing_model.pkl")

    def _get_error_message(self, model_path: str) -> str:
        """Информативное сообщение об ошибке"""
        return f"""
        ❌ Файл модели не найден!

        Ожидаемый путь: {model_path}

        Что сделать:
        1. Убедитесь, что файл housing_model.pkl существует
        2. Положите его в одну из папок:
           - {os.path.join(os.getcwd(), "models")}/
           - {os.path.join(os.getcwd(), "housing_price_service", "models")}/
           - {os.getcwd()}/

        Текущая директория: {os.getcwd()}
        """

    def predict(self, house_data: Dict[str, Any]) -> float:
        if not self.is_loaded:
            raise ValueError("Модель не загружена")

        try:
            df = pd.DataFrame([house_data])
            prediction = self.model.predict(df)[0]
            logger.info(f"Предсказание: ${prediction:,.2f}")
            return float(prediction)

        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            raise

    def predict_batch(self, houses_data: List[Dict[str, Any]]) -> List[float]:
        if not self.is_loaded:
            raise ValueError("Модель не загружена")

        try:
            df = pd.DataFrame(houses_data)
            predictions = self.model.predict(df)
            return [float(p) for p in predictions]

        except Exception as e:
            logger.error(f"Ошибка пакетного предсказания: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        info = {
            "is_loaded": self.is_loaded,
            "model_type": type(self.model).__name__ if self.is_loaded else None,
        }

        if self.is_loaded and hasattr(self.model, 'named_steps'):
            steps = {}
            for name, step in self.model.named_steps.items():
                steps[name] = type(step).__name__
            info["pipeline_steps"] = steps

        return info

    def health_check(self) -> bool:
        if not self.is_loaded:
            return False

        try:
            test_data = {
                "status": "active",
                "propertyType": "single_family",
                "beds": 3,
                "baths": 2.5,
                "sqft": 1800,
            }
            _ = self.predict(test_data)
            return True
        except Exception:
            return False