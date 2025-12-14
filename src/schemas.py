from pydantic import BaseModel, Field, validator, ConfigDict
from typing import Optional, List, Dict, Any
from enum import Enum
import re


class StatusEnum(str, Enum):
    """Статус объекта недвижимости"""
    ACTIVE = "Active"
    FOR_SALE = "for sale"
    PENDING = "pending"
    SOLD = "sold"
    CONTINGENT = "contingent"
    UNKNOWN = "unknown"


class PropertyTypeEnum(str, Enum):
    """Тип недвижимости"""
    SINGLE_FAMILY = "Single Family Home"
    SINGLE_FAMILY_HOME = "single-family home"
    CONDO = "condo"
    TOWNHOUSE = "townhouse"
    MULTI_FAMILY = "multi-family"
    LOT_LAND = "lot/land"
    OTHER = "other"

# Вложенные модели
class FactItem(BaseModel):
    factLabel: str
    factValue: str

class HomeFacts(BaseModel):
    atAGlanceFacts: List[FactItem]

class SchoolData(BaseModel):
    Distance: List[str]
    Grades: List[str]

class School(BaseModel):
    rating: List[str]
    data: SchoolData
    name: List[str]

class HouseInput(BaseModel):
    """Схема входных данных на основе сырых столбцов"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "status": "Active",
                "private pool": "yes",
                "propertyType": "Single Family Home",
                "street": "240 Heather Ln",
                "baths": "3.5",
                "homeFacts": {
                    "atAGlanceFacts": [
                        {"factValue": "1920", "factLabel": "Year built"},
                        {"factValue": "2025", "factLabel": "Remodeled year"},
                        {"factValue": "Forced Air", "factLabel": "Heating"},
                        {"factValue": "Central", "factLabel": "Cooling"},
                        {"factValue": "yes", "factLabel": "Parking"},
                        {"factValue": "680 sqft", "factLabel": "lotsize"},
                        {"factValue": "$233/sqft", "factLabel": "Price/sqft"}
                    ]
                },
                "fireplace": "Gas Logs",
                "city": "Southern Pines",
                "schools": [
                    {
                        "rating": ["2", "2", "4"],
                        "data": {
                            "Distance": ["5.6 mi", "5.6 mi", "6.8 mi"],
                            "Grades": ["PK-4", "5-6", "9-12"]
                        },
                        "name": [
                            "Roosevelt Elementary School",
                            "Lincoln Intermediate School",
                            "Mason City High School"
                        ]
                    }
                ],
                "sqft": "2900",
                "zipcode": "28387",
                "beds": "4",
                "state": "NC",
                "stories": "2.0",
                "mls-id": "",
                "PrivatePool": "",
                "MlsId": "611019"
            }
        }
    )

    status: Optional[str] = Field(None, description="Статус объекта недвижимости")

    private_pool: Optional[str] = Field(
        None,
        alias="private pool",  # JSON: "private pool", Python: private_pool
        description="Частный бассейн"
    )

    propertyType: Optional[str] = Field(None, description="Тип недвижимости")
    street: Optional[str] = Field(None, description="Улица")
    baths: Optional[str] = Field(None, description="Количество ванных")
    homeFacts: Optional[HomeFacts] = Field(None, description="Факты о доме")
    fireplace: Optional[str] = Field(None, description="Камин")
    city: Optional[str] = Field(None, description="Город")
    schools: Optional[List[School]] = Field(None, description="Информация о школах")
    sqft: Optional[str] = Field(None, description="Площадь")
    zipcode: Optional[str] = Field(None, description="Почтовый индекс")
    beds: Optional[str] = Field(None, description="Количество спален")
    state: Optional[str] = Field(None, description="Штат", min_length=2, max_length=2)
    stories: Optional[str] = Field(None, description="Количество этажей")

    # Поле с дефисом - используем alias
    mls_id: Optional[str] = Field(
        None,
        alias="mls-id",  # JSON: "mls-id", Python: mls_id
        description="MLS ID"
    )

    PrivatePool: Optional[str] = Field(None, description="Частный бассейн (дубликат)")
    MlsId: Optional[str] = Field(None, description="MLS ID (дубликат)")


class PredictionResponse(BaseModel):
    """Схема ответа с предсказанием"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "predicted_price": 418000.0,
                "predicted_price_formatted": "$418,000",
                "message": "Предсказание успешно выполнено"
            }
        }
    )

    success: bool = Field(..., description="Успешно ли выполнено предсказание")
    predicted_price: float = Field(..., description="Предсказанная цена в долларах", example=418000.0)
    predicted_price_formatted: str = Field(..., description="Отформатированная цена", example="$418,000")
    message: Optional[str] = Field(None, description="Дополнительное сообщение")


# class BatchPredictionRequest(BaseModel):
#     """Схема для пакетного предсказания"""
#     model_config = ConfigDict(
#         json_schema_extra={
#             "example": {
#                 "houses": [
#                     {
#                         "status": "Active",
#                         "propertyType": "Single Family Home",
#                         "beds": "4",
#                         "baths": "3.5",
#                         "sqft": "2900",
#                         "state": "NC"
#                     },
#                     {
#                         "status": "for sale",
#                         "propertyType": "single-family home",
#                         "beds": "3 Beds",
#                         "baths": "3 Baths",
#                         "sqft": "1,947 sqft",
#                         "state": "WA"
#                     }
#                 ]
#             }
#         }
#     )
#
#     houses: List[HouseInput] = Field(..., description="Список домов для предсказания")
#
#
# class BatchPredictionResponse(BaseModel):
#     """Схема ответа для пакетного предсказания"""
#     model_config = ConfigDict(
#         json_schema_extra={
#             "example": {
#                 "success": True,
#                 "predictions": [418000.0, 310000.0],
#                 "count": 2,
#                 "message": "Пакетное предсказание успешно выполнено"
#             }
#         }
#     )
#
#     success: bool = Field(..., description="Успешно ли выполнено предсказание")
#     predictions: List[float] = Field(..., description="Список предсказанных цен")
#     count: int = Field(..., description="Количество предсказаний")
#     message: Optional[str] = Field(None, description="Дополнительное сообщение")


class HealthResponse(BaseModel):
    """Схема ответа для проверки здоровья"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "timestamp": "2024-01-15T10:30:00"
            }
        }
    )

    status: str = Field(..., description="Статус сервиса", example="healthy")
    model_loaded: bool = Field(..., description="Загружена ли модель", example=True)
    timestamp: str = Field(..., description="Время проверки")
