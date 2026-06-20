from pydantic import BaseModel, field_validator
from fastapi import HTTPException

class ErrorResponse(BaseModel):
    error: str
    details: str | None = None

class PredictionRequest(BaseModel):
    age: int
    days_as_customer: int
    total_payment_amount: float
    total_payments_count: int
    avg_payment_amount: float
    days_since_last_activity: int
    total_activity_duration: float
    avg_activity_duration: float

    @field_validator('age')
    def check_age(cls, v):
        if v < 18 or v > 100:
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    error="ValidationError",
                    details="Age must be between 18 and 100"
                ).model_dump()
            )
        return v

class PredictionResponse(BaseModel):
    churn_probability: list[float]   
    churn_prediction: list[int]      
    status: str