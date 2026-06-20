
from fastapi import FastAPI,Request, status
from .service import inference
from .schemas import PredictionRequest,PredictionResponse,ErrorResponse
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

app=FastAPI()


@app.exception_handler(RequestValidationError)
def request_val_err(request:Request,exc:RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="ValidationError",
            details=str(exc.errors())
        ).model_dump()
    )


@app.post("/predict")
def predict(data:PredictionRequest):
    features = [
        data.age,
        data.days_as_customer,
        data.total_payment_amount,
        data.total_payments_count,
        data.avg_payment_amount,
        data.days_since_last_activity,
        data.total_activity_duration,
        data.avg_activity_duration
    ]
    result=inference(features)

    return PredictionResponse(
        churn_probability=result[0],
        churn_prediction=result[1],
        status="success"
    )
 

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}  