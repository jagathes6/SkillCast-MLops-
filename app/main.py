
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import time
from prometheus_client import Counter, Histogram, make_asgi_app

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

app = FastAPI(
    title="SkillCast API",
    description="Job Description Technology Role Prediction API",
    version="1.0.0"
)

REQUEST_COUNT = Counter(
    "skillcast_requests_total",
    "Total prediction requests"
)

PREDICTION_LATENCY = Histogram(
    "skillcast_prediction_latency_seconds",
    "Prediction latency"
)


class JobRequest(BaseModel):
    job_description: str


@app.get("/")
def root():
    return {
        "project": "SkillCast",
        "status": "running",
        "message": "Job Description Technology Role Prediction API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded"
    }


@app.post("/predict")
def predict(request: JobRequest):
    start = time.time()
    REQUEST_COUNT.inc()

    text = [request.job_description]
    vectorized = vectorizer.transform(text)
    prediction = model.predict(vectorized)[0]

    PREDICTION_LATENCY.observe(time.time() - start)

    return {
        "prediction": prediction
    }


metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

