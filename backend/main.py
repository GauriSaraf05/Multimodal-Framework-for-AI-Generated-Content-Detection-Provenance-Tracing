
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.text import router as text_router

app = FastAPI(
    title="AI Content Detection API",
    version="0.1.0"
)

# Allow local React development servers
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):517\d+",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.include_router(text_router)

@app.get("/")
def home():
    return {
        "message": "AI Content Detection API is running"
    }

@app.get("/health")
def health():
    return {"status": "ok"}
