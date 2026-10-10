
from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.services.prediction_service import analyze_text

router = APIRouter(
    prefix="/api/text",
    tags=["Text Detection"]
)

class TextDetectionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100000)

@router.post("/detect")
def detect_text(request: TextDetectionRequest):
    return analyze_text(request.text)
