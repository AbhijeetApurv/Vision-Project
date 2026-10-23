"""
PLANE 1B: FastAPI Over-The-Air (OTA) Model & Heuristic Server.
Serves /v1/manifest and /models/{file_name} for Android Edge-AI clients.
"""

import json
from pathlib import Path
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

MODELS_DIR = Path(__file__).parent / "models"

app = FastAPI(
    title="Vision-Project OTA Model & Heuristic Server",
    version="1.0.0",
    description="Serves quantized LiteRT (.tflite) models and dynamic hazard heuristics.",
)


class HeuristicsConfig(BaseModel):
    confidence_threshold: float
    iou_threshold: float
    cooldown_ms: int
    critical_area_ratio: float
    priority_classes: List[str]


class ModelManifest(BaseModel):
    schema_version: int
    model_version: str
    published_at: str
    model_file: str
    sha256: str
    size_bytes: int
    input_size: List[int]
    quantization: str
    heuristics: HeuristicsConfig


@app.get("/healthz")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/manifest", response_model=ModelManifest)
def get_manifest() -> ModelManifest:
    manifest_path = MODELS_DIR / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="manifest.json not found in models directory.")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return ModelManifest(**data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Invalid manifest format: {exc}") from exc


@app.get("/models/{file_name}")
def download_model(file_name: str) -> FileResponse:
    if "/" in file_name or "\" in file_name or ".." in file_name:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    model_path = MODELS_DIR / file_name
    if not model_path.exists() or not model_path.is_file():
        raise HTTPException(status_code=404, detail=f"Model artifact '{file_name}' not found.")
    return FileResponse(
        path=model_path,
        media_type="application/octet-stream",
        filename=file_name,
    )
