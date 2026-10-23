# Vision-Project

An end-to-end Edge-AI Hazard Detection & Over-The-Air (OTA) Model Delivery system divided into three operational planes:

- **Plane 1A (`ml_pipeline/`)**: Cloud Colab Training & LiteRT INT8 Quantization pipeline (`YOLOv8n` export, SHA-256 integrity hashing, manifest generation, and host-side latency/shape verification).
- **Plane 1B (`ota_server/`)**: FastAPI Over-The-Air Model & Heuristic Server (`Dockerfile` ready for Google Cloud Run, serving `/v1/manifest` and `/models/{file_name}`).
- **Plane 2 & 3 (`android_app/`)**: Android Edge-AI Client featuring 3-tier model storage fallback (`AdaptiveModelManager.kt`) with atomic `ReentrantReadWriteLock` hot-swapping and a 30 FPS CameraX `ImageAnalysis.Analyzer` (`HazardAnalyzer.kt`) with anti-fatigue spatial audio and haptic feedback.

## Project Structure

```text
Vision-Project/
├── .gitignore                        # Excludes heavy .tflite/.pt binaries & Android build cache
├── README.md                         # System architecture & setup documentation
├── ml_pipeline/                      # PLANE 1A: Cloud Colab Training & LiteRT Quantization
│   ├── requirements.txt              # Ultralytics, ai-edge-litert, onnx
│   ├── export_and_quantize.py        # Exports INT8 YOLOv8n + computes SHA-256 + builds manifest
│   └── verify_litert_model.py        # Sanity-checks tensor shapes & benchmarks host latency
├── ota_server/                       # PLANE 1B: FastAPI Over-The-Air Model & Heuristic Server
│   ├── requirements.txt              # FastAPI, Uvicorn, Pydantic
│   ├── Dockerfile                    # Ready for Google Cloud Run deployment
│   ├── main.py                       # Serves /v1/manifest and /models/{file_name}
│   └── models/                       # Local staging folder for exported .tflite & manifest.json
└── android_app/                      # PLANE 2 & 3: Android Edge-AI Client
    └── app/src/main/
        ├── assets/
        │   └── manifest.json         # Tier-1 factory fallback configuration
        └── java/com/visionproject/
            ├── ml/
            │   └── AdaptiveModelManager.kt  # 3-Tier storage, SHA-256 check & ReadWriteLock swap
            └── pipeline/
                └── HazardAnalyzer.kt        # CameraX 30 FPS analyzer + Anti-fatigue audio/haptics
```

## Quick Start

### 1. Export & Quantize Model (Plane 1A)
```bash
cd ml_pipeline
pip install -r requirements.txt
python export_and_quantize.py --weights yolov8n.pt --output-dir ../ota_server/models
python verify_litert_model.py --model ../ota_server/models/yolov8n_int8.tflite
```

### 2. Run OTA Server (Plane 1B)
```bash
cd ota_server
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080
```

### 3. Android Edge-AI Client (Plane 2 & 3)
Open `android_app/` in Android Studio, ensure the bundled fallback model and `assets/manifest.json` are present, and deploy to a physical Android device.
