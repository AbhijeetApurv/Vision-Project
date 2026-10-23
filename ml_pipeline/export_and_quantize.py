"""
PLANE 1A: Exports YOLOv8n to INT8 TFLite (LiteRT), computes SHA-256 hash,
and generates the OTA manifest.json file.
"""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from ultralytics import YOLO


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def export_and_build_manifest(
    weights_path: str,
    output_dir: Path,
    model_version: str,
    imgsz: int = 320,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights_path)

    print(f"[Plane 1A] Exporting {weights_path} to INT8 TFLite (imgsz={imgsz})...")
    exported_path_str = model.export(format="tflite", int8=True, imgsz=imgsz)
    exported_path = Path(exported_path_str)

    target_model_name = f"yolov8n_int8_v{model_version.replace('.', '_')}.tflite"
    target_model_path = output_dir / target_model_name
    shutil.copy2(exported_path, target_model_path)

    sha256_Digest = compute_sha256(target_model_path)
    file_size_bytes = target_model_path.stat().st_size

    manifest = {
        "schema_version": 1,
        "model_version": model_version,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "model_file": target_model_name,
        "sha256": sha256_Digest,
        "size_bytes": file_size_bytes,
        "input_size": [imgsz, imgsz, 3],
        "quantization": "INT8",
        "heuristics": {
            "confidence_threshold": 0.45,
            "iou_threshold": 0.50,
            "cooldown_ms": 3500,
            "critical_area_ratio": 0.18,
            "priority_classes": ["person", "car", "bicycle", "motorcycle", "bus", "truck", "stop sign"],
        },
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[Plane 1A] Saved model to {target_model_path} ({file_size_bytes} bytes)")
    print(f"[Plane 1A] SHA-256: {sha256_Digest}")
    print(f"[Plane 1A] Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export INT8 YOLOv8n and build OTA manifest.")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to input PyTorch weights")
    parser.add_argument("--output-dir", type=Path, default=Path("../ota_server/models"), help="Staging folder")
    parser.add_argument("--version", type=str, default="1.0.0", help="Semantic version for manifest")
    parser.add_argument("--imgsz", type=int, default=320, help="Input square image resolution")
    args = parser.parse_args()

    export_and_build_manifest(
        weights_path=args.weights,
        output_dir=args.output_dir,
        model_version=args.version,
        imgsz=args.imgsz,
    )
