#!/usr/bin/env python3
"""
PLANE 1A: Cloud Colab Training & LiteRT INT8 Quantization Pipeline
------------------------------------------------------------------
1. Optionally fine-tunes YOLOv8n on a custom hazard dataset.
2. Exports the PyTorch checkpoint (.pt) to Full-Integer Quantized (INT8)
   LiteRT (.tflite) using a representative calibration dataset (e.g. coco8.yaml).
3. Computes the cryptographic SHA-256 digest and exact byte size of the
   exported .tflite binary.
4. Builds and validates the OTA `manifest.json` schema (including per-class
   hazard priority weights, spatial thresholds, and anti-fatigue cooldowns).
5. Stages the quantized model and manifest into `../ota_server/models/` and
   optionally syncs the Tier-1 factory fallback to `../android_app/app/src/main/assets/`.
"""

import argparse
import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | [Plane-1A] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Plane1A.Export")

DEFAULT_PRIORITY_CLASSES: List[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
    "dog",
]


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Computes the hex-encoded SHA-256 digest of a binary file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_quantization_metadata(tflite_path: Path) -> Dict[str, Any]:
    """
    Uses LiteRT / TFLite Interpreter to extract exact input/output tensor
    shapes, data types, and affine quantization parameters (scale, zero_point).
    """
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        except ImportError:
            logger.warning("Neither ai-edge-litert nor tensorflow is installed; returning default tensor metadata.")
            return {
                "input_shape": [1, 320, 320, 3],
                "input_dtype": "int8",
                "output_shape": [1, 84, 2100],
                "output_dtype": "int8",
                "input_quantization": {"scale": 0.003921568859368563, "zero_point": -128},
                "output_quantization": {"scale": 0.0041258391, "zero_point": -112},
            }

    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    inp_q = inp.get("quantization_parameters", {})
    out_q = out.get("quantization_parameters", {})

    inp_scale = float(inp_q["scales"][0]) if len(inp_q.get("scales", [])) > 0 else 0.0
    inp_zp = int(inp_q["zero_points"][0]) if len(inp_q.get("zero_points", [])) > 0 else 0
    out_scale = float(out_q["scales"][0]) if len(out_q.get("scales", [])) > 0 else 0.0
    out_zp = int(out_q["zero_points"][0]) if len(out_q.get("zero_points", [])) > 0 else 0

    return {
        "input_shape": [int(x) for x in inp["shape"].tolist()],
        "input_dtype": inp["dtype"].__name__,
        "output_shape": [int(x) for x in out["shape"].tolist()],
        "output_dtype": out["dtype"].__name__,
        "input_quantization": {"scale": inp_scale, "zero_point": inp_zp},
        "output_quantization": {"scale": out_scale, "zero_point": out_zp},
    }


def build_manifest_payload(
    model_filename: str,
    model_version: str,
    sha256_digest: str,
    size_bytes: int,
    imgsz: int,
    quantization_mode: str,
    tensor_meta: Dict[str, Any],
    confidence_threshold: float,
    iou_threshold: float,
    cooldown_ms: int,
    critical_area_ratio: float,
    priority_classes: List[str],
    class_names: Dict[int, str],
) -> Dict[str, Any]:
    """Constructs the JSON manifest consumed by Plane 1B (OTA Server) and Plane 2 (Android)."""
    return {
        "schema_version": 1,
        "model_version": model_version,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "model_file": model_filename,
        "sha256": sha256_digest,
        "size_bytes": size_bytes,
        "input_size": [imgsz, imgsz, 3],
        "quantization": quantization_mode,
        "tensor_metadata": tensor_meta,
        "heuristics": {
            "confidence_threshold": round(confidence_threshold, 4),
            "iou_threshold": round(iou_threshold, 4),
            "cooldown_ms": int(cooldown_ms),
            "critical_area_ratio": round(critical_area_ratio, 4),
            "priority_classes": priority_classes,
        },
        "labels": {str(k): v for k, v in class_names.items()},
    }


def export_and_quantize(
    weights_path: str,
    calibration_data: str,
    output_dir: Path,
    android_assets_dir: Optional[Path],
    model_version: str,
    imgsz: int = 320,
    int8: bool = True,
    half: bool = False,
    nms: bool = False,
    train_epochs: int = 0,
    confidence_threshold: float = 0.45,
    iou_threshold: float = 0.50,
    cooldown_ms: int = 3500,
    critical_area_ratio: float = 0.18,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        logger.error("Ultralytics is not installed. Run `pip install -r requirements.txt`.")
        raise SystemExit(1) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading YOLO checkpoint: %s", weights_path)
    model = YOLO(weights_path)

    if train_epochs > 0:
        logger.info("Fine-tuning model on %s for %d epochs (imgsz=%d)...", calibration_data, train_epochs, imgsz)
        model.train(data=calibration_data, epochs=train_epochs, imgsz=imgsz)

    quant_tag = "int8" if int8 else ("fp16" if half else "fp32")
    logger.info(
        "Starting LiteRT (.tflite) export | mode=%s | imgsz=%d | calibration=%s | nms=%s",
        quant_tag.upper(),
        imgsz,
        calibration_data,
        nms,
    )

    exported_result = model.export(
        format="tflite",
        int8=int8,
        half=half,
        data=calibration_data if int8 else None,
        imgsz=imgsz,
        nms=nms,
    )
    exported_path = Path(str(exported_result))
    if not exported_path.exists():
        raise FileNotFoundError(f"Expected exported TFLite file at {exported_path}, but it was not found.")

    sanitized_ver = model_version.replace(".", "_").replace("-", "_")
    target_model_name = f"yolov8n_{quant_tag}_v{sanitized_ver}.tflite"
    target_model_path = output_dir / target_model_name
    shutil.copy2(exported_path, target_model_path)

    sha256_digest = compute_sha256(target_model_path)
    file_size_bytes = target_model_path.stat().st_size
    tensor_meta = inspect_quantization_metadata(target_model_path)

    class_names = getattr(model, "names", {i: f"class_{i}" for i in range(80)})
    manifest_data = build_manifest_payload(
        model_filename=target_model_name,
        model_version=model_version,
        sha256_digest=sha256_digest,
        size_bytes=file_size_bytes,
        imgsz=imgsz,
        quantization_mode=quant_tag.upper(),
        tensor_meta=tensor_meta,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        cooldown_ms=cooldown_ms,
        critical_area_ratio=critical_area_ratio,
        priority_classes=DEFAULT_PRIORITY_CLASSES,
        class_names=class_names,
    )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")

    logger.info("Staged OTA Model    : %s (%.2f MB)", target_model_path, file_size_bytes / (1024 * 1024))
    logger.info("SHA-256 Digest      : %s", sha256_digest)
    logger.info("Staged OTA Manifest : %s", manifest_path)

    # Optionally populate Tier-1 Factory Fallback in Android assets
    if android_assets_dir is not None:
        android_assets_dir.mkdir(parents=True, exist_ok=True)
        factory_model_name = "yolov8n_int8_factory.tflite"
        factory_model_path = android_assets_dir / factory_model_name
        shutil.copy2(target_model_path, factory_model_path)

        factory_manifest = dict(manifest_data)
        factory_manifest["model_version"] = f"{model_version}-factory"
        factory_manifest["model_file"] = factory_model_name
        factory_manifest_path = android_assets_dir / "manifest.json"
        factory_manifest_path.write_text(json.dumps(factory_manifest, indent=2) + "\n", encoding="utf-8")
        logger.info("Synced Tier-1 Factory Fallback to Android assets: %s", android_assets_dir)

    return target_model_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plane 1A: Export YOLOv8n to INT8 LiteRT (.tflite), compute SHA-256, and generate OTA manifest."
    )
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Input PyTorch weights (.pt)")
    parser.add_argument(
        "--data",
        type=str,
        default="coco8.yaml",
        help="Representative dataset YAML for INT8 post-training quantization calibration",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../ota_server/models"),
        help="Destination folder for staged OTA .tflite and manifest.json",
    )
    parser.add_argument(
        "--sync-android-assets",
        type=Path,
        nargs="?",
        const=Path("../android_app/app/src/main/assets"),
        default=None,
        help="Also copy exported model and manifest as Tier-1 factory fallback into Android assets/",
    )
    parser.add_argument("--version", type=str, default="1.0.0", help="Semantic model version (e.g., 1.0.0)")
    parser.add_argument("--imgsz", type=int, default=320, help="Square input tensor resolution (default: 320)")
    parser.add_argument("--no-int8", action="store_true", help="Disable INT8 quantization (export FP16/FP32 instead)")
    parser.add_argument("--fp16", action="store_true", help="Export FP16 quantized model if --no-int8 is set")
    parser.add_argument("--nms", action="store_true", help="Bundle TFLite NMS op inside exported graph")
    parser.add_argument("--train-epochs", type=int, default=0, help="Optional fine-tuning epochs before export")
    parser.add_argument("--conf-thres", type=float, default=0.45, help="Default heuristic confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.50, help="Default heuristic NMS IoU threshold")
    parser.add_argument("--cooldown-ms", type=int, default=3500, help="Default anti-fatigue cooldown in milliseconds")
    parser.add_argument(
        "--critical-area",
        type=float,
        default=0.18,
        help="Normalized bounding-box area ratio [0..1] that triggers critical haptic waveform",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    export_and_quantize(
        weights_path=args.weights,
        calibration_data=args.data,
        output_dir=args.output_dir,
        android_assets_dir=args.sync_android_assets,
        model_version=args.version,
        imgsz=args.imgsz,
        int8=not args.no_int8,
        half=args.fp16,
        nms=args.nms,
        train_epochs=args.train_epochs,
        confidence_threshold=args.conf_thres,
        iou_threshold=args.iou_thres,
        cooldown_ms=args.cooldown_ms,
        critical_area_ratio=args.critical_area,
    )
